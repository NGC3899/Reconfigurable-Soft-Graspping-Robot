# -*- coding: utf-8 -*-
import numpy as np
import pyvista as pv
import numpy.linalg as LA
import torch
import torch.nn as nn
import joblib
import sys
from scipy.spatial.distance import cdist
import traceback
import time
# --- 新增：导入贝叶斯优化库 ---
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
except ImportError:
    print("错误：需要安装 scikit-optimize 库。请运行 'pip install scikit-optimize'")
    sys.exit()


# --- 1. ML 模型定义 ---
class MLPRegression(nn.Module):
    def __init__(self, input_dim, output_dim, h1, h2, h3):
        super(MLPRegression, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, h3), nn.Tanh(),
            nn.Linear(h3, output_dim) )
    def forward(self, x): return self.network(x)

# --- 2. ML 相关参数定义 ---
INPUT_DIM = 1; NODE_COUNT = 63; OUTPUT_DIM = NODE_COUNT * 3
HIDDEN_LAYER_1 = 128; HIDDEN_LAYER_2 = 256; HIDDEN_LAYER_3 = 128 # <<< 确认

# --- 3. 文件路径定义 ---
MODEL_PATH = 'best_mlp_model.pth'; X_SCALER_PATH = 'x_scaler.joblib'
Y_SCALER_PATH = 'y_scaler.joblib'; INITIAL_COORDS_PATH = 'initial_coordinates.txt'

# --- 4. 配置参数 ---
tray_radius = 50.0; tray_height = 10.0
tray_center = np.array([0.0, 0.0, -tray_height / 2.0])
finger_width = 10.0
sphere_radius = 15.0; num_sphere_points = 400
sphere_rotation_angle_z = np.radians(30)
sphere_translation = np.array([0, 0, sphere_radius + 35.0])
show_axes = True; finger_color = 'lightcoral'; tray_color = 'tan'; sphere_color = 'blue'
colliding_point_color = 'magenta'; intersection_point_color = 'yellow'; overlap_point_color = 'orange'
collision_threshold = 1.0; overlap_threshold = 1e-4
friction_coefficient = 0.5; eigenvalue_threshold = 1e-6
max_pressure = 40000.0
# 贝叶斯优化相关参数
N_CALLS_BO = 50
R_BOUNDS = (tray_radius * 0.3, tray_radius * 0.9) # <<< r 的搜索范围
P_BOUNDS = (0.0, max_pressure) # P 的搜索范围
# 可视化参数
contact_marker_radius = 1.0; contact_normal_length = 5.0; contact_plane_size = 4.0

# --- 5. 辅助函数 ---
# ... (所有辅助函数保持不变) ...
def create_rotation_matrix(axis, angle_rad):
    axis = np.asarray(axis).astype(float); axis /= LA.norm(axis)
    a = np.cos(angle_rad / 2.0); b, c, d = -axis * np.sin(angle_rad / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d; bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)], [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)], [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
def create_rotation_matrix_x(a): return create_rotation_matrix([1,0,0], a)
def create_rotation_matrix_y(a): return create_rotation_matrix([0,1,0], a)
def create_rotation_matrix_z(a): return create_rotation_matrix([0,0,1], a)
def create_translation_matrix(t): matrix=np.identity(4); matrix[:3, 3]=t; return matrix
def create_transformation_matrix(r, t): matrix=np.identity(4); matrix[:3, :3]=r if r is not None else np.identity(3); matrix[:3, 3]=t if t is not None else np.zeros(3); return matrix
def generate_sphere_points(radius, num_points):
    points = []; phi = np.pi * (3. - np.sqrt(5.))
    for i in range(num_points):
        z = 1 - (i / float(num_points - 1)) * 2; radius_at_z = np.sqrt(1 - z * z); theta = phi * i
        x = np.cos(theta) * radius_at_z; y = np.sin(theta) * radius_at_z; points.append((x, y, z));
    return np.array(points) * radius
def transform_points(points, matrix):
    if points.shape[1] != 3: raise ValueError("Points shape");
    if matrix.shape != (4, 4): raise ValueError("Matrix shape");
    num_pts = points.shape[0]; h_points = np.hstack((points, np.ones((num_pts, 1))))
    t_h = h_points @ matrix.T; return t_h[:, :3] / t_h[:, 3, np.newaxis]
def load_initial_coordinates(file_path, expected_nodes):
    try: coords = np.loadtxt(file_path, dtype=np.float32, usecols=(0, 1, 2))
    except FileNotFoundError: print(f"错误：找不到文件 {file_path}"); return None
    except Exception as e: print(f"加载初始坐标时出错: {e}"); return None
    if coords.shape == (expected_nodes, 3): print(f"成功加载 {coords.shape[0]} 个初始节点坐标。"); return coords
    else: print(f"错误：坐标形状 {coords.shape} 与预期 ({expected_nodes}, 3) 不符。"); return None
def create_faces_array(num_nodes_per_curve):
    faces = []; num_quads = num_nodes_per_curve - 1
    if num_quads <= 0: return np.array([], dtype=int)
    for i in range(num_quads): p1, p2=i, i+1; p3, p4=(i+1)+num_nodes_per_curve, i+num_nodes_per_curve; faces.append([4,p1,p2,p3,p4]);
    return np.hstack(faces)
def sort_points_spatially(points):
    if points is None or points.shape[0] < 2: return points
    num_points = points.shape[0]; sorted_indices = []; remaining_indices = list(range(num_points))
    start_node_index = np.argmin(points[:, 0]); current_index = start_node_index
    sorted_indices.append(current_index); remaining_indices.pop(remaining_indices.index(current_index))
    while remaining_indices:
        last_point = points[current_index, np.newaxis]; remaining_points = points[remaining_indices]
        distances = cdist(last_point, remaining_points)[0]
        nearest_neighbor_relative_index = np.argmin(distances)
        nearest_neighbor_absolute_index = remaining_indices[nearest_neighbor_relative_index]
        sorted_indices.append(nearest_neighbor_absolute_index); current_index = nearest_neighbor_absolute_index
        remaining_indices.pop(nearest_neighbor_relative_index)
    return points[sorted_indices]
def get_orthogonal_vectors(normal_vector):
    n = np.asarray(normal_vector).astype(float); norm_n = LA.norm(n)
    if norm_n < 1e-9: raise ValueError("Normal vector zero.")
    n /= norm_n
    if np.abs(n[0]) > 0.9: v_arbitrary = np.array([0., 1., 0.])
    else: v_arbitrary = np.array([1., 0., 0.])
    t1 = np.cross(n, v_arbitrary); norm_t1 = LA.norm(t1)
    if norm_t1 < 1e-9:
        v_arbitrary = np.array([0., 0., 1.]); t1 = np.cross(n, v_arbitrary); norm_t1 = LA.norm(t1)
        if norm_t1 < 1e-9:
             if np.abs(n[0]) > 0.9: t1 = np.array([0.,1.,0.]); t2_temp = np.array([0.,0.,1.])
             elif np.abs(n[1]) > 0.9: t1 = np.array([1.,0.,0.]); t2_temp = np.array([0.,0.,1.])
             else: t1 = np.array([1.,0.,0.]); t2_temp = np.array([0.,1.,0.])
             t1 /= LA.norm(t1); t2 = t2_temp / LA.norm(t2_temp); t2 = np.cross(n, t1); t2 /= LA.norm(t2); return t1, t2
    t1 /= norm_t1; t2_temp = np.cross(n, t1); norm_t2 = LA.norm(t2_temp)
    if norm_t2 < 1e-9: raise ValueError("Cannot compute tangent 2.")
    t2 = t2_temp / norm_t2; return t1, t2
def load_prediction_components(model_path, x_scaler_path, y_scaler_path, input_dim, output_dim, h1, h2, h3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"ML使用设备: {device}")
    model = MLPRegression(input_dim, output_dim, h1, h2, h3)
    try: model.load_state_dict(torch.load(model_path, map_location=device)); model.to(device); model.eval()
    except Exception as e: print(f"加载模型 {model_path} 时出错: {e}"); return None, None, None, None
    scaler_X, scaler_y = None, None
    try: scaler_X = joblib.load(x_scaler_path); print(f"X Scaler 加载成功。")
    except FileNotFoundError: print(f"警告: 未找到 X scaler '{x_scaler_path}'。")
    except Exception as e: print(f"加载 X Scaler '{x_scaler_path}' 时出错: {e}"); return None, None, None, None
    try: scaler_y = joblib.load(y_scaler_path); print(f"Y Scaler 加载成功。")
    except FileNotFoundError: print(f"警告: 未找到 Y scaler '{y_scaler_path}'。")
    except Exception as e: print(f"加载 Y Scaler '{y_scaler_path}' 时出错: {e}"); return None, None, None, None
    return model, scaler_X, scaler_y, device
def predict_displacements_for_pressure(model, scaler_X, scaler_y, device, pressure_value):
    if model is None: print("模型未加载。"); return None
    pressure_value = np.clip(pressure_value, P_BOUNDS[0], P_BOUNDS[1])
    input_p = np.array([[pressure_value]], dtype=np.float32)
    if scaler_X:
        try: input_p_scaled = scaler_X.transform(input_p)
        except Exception as e: print(f"X scaler 标准化出错: {e}"); return None
    else: input_p_scaled = input_p
    input_tensor = torch.tensor(input_p_scaled, dtype=torch.float32).to(device)
    predicted_original_scale = None
    with torch.no_grad():
        try:
            predicted_scaled_tensor = model(input_tensor); predicted_scaled = predicted_scaled_tensor.cpu().numpy()
            if scaler_y:
                try: predicted_original_scale = scaler_y.inverse_transform(predicted_scaled)
                except Exception as e: print(f"Y scaler 反标准化出错: {e}"); return None
            else: predicted_original_scale = predicted_scaled
        except Exception as e: print(f"模型预测出错: {e}"); return None
    if predicted_original_scale is not None:
        if predicted_original_scale.shape[1] != OUTPUT_DIM: print(f"错误：模型输出维度错误"); return None
        return predicted_original_scale.reshape(NODE_COUNT, 3)
    else: return None
def calculate_gii_multi_contact(contacts_info, object_centroid, mu, eigenvalue_threshold):
    if not contacts_info or len(contacts_info) == 0: print("无接触点信息，无法计算 GII。"); return None
    all_wrenches = []
    for contact in contacts_info:
        if isinstance(contact, dict): pt_on_mesh = contact.get('pt_on_mesh'); normal_finger = contact.get('normal')
        elif isinstance(contact, (tuple, list)) and len(contact) >= 6: pt_on_mesh = contact[4]; normal_finger = contact[5]
        else: print("警告：接触点信息格式不兼容，跳过此点。"); continue
        if pt_on_mesh is None or normal_finger is None: print("警告：接触点坐标或法向量缺失，跳过此点。"); continue
        norm_mag = LA.norm(normal_finger)
        if norm_mag < 1e-6: print("警告：接触点法向量接近零，跳过此点。"); continue
        n_contact = - (normal_finger / norm_mag)
        try: t1, t2 = get_orthogonal_vectors(n_contact)
        except ValueError as e: print(f"警告：计算切向量失败: {e}，跳过此接触点。"); continue
        r_contact = pt_on_mesh - object_centroid
        d1 = n_contact + mu * t1; d2 = n_contact - mu * t1; d3 = n_contact + mu * t2; d4 = n_contact - mu * t2
        w1 = np.concatenate((d1, np.cross(r_contact, d1))); w2 = np.concatenate((d2, np.cross(r_contact, d2)))
        w3 = np.concatenate((d3, np.cross(r_contact, d3))); w4 = np.concatenate((d4, np.cross(r_contact, d4)))
        all_wrenches.extend([w1, w2, w3, w4])
    if not all_wrenches: print("未能成功计算任何接触点的 Wrench。"); return None
    grasp_matrix_G = np.column_stack(all_wrenches); # print(f"构建总抓取矩阵 G，形状: {grasp_matrix_G.shape}")
    J = grasp_matrix_G @ grasp_matrix_G.T
    try:
        eigenvalues = LA.eigvalsh(J); non_zero_eigenvalues = eigenvalues[eigenvalues > eigenvalue_threshold]
        if len(non_zero_eigenvalues) > 0:
            lambda_min = np.min(non_zero_eigenvalues); lambda_max = np.max(non_zero_eigenvalues)
            if lambda_max < eigenvalue_threshold: return 0.0
            if lambda_min < -eigenvalue_threshold: print(f"警告: lambda_min < 0 ({lambda_min:.2e})"); return None
            elif lambda_min < 0: lambda_min = 0.0
            return np.sqrt(lambda_min / lambda_max) if lambda_max > 0 else 0.0
        else: return 0.0
    except LA.LinAlgError as e_eig: print(f"计算特征值时出错: {e_eig}"); return None

# --- 全局变量 ---
initial_coords_ref_global = None
model_global, scaler_X_global, scaler_y_global, device_global = None, None, None, None
object_points_global_static = None
object_centroid_global_static = None
num_object_points_global_static = 0
faces_np_global = None
width_translation_vector_global = None
T1_translate_global = None
T2_rotate_global = None


# --- 贝叶斯优化目标函数 ---
param_names = ['r', 'p1', 'p2', 'p3']
dimensions = [
    Real(name='r', low=R_BOUNDS[0], high=R_BOUNDS[1]),
    Real(name='p1', low=P_BOUNDS[0], high=P_BOUNDS[1]),
    Real(name='p2', low=P_BOUNDS[0], high=P_BOUNDS[1]),
    Real(name='p3', low=P_BOUNDS[0], high=P_BOUNDS[1])
]

@use_named_args(dimensions=dimensions)
def evaluate_grasp(r, p1, p2, p3):
    """目标函数：输入参数 r, p1, p2, p3，返回负的 GII 值或惩罚值。"""
    global initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global
    global object_points_global_static, object_centroid_global_static, num_object_points_global_static
    global faces_np_global, width_translation_vector_global, T1_translate_global, T2_rotate_global

    # (移除 evaluate_grasp 内部的全局变量检查)

    print(f"  评估: r={r:.2f}, p1={p1:.0f}, p2={p2:.0f}, p3={p3:.0f}", end=" -> ")
    current_pressures = [p1, p2, p3]
    current_placement_radius = r

    # --- 1. 计算变形 ---
    deformed_finger_meshes_world_this_eval = []
    for i in range(3):
        displacements_matrix = predict_displacements_for_pressure(model_global, scaler_X_global, scaler_y_global, device_global, current_pressures[i])
        if displacements_matrix is None: print(f"预测失败！惩罚=10.0"); return 10.0

        deformed_curve1_ref_unordered = initial_coords_ref_global + displacements_matrix
        curve2_ref = initial_coords_ref_global + width_translation_vector_global
        deformed_curve2_ref_unordered = curve2_ref + displacements_matrix
        sorted_deformed_curve1_ref = sort_points_spatially(deformed_curve1_ref_unordered)
        sorted_deformed_curve2_ref = sort_points_spatially(deformed_curve2_ref_unordered)
        sorted_deformed_vertices_ref = np.vstack((sorted_deformed_curve1_ref, sorted_deformed_curve2_ref))
        deformed_mesh_ref = pv.PolyData(sorted_deformed_vertices_ref, faces=faces_np_global)

        angle_rad = np.radians([0, 120, 240][i])
        rot_angle_z_placing = angle_rad + np.pi / 2.0
        rot_z_placing = create_rotation_matrix_z(rot_angle_z_placing)
        target_pos_on_circle = np.array([ current_placement_radius * np.cos(angle_rad), current_placement_radius * np.sin(angle_rad), 0.0 ])
        T3_place = create_transformation_matrix(rot_z_placing, target_pos_on_circle)
        T_final = T3_place @ T2_rotate_global @ T1_translate_global
        final_transformed_mesh = deformed_mesh_ref.transform(T_final, inplace=False)
        final_transformed_mesh.clean(inplace=True)
        final_transformed_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True, auto_orient_normals=True)
        deformed_finger_meshes_world_this_eval.append(final_transformed_mesh)

    # --- 2. 碰撞检测 ---
    object_point_status = ['Non-Contact'] * num_object_points_global_static
    closest_contact_per_finger_eval = [None] * 3
    min_dist_per_finger_eval = [float('inf')] * 3

    for obj_pt_idx, obj_point in enumerate(object_points_global_static):
        closest_dist_for_this_pt = float('inf'); info_for_this_pt = None
        finger_idx_for_this_pt = -1
        for finger_idx, finger_mesh in enumerate(deformed_finger_meshes_world_this_eval):
            if finger_mesh is None or finger_mesh.n_cells == 0 or 'Normals' not in finger_mesh.cell_data: continue
            try:
                closest_cell_id, pt_on_mesh = finger_mesh.find_closest_cell(obj_point, return_closest_point=True)
                if closest_cell_id < 0 or closest_cell_id >= finger_mesh.n_cells: continue
                dist = LA.norm(obj_point - pt_on_mesh)
                if dist < closest_dist_for_this_pt:
                    closest_dist_for_this_pt = dist; finger_idx_for_this_pt = finger_idx
                    if closest_cell_id < len(finger_mesh.cell_data['Normals']): normal = finger_mesh.cell_normals[closest_cell_id]; info_for_this_pt = (dist, finger_idx, closest_cell_id, pt_on_mesh, normal)
                    else: info_for_this_pt = None
                if dist < collision_threshold and dist < min_dist_per_finger_eval[finger_idx]:
                     min_dist_per_finger_eval[finger_idx] = dist
                     if closest_cell_id < len(finger_mesh.cell_data['Normals']): normal_f = finger_mesh.cell_normals[closest_cell_id]; closest_contact_per_finger_eval[finger_idx] = (dist, obj_pt_idx, finger_idx, closest_cell_id, pt_on_mesh, normal_f)
                     else: closest_contact_per_finger_eval[finger_idx] = None
            except Exception: continue

        if info_for_this_pt:
            dist, _, _, _, _ = info_for_this_pt
            if dist < overlap_threshold: object_point_status[obj_pt_idx] = 'Overlap'
            elif dist < collision_threshold: object_point_status[obj_pt_idx] = 'Contact'

    # --- 3. 计算 GII 或惩罚值 ---
    final_contacts_for_gii_eval = [info for info in closest_contact_per_finger_eval if info is not None]
    has_overlap_eval = any(s == 'Overlap' for s in object_point_status)

    if has_overlap_eval:
        print("重叠！惩罚=5.0")
        return 5.0
    elif len(final_contacts_for_gii_eval) == 3:
        print("3接触点，计算GII...", end="")
        gii = calculate_gii_multi_contact(final_contacts_for_gii_eval, object_centroid_global_static, friction_coefficient, eigenvalue_threshold)
        if gii is not None and gii > 1e-9:
            print(f"GII={gii:.4f}. 返回 {-gii:.4f}")
            return -gii
        else:
            print(f"GII=N/A或0. 惩罚=2.0")
            return 2.0
    else:
        penalty = 3.0 - len(final_contacts_for_gii_eval)
        print(f"{len(final_contacts_for_gii_eval)}接触点. 惩罚={penalty + 1.0:.1f}")
        return penalty + 1.0

# --- 主脚本 ---
if __name__ == '__main__':
    # --- 初始化全局资源 ---
    initial_coords_ref_global = load_initial_coordinates(INITIAL_COORDS_PATH, NODE_COUNT)
    model_global, scaler_X_global, scaler_y_global, device_global = load_prediction_components(MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH, INPUT_DIM, OUTPUT_DIM, HIDDEN_LAYER_1, HIDDEN_LAYER_2, HIDDEN_LAYER_3)
    faces_np_global = create_faces_array(NODE_COUNT)
    width_translation_vector_global = np.array([0, finger_width, 0])
    if initial_coords_ref_global is not None:
        bottom_node_index_ref_global = np.argmin(initial_coords_ref_global[:, 0])
        ref_bottom_midpoint_global = initial_coords_ref_global[bottom_node_index_ref_global] + width_translation_vector_global / 2.0
    else: print("错误：无法加载初始坐标。"); sys.exit()
    T1_translate_global = create_translation_matrix(-ref_bottom_midpoint_global)
    rotation_ref_to_local_global = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    T2_rotate_global = create_transformation_matrix(rotation_ref_to_local_global, None)

    # 生成静态物体点云和质心
    object_points_local = generate_sphere_points(sphere_radius, num_sphere_points)
    object_transform = create_transformation_matrix(create_rotation_matrix_z(sphere_rotation_angle_z), sphere_translation)
    object_points_global_static = transform_points(object_points_local, object_transform)
    if object_points_global_static is None: print("错误：无法生成物体点云。"); sys.exit()
    object_centroid_global_static = np.mean(object_points_global_static, axis=0)
    num_object_points_global_static = object_points_global_static.shape[0]

    # --- 修正：严格检查初始化是否成功 (单独检查) ---
    if initial_coords_ref_global is None: print("错误：initial_coords_ref_global 未初始化。"); sys.exit()
    if model_global is None: print("错误：model_global 未初始化。"); sys.exit()
    if faces_np_global is None or faces_np_global.size == 0: print("错误：faces_np_global 未初始化或为空。"); sys.exit()
    if object_points_global_static is None: print("错误：object_points_global_static 未初始化。"); sys.exit()
    if T1_translate_global is None: print("错误：T1_translate_global 未初始化。"); sys.exit()
    if T2_rotate_global is None: print("错误：T2_rotate_global 未初始化。"); sys.exit()
    # --- 结束修正 ---
    print(f"静态对象点云质心 (全局): {object_centroid_global_static.round(3)}")

    # --- 创建静态 Tray 对象 ---
    # <<< 在优化循环外创建 tray 对象 >>>
    tray = pv.Cylinder(center=tray_center, radius=tray_radius, height=tray_height, direction=(0, 0, 1), resolution=100)
    # <<< 预定义颜色映射表 >>>
    color_map = { 'Non-Contact': list(pv.Color(sphere_color).int_rgb) + [255], 'Contact': list(pv.Color(colliding_point_color).int_rgb) + [255], 'Overlap': list(pv.Color(overlap_point_color).int_rgb) + [255], 'Intersection': list(pv.Color(intersection_point_color).int_rgb) + [255], 'Gray': list(pv.Color('grey').int_rgb) + [255] }

    # --- 执行贝叶斯优化 ---
    print("\n开始贝叶斯优化...")
    result = None
    try:
        result = gp_minimize(
            func=evaluate_grasp,
            dimensions=dimensions,
            acq_func="EI",
            n_calls=N_CALLS_BO,
            n_initial_points=10,
            random_state=123,
            noise=1e-6
        )
    except Exception as e_opt:
         print(f"\n贝叶斯优化过程中发生错误: {e_opt}")
         traceback.print_exc()
         sys.exit()

    print("\n贝叶斯优化结束。")

    # --- 处理和显示结果 ---
    if result is None: print("优化未能成功执行。"); sys.exit()

    best_params_list = result.x
    best_neg_gii = result.fun
    best_params = dict(zip(param_names, best_params_list))
    best_gii = -best_neg_gii if best_neg_gii < -1e-9 else 0.0

    print("\n找到的最优参数:")
    print(f"  r  = {best_params['r']:.4f}")
    print(f"  P1 = {best_params['p1']:.0f}")
    print(f"  P2 = {best_params['p2']:.0f}")
    print(f"  P3 = {best_params['p3']:.0f}")
    print(f"对应的 (负) 目标函数值: {best_neg_gii:.4f}")
    print(f"对应的最大 GII 值: {best_gii:.4f}")

    # --- 使用最优参数进行最终可视化 ---
    print("\n使用最优参数进行最终可视化...")
    final_r = best_params['r']
    final_pressures = [best_params['p1'], best_params['p2'], best_params['p3']]
    final_finger_meshes = []
    final_object_point_status = ['Non-Contact'] * num_object_points_global_static
    final_closest_contact_per_finger = [None] * 3
    final_min_dist_per_finger = [float('inf')] * 3

    # 重新计算最优状态下的网格
    all_final_preds_valid = True
    for i in range(3):
        displacements_matrix = predict_displacements_for_pressure(model_global, scaler_X_global, scaler_y_global, device_global, final_pressures[i])
        if displacements_matrix is None: print(f"最终可视化: 手指 {i+1} 预测失败。"); all_final_preds_valid = False; break
        deformed_curve1_ref_unordered = initial_coords_ref_global + displacements_matrix
        curve2_ref = initial_coords_ref_global + width_translation_vector_global
        deformed_curve2_ref_unordered = curve2_ref + displacements_matrix
        sorted_deformed_curve1_ref = sort_points_spatially(deformed_curve1_ref_unordered)
        sorted_deformed_curve2_ref = sort_points_spatially(deformed_curve2_ref_unordered)
        sorted_deformed_vertices_ref = np.vstack((sorted_deformed_curve1_ref, sorted_deformed_curve2_ref))
        deformed_mesh_ref = pv.PolyData(sorted_deformed_vertices_ref, faces=faces_np_global)

        angle_rad = np.radians([0, 120, 240][i])
        rot_angle_z_placing = angle_rad + np.pi / 2.0
        rot_z_placing = create_rotation_matrix_z(rot_angle_z_placing)
        target_pos_on_circle = np.array([ final_r * np.cos(angle_rad), final_r * np.sin(angle_rad), 0.0 ])
        T3_place = create_transformation_matrix(rot_z_placing, target_pos_on_circle)
        T_final = T3_place @ T2_rotate_global @ T1_translate_global
        final_transformed_mesh = deformed_mesh_ref.transform(T_final, inplace=False)
        final_transformed_mesh.clean(inplace=True)
        final_transformed_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True, auto_orient_normals=True)
        final_finger_meshes.append(final_transformed_mesh)

    if all_final_preds_valid and len(final_finger_meshes) == 3:
        # 最终碰撞检测
        for obj_pt_idx, obj_point in enumerate(object_points_global_static):
             closest_dist_for_this_pt = float('inf'); info_for_this_pt = None; finger_idx_for_this_pt = -1
             for finger_idx, finger_mesh in enumerate(final_finger_meshes):
                 if finger_mesh is None or finger_mesh.n_cells == 0 or 'Normals' not in finger_mesh.cell_data: continue
                 try:
                     closest_cell_id, pt_on_mesh = finger_mesh.find_closest_cell(obj_point, return_closest_point=True)
                     if closest_cell_id < 0 or closest_cell_id >= finger_mesh.n_cells: continue
                     dist = LA.norm(obj_point - pt_on_mesh)
                     if dist < closest_dist_for_this_pt:
                         closest_dist_for_this_pt = dist; finger_idx_for_this_pt = finger_idx
                         if closest_cell_id < len(finger_mesh.cell_data['Normals']): normal = finger_mesh.cell_normals[closest_cell_id]; info_for_this_pt = (dist, finger_idx, closest_cell_id, pt_on_mesh, normal)
                         else: info_for_this_pt = None
                     if dist < collision_threshold and dist < final_min_dist_per_finger[finger_idx]:
                          final_min_dist_per_finger[finger_idx] = dist
                          if closest_cell_id < len(finger_mesh.cell_data['Normals']): normal_f = finger_mesh.cell_normals[closest_cell_id]; final_closest_contact_per_finger[finger_idx] = (dist, obj_pt_idx, finger_idx, closest_cell_id, pt_on_mesh, normal_f)
                          else: final_closest_contact_per_finger[finger_idx] = None
                 except Exception: continue
             if info_for_this_pt:
                 dist, _, _, _, _ = info_for_this_pt
                 if dist < overlap_threshold: final_object_point_status[obj_pt_idx] = 'Overlap'
                 elif dist < collision_threshold: final_object_point_status[obj_pt_idx] = 'Contact'

        # 最终 GII (使用优化结果的 GII)
        final_contacts_for_gii = [info for info in final_closest_contact_per_finger if info is not None]
        final_gii = best_gii

        if any(s=='Overlap' for s in final_object_point_status): final_gii_text_display = "Optimal GII: N/A (Overlap)"
        elif len(final_contacts_for_gii) < 3: final_gii_text_display = "Optimal GII: N/A (<3 Contacts)"
        elif final_gii is not None: final_gii_text_display = f"Optimal GII: {final_gii:.4f}"
        else: final_gii_text_display = "Optimal GII: N/A (Error?)"


        # 可视化
        plotter_final = pv.Plotter(window_size=[1000, 800])
        # --- 修正：使用主作用域中定义的 tray 对象 ---
        plotter_final.add_mesh(tray, color=tray_color, opacity=0.5, name='tray')
        # --- 结束修正 ---
        plotter_final.add_mesh(pv.PointSet(object_centroid_global_static), color='green', point_size=15, render_points_as_spheres=True, name='centroid')
        if show_axes: plotter_final.add_axes_at_origin(labels_off=False)
        for i, mesh in enumerate(final_finger_meshes):
             plotter_final.add_mesh(mesh, color=finger_color, style='surface', edge_color='grey', opacity=0.85, smooth_shading=True, name=f'finger_{i}')

        final_rgba_colors = np.array([color_map.get(status, color_map['Gray']) for status in final_object_point_status], dtype=np.uint8)
        final_point_cloud_polydata = pv.PolyData(object_points_global_static)
        if final_rgba_colors.shape == (num_object_points_global_static, 4): final_point_cloud_polydata.point_data['colors'] = final_rgba_colors
        plotter_final.add_mesh(final_point_cloud_polydata, scalars='colors', rgba=True, style='points', point_size=7, render_points_as_spheres=True, name='sphere_points')

        # 可视化接触点
        contact_marker_colors = ['red', 'green', 'cyan']
        if len(final_contacts_for_gii) > 0:
            print("正在添加最终接触点可视化元素...")
            for k, contact_info in enumerate(final_contacts_for_gii):
                 if contact_info is None: continue
                 pt_on_mesh = contact_info[4]; normal_finger = contact_info[5]
                 # --- 已修正：使用预定义颜色 ---
                 plotter_final.add_mesh(pv.Sphere(center=pt_on_mesh, radius=contact_marker_radius),
                                        color=contact_marker_colors[k % len(contact_marker_colors)],
                                        name=f'contact_marker_{k}')
                 if normal_finger is not None and LA.norm(normal_finger) > 1e-6:
                     norm_viz = normal_finger / LA.norm(normal_finger)
                     plotter_final.add_arrows(cent=pt_on_mesh, direction=norm_viz, mag=contact_normal_length, color='black', name=f'contact_normal_{k}')
                     try:
                         tangent_plane = pv.Plane(center=pt_on_mesh, direction=norm_viz, i_size=contact_plane_size, j_size=contact_plane_size, i_resolution=1, j_resolution=1)
                         plotter_final.add_mesh(tangent_plane, color='gray', opacity=0.4, style='surface', name=f'tangent_plane_{k}')
                     except Exception as e_plane: print(f"最终可视化绘制切面 {k+1} 出错: {e_plane}")

        final_params_text = f"r={best_params['r']:.2f}, P1={best_params['p1']:.0f}, P2={best_params['p2']:.0f}, P3={best_params['p3']:.0f}"
        plotter_final.add_text(f"{final_gii_text_display}\n{final_params_text}", position="upper_edge", font_size=10, name='final_status')
        plotter_final.camera_position = 'iso'
        print("\n显示最优抓取配置。按 Q 键退出。")
        plotter_final.show()

    else:
        print("未能成功为最优参数生成所有手指网格，无法进行最终可视化。")

    print("\n程序结束。")