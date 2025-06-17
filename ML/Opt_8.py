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
# --- 导入贝叶斯优化库 ---
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
HIDDEN_LAYER_1 = 128; HIDDEN_LAYER_2 = 256; HIDDEN_LAYER_3 = 128

# --- 3. 文件路径定义 ---
MODEL_PATH = 'best_mlp_model.pth'; X_SCALER_PATH = 'x_scaler.joblib'
Y_SCALER_PATH = 'y_scaler.joblib'; INITIAL_COORDS_PATH = 'initial_coordinates.txt'

# --- 4. 配置参数 ---
tray_radius = 50.0; tray_height = 10.0
tray_center = np.array([0.0, 0.0, -tray_height / 2.0])
finger_width = 10.0

# 物体参数
# OBJECT_TYPE = 'cuboid' # This will be effectively replaced by PLY loading logic
sphere_radius = 15.0; num_sphere_points = 400 # Kept for potential reference if needed elsewhere
sphere_rotation_angle_z = np.radians(30)
sphere_translation = np.array([0, 0, sphere_radius + 35.0])

cuboid_dims = np.array([25.0, 20.0, 30.0]) # Kept for potential reference
num_cuboid_points = 500 # Kept for potential reference
cuboid_rotation_angle_z = np.radians(0) # Used for orienting the loaded PLY
cuboid_center_translation = np.array([0, 0, cuboid_dims[2]/2.0 + 25.0]) # Used for positioning the loaded PLY centroid

# 显示与碰撞参数
show_axes = True; finger_color = 'lightcoral'; tray_color = 'tan'; object_point_color = 'blue'
colliding_point_color = 'magenta'; intersection_point_color = 'yellow'; overlap_point_color = 'orange'
collision_threshold = 1.0; overlap_threshold = 1e-4
friction_coefficient = 0.5; eigenvalue_threshold = 1e-6
max_pressure = 40000.0

# === START: 修改优化和初始搜索参数 ===
NUM_INITIAL_SEARCHES = 1 # <<< 初始搜索尝试次数
PRESSURE_STEP_INIT_SEARCH = 100.0 # 初始搜索压力步长
N_CALLS_BO = 200
N_INITIAL_POINTS_BO = 15
R_BOUNDS = (tray_radius * 0.3, tray_radius * 0.95) # Min R might be adjusted based on loaded object
P_BOUNDS = (0.0, max_pressure)
ANGLE_OFFSET_BOUNDS = (0.0, 120.0) # 角度偏移范围 (度)
# === END: 修改优化和初始搜索参数 ===

# 可视化参数
contact_marker_radius = 1.0; contact_normal_length = 5.0; contact_plane_size = 4.0
show_finger_normals = True
finger_normal_vis_scale = 1.0
finger_normal_color = 'gray'
contact_normal_color = 'black'

# BO 维度定义 (5维)
param_names = ['r', 'p1', 'p2', 'p3', 'angle_offset']
dimensions = [
    Real(name='r', low=R_BOUNDS[0], high=R_BOUNDS[1]),
    Real(name='p1', low=P_BOUNDS[0], high=P_BOUNDS[1]),
    Real(name='p2', low=P_BOUNDS[0], high=P_BOUNDS[1]),
    Real(name='p3', low=P_BOUNDS[0], high=P_BOUNDS[1]),
    Real(name='angle_offset', low=ANGLE_OFFSET_BOUNDS[0], high=ANGLE_OFFSET_BOUNDS[1])
]


# --- 5. 辅助函数 ---
# (辅助函数不变)
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
def transform_points(points, matrix):
    if points.shape[1] != 3: raise ValueError("Points shape");
    if matrix.shape != (4, 4): raise ValueError("Matrix shape");
    num_pts = points.shape[0]; h_points = np.hstack((points, np.ones((num_pts, 1))))
    t_h = h_points @ matrix.T; return t_h[:, :3] / t_h[:, 3, np.newaxis]
def generate_sphere_points(radius, num_points): # Original function, not used for PLY
    points = []; phi = np.pi * (3. - np.sqrt(5.))
    for i in range(num_points):
        z = 1 - (i / float(num_points - 1)) * 2; radius_at_z = np.sqrt(1 - z * z); theta = phi * i
        x = np.cos(theta) * radius_at_z; y = np.sin(theta) * radius_at_z; points.append((x, y, z));
    return np.array(points) * radius
def generate_cuboid_points(dimensions, num_points): # Original function, not used for PLY
    l, w, h = dimensions[0], dimensions[1], dimensions[2]; hl, hw, hh = l / 2.0, w / 2.0, h / 2.0
    area_xy = l * w; area_yz = w * h; area_xz = l * h; total_area = 2 * (area_xy + area_yz + area_xz)
    if total_area <= 1e-6: return np.zeros((0, 3))
    n_xy = int(np.round(num_points * (area_xy / total_area))); n_yz = int(np.round(num_points * (area_yz / total_area))); n_xz = int(np.round(num_points * (area_xz / total_area)))
    points = []
    if n_xy > 0:
        x_top = np.random.uniform(-hl, hl, n_xy); y_top = np.random.uniform(-hw, hw, n_xy); points.append(np.column_stack((x_top, y_top, np.full(n_xy, hh))))
        x_bot = np.random.uniform(-hl, hl, n_xy); y_bot = np.random.uniform(-hw, hw, n_xy); points.append(np.column_stack((x_bot, y_bot, np.full(n_xy, -hh))))
    if n_yz > 0:
        y_right = np.random.uniform(-hw, hw, n_yz); z_right = np.random.uniform(-hh, hh, n_yz); points.append(np.column_stack((np.full(n_yz, hl), y_right, z_right)))
        y_left = np.random.uniform(-hw, hw, n_yz); z_left = np.random.uniform(-hh, hh, n_yz); points.append(np.column_stack((np.full(n_yz, -hl), y_left, z_left)))
    if n_xz > 0:
        x_front = np.random.uniform(-hl, hl, n_xz); z_front = np.random.uniform(-hh, hh, n_xz); points.append(np.column_stack((x_front, np.full(n_xz, hw), z_front)))
        x_back = np.random.uniform(-hl, hl, n_xz); z_back = np.random.uniform(-hh, hh, n_xz); points.append(np.column_stack((x_back, np.full(n_xz, -hw), z_back)))
    if not points: return np.zeros((0, 3))
    return np.concatenate(points, axis=0)
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
    if points is None: return None
    points = np.asarray(points)
    if points.shape[0] < 2: return points
    num_points = points.shape[0]; sorted_indices = []; remaining_indices = list(range(num_points))
    start_node_index = np.argmin(points[:, 0]); current_index = start_node_index
    sorted_indices.append(current_index)
    if current_index in remaining_indices: remaining_indices.pop(remaining_indices.index(current_index))
    else: print(f"Warning: Start index {current_index} not found in remaining indices.");
    while remaining_indices:
        last_point = points[current_index, np.newaxis]
        remaining_points_array = points[remaining_indices]
        if remaining_points_array.ndim == 1: remaining_points_array = remaining_points_array[np.newaxis, :]
        if remaining_points_array.shape[0] == 0: break
        try: distances = cdist(last_point, remaining_points_array)[0]
        except Exception as e_cdist: print(f"Error during cdist: {e_cdist}"); break
        if distances.size == 0: print("Warning: cdist returned empty distances."); break
        nearest_neighbor_relative_index = np.argmin(distances)
        nearest_neighbor_absolute_index = remaining_indices[nearest_neighbor_relative_index]
        sorted_indices.append(nearest_neighbor_absolute_index); current_index = nearest_neighbor_absolute_index
        if nearest_neighbor_absolute_index in remaining_indices: remaining_indices.pop(remaining_indices.index(nearest_neighbor_absolute_index))
        else: print(f"Warning: Nearest index {nearest_neighbor_absolute_index} not found in remaining indices."); break
    if len(sorted_indices) != num_points: print(f"Warning: Spatial sort only processed {len(sorted_indices)} of {num_points} points.")
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
    if not contacts_info or len(contacts_info) == 0: return None
    all_wrenches = []; valid_contacts_for_gii = 0
    for contact in contacts_info:
        if isinstance(contact, (tuple, list)) and len(contact) >= 6: pt_on_mesh = contact[4]; normal_finger = contact[5]
        else: continue
        if pt_on_mesh is None or normal_finger is None: continue
        pt_on_mesh = np.asarray(pt_on_mesh); normal_finger = np.asarray(normal_finger)
        if pt_on_mesh.size != 3 or normal_finger.size != 3: continue
        norm_mag = LA.norm(normal_finger); 
        if norm_mag < 1e-6: continue
        n_contact = - (normal_finger / norm_mag)
        try: t1, t2 = get_orthogonal_vectors(n_contact)
        except ValueError: continue
        r_contact = pt_on_mesh - object_centroid
        d1 = n_contact + mu * t1; d2 = n_contact - mu * t1; d3 = n_contact + mu * t2; d4 = n_contact - mu * t2
        w1 = np.concatenate((d1, np.cross(r_contact, d1))); w2 = np.concatenate((d2, np.cross(r_contact, d2)))
        w3 = np.concatenate((d3, np.cross(r_contact, d3))); w4 = np.concatenate((d4, np.cross(r_contact, d4)))
        all_wrenches.extend([w1, w2, w3, w4]); valid_contacts_for_gii += 1
    if valid_contacts_for_gii == 0: return None
    grasp_matrix_G = np.column_stack(all_wrenches); J = grasp_matrix_G @ grasp_matrix_G.T
    try:
        eigenvalues = LA.eigvalsh(J); non_zero_eigenvalues = eigenvalues[eigenvalues > eigenvalue_threshold]
        if len(non_zero_eigenvalues) > 0:
            lambda_min = np.min(non_zero_eigenvalues);
            if lambda_min < 0 and np.abs(lambda_min) < eigenvalue_threshold: lambda_min = 0.0
            elif lambda_min < 0: return None
            lambda_max = np.max(non_zero_eigenvalues)
            if lambda_max < eigenvalue_threshold: return 0.0
            gii = np.sqrt(lambda_min / lambda_max); return gii
        else: return 0.0
    except LA.LinAlgError: return None


# --- 全局变量 ---
initial_coords_ref_global = None
model_global, scaler_X_global, scaler_y_global, device_global = None, None, None, None
object_points_global_static = None
object_centroid_global_static = None
num_object_points_global_static = 0
object_mesh_global_static = None
object_center_global_static = np.array([0.,0.,0.]) # Will be updated
faces_np_global = None
width_translation_vector_global = None
T1_translate_global = None
T2_rotate_global = None


# --- 贝叶斯优化目标函数 (evaluate_grasp) ---
@use_named_args(dimensions=dimensions)
def evaluate_grasp(r, p1, p2, p3, angle_offset): # <<< 包含 angle_offset
    global initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global
    global object_points_global_static, object_centroid_global_static, num_object_points_global_static
    global faces_np_global, width_translation_vector_global, T1_translate_global, T2_rotate_global

    print(f"  评估: r={r:.4f}, p1={p1:.1f}, p2={p2:.1f}, p3={p3:.1f}, offset={angle_offset:.2f}°", end="") # <<< 包含 angle_offset

    current_pressures = [p1, p2, p3]; current_placement_radius = r
    deformed_finger_meshes_world_this_eval = []; mesh_generation_successful = True
    for i in range(3):
        displacements_matrix = predict_displacements_for_pressure(model_global, scaler_X_global, scaler_y_global, device_global, current_pressures[i])
        if displacements_matrix is None: mesh_generation_successful = False; break
        deformed_curve1_ref_unordered = initial_coords_ref_global + displacements_matrix; curve2_ref = initial_coords_ref_global + width_translation_vector_global; deformed_curve2_ref_unordered = curve2_ref + displacements_matrix
        sorted_deformed_curve1_ref = sort_points_spatially(deformed_curve1_ref_unordered); sorted_deformed_curve2_ref = sort_points_spatially(deformed_curve2_ref_unordered)
        if sorted_deformed_curve1_ref is None or sorted_deformed_curve2_ref is None: mesh_generation_successful = False; break
        sorted_deformed_vertices_ref = np.vstack((sorted_deformed_curve1_ref, sorted_deformed_curve2_ref))
        if faces_np_global is None or faces_np_global.size == 0: mesh_generation_successful = False; break
        try: deformed_mesh_ref = pv.PolyData(sorted_deformed_vertices_ref, faces=faces_np_global)
        except Exception: mesh_generation_successful = False; break
        base_angle_deg = [0, 120, 240][i]; current_angle_deg = base_angle_deg + angle_offset; angle_rad = np.radians(current_angle_deg)
        rot_angle_z_placing = angle_rad + np.pi / 2.0; rot_z_placing = create_rotation_matrix_z(rot_angle_z_placing)
        target_pos_on_circle = np.array([ current_placement_radius * np.cos(angle_rad), current_placement_radius * np.sin(angle_rad), 0.0 ]); T3_place = create_transformation_matrix(rot_z_placing, target_pos_on_circle)
        T_final = T3_place @ T2_rotate_global @ T1_translate_global;
        try:
             final_transformed_mesh = deformed_mesh_ref.transform(T_final, inplace=False);
             if final_transformed_mesh is None or final_transformed_mesh.n_points == 0: mesh_generation_successful = False; break
             final_transformed_mesh.clean(inplace=True);
             if final_transformed_mesh.n_cells > 0: final_transformed_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True, auto_orient_normals=True, non_manifold_traversal=False)
        except Exception: mesh_generation_successful = False; break
        deformed_finger_meshes_world_this_eval.append(final_transformed_mesh)
    if not mesh_generation_successful: print(" -> 网格生成失败, Cost=10.0"); return 10.0
    object_point_status = ['Non-Contact'] * num_object_points_global_static; closest_contact_per_finger_eval = [None] * 3
    min_dist_per_finger_eval = [float('inf')] * 3; finger_dot_products = [[] for _ in range(3)]
    has_overlap_eval = False; num_contact_points = 0
    for obj_pt_idx, obj_point in enumerate(object_points_global_static):
        closest_dist_for_this_pt = float('inf'); finger_idx_for_this_pt = -1
        normal_for_this_pt = None; pt_on_mesh_for_this_pt = None
        for finger_idx, finger_mesh in enumerate(deformed_finger_meshes_world_this_eval):
            if finger_mesh is None or finger_mesh.n_cells == 0 : continue
            has_normals = 'Normals' in finger_mesh.cell_data and finger_mesh.cell_data['Normals'] is not None
            try:
                closest_cell_id, pt_on_mesh_candidate = finger_mesh.find_closest_cell(obj_point, return_closest_point=True)
                if closest_cell_id < 0 or closest_cell_id >= finger_mesh.n_cells: continue
                dist = LA.norm(obj_point - pt_on_mesh_candidate)
                if dist < closest_dist_for_this_pt:
                    closest_dist_for_this_pt = dist; finger_idx_for_this_pt = finger_idx
                    pt_on_mesh_for_this_pt = pt_on_mesh_candidate; current_normal = None
                    if has_normals and closest_cell_id < len(finger_mesh.cell_data['Normals']): current_normal = finger_mesh.cell_normals[closest_cell_id]
                    normal_for_this_pt = current_normal
                if dist < collision_threshold and dist < min_dist_per_finger_eval[finger_idx]:
                     min_dist_per_finger_eval[finger_idx] = dist; current_normal_for_gii = None
                     if has_normals and closest_cell_id < len(finger_mesh.cell_data['Normals']): current_normal_for_gii = finger_mesh.cell_normals[closest_cell_id]
                     if current_normal_for_gii is not None: closest_contact_per_finger_eval[finger_idx] = (dist, obj_pt_idx, finger_idx, closest_cell_id, pt_on_mesh_candidate, current_normal_for_gii)
            except Exception: continue
        if finger_idx_for_this_pt != -1:
            dist = closest_dist_for_this_pt; normal_closest = normal_for_this_pt; pt_on_mesh_closest = pt_on_mesh_for_this_pt
            if dist < overlap_threshold: object_point_status[obj_pt_idx] = 'Overlap'; has_overlap_eval = True
            elif dist < collision_threshold:
                 object_point_status[obj_pt_idx] = 'Contact'; num_contact_points += 1
                 if normal_closest is not None and LA.norm(normal_closest) > 1e-9:
                      vector_to_point = obj_point - pt_on_mesh_closest
                      if LA.norm(vector_to_point) > 1e-9:
                          dot_prod = np.dot(vector_to_point, normal_closest)
                          finger_dot_products[finger_idx_for_this_pt].append(dot_prod)
    final_contacts_for_gii_eval = [info for info in closest_contact_per_finger_eval if info is not None]
    grasp_state_eval = "No Contact"; cost = 4.0
    if has_overlap_eval: grasp_state_eval = "Overlap"; cost = 5.0
    else:
        finger_intersects = [False] * 3; dot_prod_tolerance = 1e-6
        for i in range(3):
            if finger_dot_products[i]:
                has_positive_dp = any(dp > dot_prod_tolerance for dp in finger_dot_products[i])
                has_negative_dp = any(dp < -dot_prod_tolerance for dp in finger_dot_products[i])
                if has_positive_dp and has_negative_dp: finger_intersects[i] = True
        if any(finger_intersects): grasp_state_eval = "Intersection"; cost = 4.0
        elif num_contact_points > 0:
            grasp_state_eval = "Contact"; num_gii_contacts = len(final_contacts_for_gii_eval)
            if num_gii_contacts >= 3:
                gii = calculate_gii_multi_contact(final_contacts_for_gii_eval, object_centroid_global_static, friction_coefficient, eigenvalue_threshold)
                if gii is not None and gii > 1e-9: cost = -gii
                else: cost = 2.0
            else: cost = 1.0 + (3.0 - num_gii_contacts)
    if grasp_state_eval == "Contact" and cost < 0: print(f" -> State={grasp_state_eval}, GII={-cost:.4f}, Cost={cost:.4f}")
    else: print(f" -> State={grasp_state_eval}, Cost={cost:.4f}")
    return cost


# === START: find_initial_grasp 函数 ===
def find_initial_grasp(initial_r, initial_angle_offset, pressure_step, max_pressure_init): # <<< 添加 initial_angle_offset
    print(f"\n--- 开始寻找初始接触点 (r = {initial_r:.2f}, offset = {initial_angle_offset:.1f}°) ---") # <<< 修改打印信息
    global initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global
    global object_points_global_static, num_object_points_global_static
    global faces_np_global, width_translation_vector_global, T1_translate_global, T2_rotate_global

    current_pressures_init = np.array([0.0, 0.0, 0.0]); finger_contact_achieved_init = [False, False, False]; init_iter_count = 0
    last_meshes = [None] * 3; last_point_status = ['Non-Contact'] * num_object_points_global_static; last_closest_contacts = [None] * 3
    dot_prod_tolerance = 1e-6

    while True:
        init_iter_count += 1; pressure_changed_init = False
        for i in range(3):
            if not finger_contact_achieved_init[i] and current_pressures_init[i] < max_pressure_init: current_pressures_init[i] += pressure_step; current_pressures_init[i] = min(current_pressures_init[i], max_pressure_init); pressure_changed_init = True
        print(f"  初始点搜索 Iter {init_iter_count}: P=[{current_pressures_init[0]:.0f}, {current_pressures_init[1]:.0f}, {current_pressures_init[2]:.0f}]")

        deformed_finger_meshes_init = []; valid_preds = True
        for i in range(3):
            displacements_matrix = predict_displacements_for_pressure(model_global, scaler_X_global, scaler_y_global, device_global, current_pressures_init[i])
            if displacements_matrix is None: valid_preds = False; break
            deformed_curve1_ref_unordered = initial_coords_ref_global + displacements_matrix; curve2_ref = initial_coords_ref_global + width_translation_vector_global; deformed_curve2_ref_unordered = curve2_ref + displacements_matrix
            sorted_deformed_curve1_ref = sort_points_spatially(deformed_curve1_ref_unordered); sorted_deformed_curve2_ref = sort_points_spatially(deformed_curve2_ref_unordered)
            if sorted_deformed_curve1_ref is None or sorted_deformed_curve2_ref is None: valid_preds = False; break
            sorted_deformed_vertices_ref = np.vstack((sorted_deformed_curve1_ref, sorted_deformed_curve2_ref));
            if faces_np_global is None or faces_np_global.size == 0: valid_preds = False; break
            try: deformed_mesh_ref = pv.PolyData(sorted_deformed_vertices_ref, faces=faces_np_global)
            except Exception: valid_preds = False; break
            base_angle_deg = [0, 120, 240][i]
            current_angle_deg = base_angle_deg + initial_angle_offset
            angle_rad = np.radians(current_angle_deg)
            rot_angle_z_placing = angle_rad + np.pi / 2.0
            rot_z_placing = create_rotation_matrix_z(rot_angle_z_placing)
            target_pos_on_circle = np.array([ initial_r * np.cos(angle_rad), initial_r * np.sin(angle_rad), 0.0 ])
            T3_place = create_transformation_matrix(rot_z_placing, target_pos_on_circle)
            T_final = T3_place @ T2_rotate_global @ T1_translate_global;
            try:
                 final_transformed_mesh = deformed_mesh_ref.transform(T_final, inplace=False);
                 if final_transformed_mesh is None or final_transformed_mesh.n_points == 0: valid_preds = False; break
                 final_transformed_mesh.clean(inplace=True);
                 if final_transformed_mesh.n_cells > 0: final_transformed_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True, auto_orient_normals=True, non_manifold_traversal=False)
                 deformed_finger_meshes_init.append(final_transformed_mesh)
            except Exception: valid_preds = False; break
        if not valid_preds: print("  初始点搜索: 预测或网格处理失败。"); return { 'status': 'PredictionFailed', 'r': initial_r, 'angle_offset': initial_angle_offset, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }
        last_meshes = deformed_finger_meshes_init
        has_overlap_init = False; num_contact_points_init = 0
        current_closest_contacts = [None] * 3; current_min_dists = [float('inf')] * 3; current_point_status = ['Non-Contact'] * num_object_points_global_static
        finger_dot_products_init = [[] for _ in range(3)]
        contacting_fingers_indices = set()
        for obj_pt_idx, obj_point in enumerate(object_points_global_static):
             closest_dist_for_this_pt = float('inf'); finger_idx_for_this_pt = -1
             normal_for_this_pt = None; pt_on_mesh_for_this_pt = None
             for finger_idx, finger_mesh in enumerate(deformed_finger_meshes_init):
                 if finger_mesh is None or finger_mesh.n_cells == 0: continue
                 has_normals = 'Normals' in finger_mesh.cell_data and finger_mesh.cell_data['Normals'] is not None
                 try:
                     closest_cell_id, pt_on_mesh_candidate = finger_mesh.find_closest_cell(obj_point, return_closest_point=True)
                     if closest_cell_id < 0 or closest_cell_id >= finger_mesh.n_cells: continue
                     dist = LA.norm(obj_point - pt_on_mesh_candidate)
                     if dist < closest_dist_for_this_pt:
                         closest_dist_for_this_pt = dist; finger_idx_for_this_pt = finger_idx
                         pt_on_mesh_for_this_pt = pt_on_mesh_candidate; current_normal = None
                         if has_normals and closest_cell_id < len(finger_mesh.cell_data['Normals']): current_normal = finger_mesh.cell_normals[closest_cell_id]
                         normal_for_this_pt = current_normal
                     if dist < collision_threshold and dist < current_min_dists[finger_idx]:
                         current_min_dists[finger_idx] = dist; current_normal_contact = None
                         if has_normals and closest_cell_id < len(finger_mesh.cell_data['Normals']): current_normal_contact = finger_mesh.cell_normals[closest_cell_id]
                         current_closest_contacts[finger_idx] = (dist, obj_pt_idx, finger_idx, closest_cell_id, pt_on_mesh_candidate, current_normal_contact)
                 except Exception: continue
             if finger_idx_for_this_pt != -1:
                 dist = closest_dist_for_this_pt; normal_closest = normal_for_this_pt; pt_on_mesh_closest = pt_on_mesh_for_this_pt
                 if dist < overlap_threshold: current_point_status[obj_pt_idx] = 'Overlap'; has_overlap_init = True; break
                 elif dist < collision_threshold:
                      current_point_status[obj_pt_idx] = 'Contact'; num_contact_points_init += 1
                      contacting_fingers_indices.add(finger_idx_for_this_pt)
                      if normal_closest is not None and LA.norm(normal_closest) > 1e-9:
                           vector_to_point = obj_point - pt_on_mesh_closest
                           if LA.norm(vector_to_point) > 1e-9:
                                dot_prod = np.dot(vector_to_point, normal_closest)
                                finger_dot_products_init[finger_idx_for_this_pt].append(dot_prod)
        last_point_status = current_point_status; last_closest_contacts = current_closest_contacts
        return_dict_base = { 'r': initial_r, 'angle_offset': initial_angle_offset, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }
        if has_overlap_init:
            print("  初始点搜索: 检测到重叠。"); return {**return_dict_base, 'status': 'Overlap'}
        finger_intersects_init = [False] * 3
        if num_contact_points_init > 0:
             for i in range(3):
                  if finger_dot_products_init[i]:
                       has_positive_dp = any(dp > dot_prod_tolerance for dp in finger_dot_products_init[i])
                       has_negative_dp = any(dp < -dot_prod_tolerance for dp in finger_dot_products_init[i])
                       if has_positive_dp and has_negative_dp: finger_intersects_init[i] = True
        if any(finger_intersects_init):
            print("  初始点搜索: 检测到交叉穿透。"); return {**return_dict_base, 'status': 'Intersection'}
        for i in range(3):
            if i in contacting_fingers_indices: finger_contact_achieved_init[i] = True
        print(f"  初始点搜索: 接触状态={list(finger_contact_achieved_init)}")
        if all(finger_contact_achieved_init):
            print(f"--- 成功找到初始接触点组合 (r={initial_r:.2f}, offset={initial_angle_offset:.1f}°, P={current_pressures_init.round(0)}) ---")
            return {**return_dict_base, 'status': 'FoundInitialPoint'}
        if not pressure_changed_init or all(p >= max_pressure_init for i, p in enumerate(current_pressures_init) if not finger_contact_achieved_init[i]):
            print("--- 初始点搜索: 达到最大压力或压力不再改变，但未实现三点接触 ---")
            return {**return_dict_base, 'status': 'MaxPressureReached'}
# === END: find_initial_grasp 函数 ===


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

    # === START: 修改后的物体加载部分 ===
    print("加载外部点云模型: Bird_Model.ply")
    # ply_path = 'C:\\Users\\admin\\Desktop\\Bird_Model.ply' # 使用双反斜杠
    ply_path = 'C:\\Users\\admin\\Desktop\\Bird_Model.ply'
    try:
        loaded_mesh_pv = pv.read(ply_path) # Use a different variable name to avoid conflict
    except FileNotFoundError:
        print(f"错误: 找不到PLY文件 {ply_path}")
        sys.exit()
    except Exception as e:
        print(f"加载PLY文件 {ply_path} 时出错: {e}")
        sys.exit()

    if loaded_mesh_pv.n_points == 0:
        print(f"错误: 从 {ply_path} 加载的点云不包含任何点。")
        sys.exit()

        # === START: 点云抽稀 (ADDED CODE BLOCK) ===
    target_point_count = 2000
    if loaded_mesh_pv.n_points > target_point_count:
        print(f"原始点云数量: {loaded_mesh_pv.n_points}，进行抽稀至约 {target_point_count} 点...")
        
        processed_mesh_for_points = None # Will hold the mesh after decimation attempt

        # Attempt mesh decimation if it's a surface mesh
        # Check for actual surface cell types
        # 确保 loaded_mesh_pv 是一个 PolyData 对象并且包含面数据
        is_surface_mesh = False # 默认值
        if loaded_mesh_pv and hasattr(loaded_mesh_pv, 'n_cells') and loaded_mesh_pv.n_cells > 0:
            try:
                # 根据错误提示 "Did you mean: 'GetCellTypes'?" 尝试调用 GetCellTypes() 方法
                cell_types_iterable = loaded_mesh_pv.GetCellTypes()

                if cell_types_iterable is not None:
                    # 确保 cell_types_iterable 可以安全地转换为集合
                    # 有些 VTK Get 方法可能需要进一步处理，但 PyVista 通常会返回 NumPy 数组
                    try:
                        unique_cell_types = set(cell_types_iterable)
                    except TypeError:
                        print("Warning: loaded_mesh_pv.GetCellTypes() did not return an iterable.")
                        unique_cell_types = set() # 或者其他错误处理

                    surface_cell_types = {pv.CellType.TRIANGLE, pv.CellType.QUAD, pv.CellType.POLYGON}
                    
                    is_surface_mesh = any(ct in surface_cell_types for ct in unique_cell_types)
                else:
                    print("Warning: loaded_mesh_pv.GetCellTypes() returned None.")
                    
            except AttributeError:
                print("Error: 'PolyData' object still has no attribute 'GetCellTypes' or 'celltypes'.")
                # 在这里可以尝试打印 PyVista 版本或对象的 dir() 来进一步调试
                # print(f"PyVista version: {pv.__version__}")
                # print(dir(loaded_mesh_pv))
            except Exception as e:
                print(f"An unexpected error occurred while getting cell types: {e}")

        else:
            if not loaded_mesh_pv:
                print("Warning: loaded_mesh_pv is None.")
            elif not hasattr(loaded_mesh_pv, 'n_cells'):
                print("Warning: loaded_mesh_pv does not have 'n_cells' attribute.")
            elif loaded_mesh_pv.n_cells == 0:
                print("Info: Mesh has no cells (n_cells is 0).")
            # is_surface_mesh 保持 False

        if is_surface_mesh:
            print("  检测到表面网格，尝试使用 decimate_pro。")
            try:
                surface_mesh = loaded_mesh_pv.extract_surface().triangulate()
                if surface_mesh.n_points > target_point_count:
                    reduction_factor_surf = 1.0 - (target_point_count / surface_mesh.n_points)
                    if reduction_factor_surf < 0: reduction_factor_surf = 0.0
                    if reduction_factor_surf >= 1.0: reduction_factor_surf = 0.99 # Max sensible reduction
                    
                    print(f"  三角化后点数: {surface_mesh.n_points}，目标缩减率: {reduction_factor_surf:.4f}")
                    decimated_mesh = surface_mesh.decimate_pro(reduction_factor_surf, preserve_topology=True)
                    
                    if decimated_mesh.n_points > 0:
                        print(f"  decimate_pro 后点云数量: {decimated_mesh.n_points}")
                        processed_mesh_for_points = decimated_mesh
                    else:
                        print("  decimate_pro 结果点数为0，将使用原始三角化网格进行随机采样。")
                        processed_mesh_for_points = surface_mesh # Fallback to triangulated if decimation fails badly
                elif surface_mesh.n_points > 0: # Triangulation okay, no decimation needed or possible
                    print("  三角化后点数已在目标范围内或无法进一步抽稀。")
                    processed_mesh_for_points = surface_mesh
                else: # Triangulation failed
                    print("  表面提取或三角化失败。")
            except Exception as e:
                print(f"  网格抽稀 (decimate_pro) 过程出错: {e}。")
        
        # If decimation was not performed, not successful, or needs further reduction
        if processed_mesh_for_points is None or processed_mesh_for_points.n_points == 0:
            if processed_mesh_for_points is None: # Decimation was not even attempted for surface
                print("  非表面网格或 decimate_pro 不适用。")
            # Use the original loaded_mesh_pv for random sampling
            processed_mesh_for_points = loaded_mesh_pv 
        
        # Final step: Random sampling if point count is still too high
        if processed_mesh_for_points.n_points > target_point_count:
            print(f"  当前点数 ({processed_mesh_for_points.n_points}) 仍高于目标，进行最终随机采样。")
            indices = np.random.choice(processed_mesh_for_points.n_points, target_point_count, replace=False)
            final_points = processed_mesh_for_points.points[indices]
            loaded_mesh_pv = pv.PolyData(final_points) # This results in a point cloud
            print(f"  最终随机采样后点云数量: {loaded_mesh_pv.n_points}")
        elif processed_mesh_for_points.n_points > 0 : # Decimation was successful and count is good
            loaded_mesh_pv = processed_mesh_for_points # Use the mesh from decimation/triangulation
            print(f"  使用抽稀/处理后的网格，点数: {loaded_mesh_pv.n_points}")
        else: # Should not happen if original had points, but as a safeguard
             print("  抽稀过程后点数为0，检查原始点云。")
             # loaded_mesh_pv remains as original if all attempts yield 0 points and original had points.
             # This case implies something went very wrong if original n_points > 0.

    elif loaded_mesh_pv.n_points == 0: # This case should have been caught before
        print(f"错误: 从 {ply_path} 加载的点云不包含任何点（抽稀前检查）。")
        sys.exit()
    else: # n_points <= target_point_count
        print(f"原始点云数量: {loaded_mesh_pv.n_points} (在 {target_point_count} 以下或等于，无需抽稀)。")
    # === END: 点云抽稀 ===
    # 计算加载点云的原始质心
    raw_points = loaded_mesh_pv.points.copy()
    raw_centroid = np.mean(raw_points, axis=0)

    # 将加载的点云的几何中心移至原点
    points_centered_at_origin = raw_points - raw_centroid
    
    # 创建一个新的PyVista对象，其点已居中，并保留原始拓扑（面）
    centered_mesh_pv = pv.PolyData(points_centered_at_origin)
    if hasattr(loaded_mesh_pv, 'faces') and loaded_mesh_pv.faces is not None and loaded_mesh_pv.faces.size > 0:
        centered_mesh_pv.faces = loaded_mesh_pv.faces.copy()

    # 使用原始长方体的旋转和平移参数，以确保与托盘的相对位置一致
    # 这些参数定义了原始长方体的目标姿态和其中心的目标全局位置
    target_object_rotation_angle_z = cuboid_rotation_angle_z
    target_object_centroid_global_position = cuboid_center_translation

    # 创建旋转矩阵和平移向量（对于变换矩阵）
    object_rotation_matrix = create_rotation_matrix_z(target_object_rotation_angle_z)
    # 变换矩阵将首先应用旋转（对于已居中的点云），然后平移到目标质心位置
    object_final_transform_matrix = create_transformation_matrix(object_rotation_matrix, target_object_centroid_global_position)

    # 对已居中的网格应用最终变换
    object_mesh_global_static = centered_mesh_pv.transform(object_final_transform_matrix, inplace=False)
    object_points_global_static = object_mesh_global_static.points.copy()

    if object_points_global_static is None or object_points_global_static.shape[0] == 0:
        print("错误：变换加载的点云失败或导致点云为空。")
        sys.exit()

    # 更新全局质心和“中心”点（此处的中心指原始物体中心的目标位置，现在也是新物体质心的目标位置）
    object_centroid_global_static = np.mean(object_points_global_static, axis=0)
    object_center_global_static = target_object_centroid_global_position.copy() # This was the target for the original object's center

    num_object_points_global_static = object_points_global_static.shape[0]
    print(f"已加载并处理外部模型 {ply_path}，包含 {num_object_points_global_static} 个点。")
    print(f"计算得到的全局质心: {object_centroid_global_static.round(3)}")
    print(f"目标物体中心/质心 (原长方体中心位置): {object_center_global_static.round(3)}")
    # === END: 修改后的物体加载部分 ===

    # 检查全局变量 (代码不变)
    if initial_coords_ref_global is None: print("错误：initial_coords_ref_global 未初始化。"); sys.exit()
    if model_global is None: print("错误：model_global 未初始化。"); sys.exit()
    if faces_np_global is None or faces_np_global.size == 0: print("错误：faces_np_global 未初始化或为空。"); sys.exit()
    if object_points_global_static is None: print("错误：object_points_global_static 未初始化。"); sys.exit()
    if T1_translate_global is None: print("错误：T1_translate_global 未初始化。"); sys.exit()
    if T2_rotate_global is None: print("错误：T2_rotate_global 未初始化。"); sys.exit()

    tray = pv.Cylinder(center=tray_center, radius=tray_radius, height=tray_height, direction=(0, 0, 1), resolution=100)
    color_map = { 'Non-Contact': list(pv.Color(object_point_color).int_rgb) + [255], 'Contact': list(pv.Color(colliding_point_color).int_rgb) + [255], 'Overlap': list(pv.Color(overlap_point_color).int_rgb) + [255], 'Intersection': list(pv.Color(intersection_point_color).int_rgb) + [255], 'Gray': list(pv.Color('grey').int_rgb) + [255] }


    # === START: 步骤 1: 恢复并修改初始搜索 ===
    print(f"\n--- 开始在 {NUM_INITIAL_SEARCHES} 个随机 (r, angle_offset) 组合上寻找初始接触点 ---")
    collected_initial_points = [] 
    last_search_attempt_result = None

    # === START: 修改r_search_min 和 r_search_max 的定义 ===
    print("\n--- 根据加载的物体计算初始搜索半径范围 ---")
    # 计算加载物体在XY平面上相对于其质心的等效半径
    x_coords_rel_centroid = object_points_global_static[:, 0] - object_centroid_global_static[0]
    y_coords_rel_centroid = object_points_global_static[:, 1] - object_centroid_global_static[1]
    distances_from_centroid_xy = np.sqrt(x_coords_rel_centroid**2 + y_coords_rel_centroid**2)
    if distances_from_centroid_xy.size > 0:
        loaded_object_effective_radius = np.max(distances_from_centroid_xy)
    else: # Fallback if no points or all points are at centroid
        print("警告: 无法从点云计算有效半径，使用默认值。")
        loaded_object_effective_radius = tray_radius * 0.1 # Small default

    r_search_min = loaded_object_effective_radius * 1.1 # 乘以一个系数作为最小搜索半径
    r_search_max = tray_radius * 0.95 # 最大搜索半径保持不变

    # 确保 r_search_min 小于 r_search_max 并且在定义的 R_BOUNDS 内 (或更新R_BOUNDS)
    r_search_min = max(r_search_min, R_BOUNDS[0]) # 不能小于总边界的最小值
    r_search_min = min(r_search_min, R_BOUNDS[1] * 0.9) # 不能太大，确保有搜索空间
    r_search_max = max(r_search_max, r_search_min + (R_BOUNDS[1] - R_BOUNDS[0])*0.1) # 确保 max > min
    r_search_max = min(r_search_max, R_BOUNDS[1]) # 不能大于总边界的最大值

    if r_search_min >= r_search_max:
        print(f"错误: 随机半径搜索范围无效 (min={r_search_min:.2f} >= max={r_search_max:.2f})。")
        print(f"将使用 R_BOUNDS: ({R_BOUNDS[0]:.2f}, {R_BOUNDS[1]:.2f})")
        r_search_min = R_BOUNDS[0]
        r_search_max = R_BOUNDS[1]
    # === END: 修改r_search_min 和 r_search_max 的定义 ===

    angle_offset_search_min = ANGLE_OFFSET_BOUNDS[0]
    angle_offset_search_max = ANGLE_OFFSET_BOUNDS[1]

    print(f"更新后的搜索范围: r in [{r_search_min:.2f}, {r_search_max:.2f}], angle_offset in [{angle_offset_search_min:.1f}, {angle_offset_search_max:.1f}]")

    for k in range(NUM_INITIAL_SEARCHES):
        print(f"\n--- 初始搜索尝试 {k+1}/{NUM_INITIAL_SEARCHES} ---")
        r_init = np.random.uniform(r_search_min, r_search_max)
        angle_offset_init = np.random.uniform(angle_offset_search_min, angle_offset_search_max)

        search_result = find_initial_grasp(
            initial_r=r_init,
            initial_angle_offset=angle_offset_init,
            pressure_step=PRESSURE_STEP_INIT_SEARCH,
            max_pressure_init=max_pressure
        )
        last_search_attempt_result = search_result

        if search_result and search_result['status'] == 'FoundInitialPoint':
            r_found = search_result['r']
            p_found = search_result['pressures']
            angle_offset_found = search_result['angle_offset']
            print(f"*** 在 r={r_found:.2f}, offset={angle_offset_found:.1f}° 处找到有效初始点: P={np.round(p_found, 0)} ***")
            params_5d = [r_found] + p_found + [angle_offset_found]
            in_bounds = (
                 R_BOUNDS[0] <= r_found <= R_BOUNDS[1] and
                all(P_BOUNDS[0] <= p <= P_BOUNDS[1] for p in p_found) and
                ANGLE_OFFSET_BOUNDS[0] <= angle_offset_found <= ANGLE_OFFSET_BOUNDS[1]
            )
            if in_bounds:
                 collected_initial_points.append(params_5d)
            else: print(f"警告: 找到的初始点参数 {params_5d} 不在贝叶斯优化边界内，将不用于 x0。")
        else:
            print(f"--- 在 r={r_init:.2f}, offset={angle_offset_init:.1f}° 处未找到有效初始接触点 (状态: {search_result.get('status', 'Unknown')}) ---")

    print(f"\n--- 初始接触点搜索完成，共找到 {len(collected_initial_points)} 个有效初始点 ---")
    # === END: 步骤 1: 恢复并修改初始搜索 ===


    # === START: 步骤 1.5: 恢复初始可视化 ===
    if last_search_attempt_result and 'finger_meshes' in last_search_attempt_result:
        print("\n可视化最后一次初始搜索尝试的状态...")
        plotter_init_viz = pv.Plotter(window_size=[800, 600], title=f"Initial Search Attempt {NUM_INITIAL_SEARCHES}/{NUM_INITIAL_SEARCHES}")
        plotter_init_viz.add_mesh(tray, color=tray_color, opacity=0.5, name='tray')
        plotter_init_viz.add_mesh(pv.PointSet(object_centroid_global_static), color='green', point_size=12, render_points_as_spheres=True, name='centroid')
        if object_mesh_global_static: plotter_init_viz.add_mesh(object_mesh_global_static, style='wireframe', color='cyan', line_width=2, name='object_outline_init')
        if show_axes: plotter_init_viz.add_axes_at_origin()
        init_meshes = last_search_attempt_result['finger_meshes']
        for i, mesh in enumerate(init_meshes):
             if mesh is not None: plotter_init_viz.add_mesh(mesh, color=finger_color, style='surface', edge_color='grey', opacity=0.85, smooth_shading=True, name=f'finger_{i}')
        init_point_status = last_search_attempt_result.get('object_point_status', ['Gray'] * num_object_points_global_static)
        init_rgba_colors = np.array([color_map.get(status, color_map['Gray']) for status in init_point_status], dtype=np.uint8)
        init_point_cloud_polydata = pv.PolyData(object_points_global_static)
        if init_rgba_colors.shape == (num_object_points_global_static, 4):
             init_point_cloud_polydata.point_data['colors'] = init_rgba_colors
             plotter_init_viz.add_mesh(init_point_cloud_polydata, scalars='colors', rgba=True, style='points', point_size=5, render_points_as_spheres=True, name='object_points')
        else: plotter_init_viz.add_mesh(init_point_cloud_polydata, color=object_point_color, style='points', point_size=5, render_points_as_spheres=True, name='object_points')
        init_status_text = (f"Last Search Attempt ({NUM_INITIAL_SEARCHES}/{NUM_INITIAL_SEARCHES}): {last_search_attempt_result.get('status', 'N/A')}\n"
                            f"r={last_search_attempt_result.get('r', 'N/A'):.2f}, "
                            f"offset={last_search_attempt_result.get('angle_offset', 'N/A'):.1f}°, "
                            f"P={np.array(last_search_attempt_result.get('pressures', [0,0,0])).round(0)}")
        plotter_init_viz.add_text(init_status_text, position="upper_edge", font_size=10)
        plotter_init_viz.camera_position = 'iso'
        print("显示最后一次初始搜索结果... 按 Q 关闭此窗口以继续。")
        plotter_init_viz.show()
    else: print("错误：初始点搜索未能生成任何可供可视化的结果。"); sys.exit()
    # === END: 步骤 1.5: 恢复初始可视化 ===


    # --- 步骤 2: 准备并执行贝叶斯优化 (使用收集到的 5D 初始点) ---
    proceed_to_bo = False
    if collected_initial_points:
        print(f"\n使用找到的 {len(collected_initial_points)} 个有效初始接触点进行贝叶斯优化。")
        proceed_to_bo = True
    else:
        print(f"\n警告：未能通过初始搜索找到任何有效的初始接触点。")
        user_choice = 'yes' 
        if user_choice == 'yes':
             print("将仅使用贝叶斯优化自身的随机初始点进行优化。")
             proceed_to_bo = True; N_INITIAL_POINTS_BO = max(N_INITIAL_POINTS_BO, 1)
        else: print("跳过贝叶斯优化。")

    result = None
    if proceed_to_bo:
        num_provided_points = len(collected_initial_points)
        num_random_points_needed = max(0, N_INITIAL_POINTS_BO - num_provided_points)
        if num_provided_points == 0 and num_random_points_needed == 0: print("错误：没有提供初始点，且 BO 的随机初始点数设置为 0。"); sys.exit()

        print(f"\n开始贝叶斯优化... (提供 {num_provided_points} 个初始点, BO将额外生成 {num_random_points_needed} 个随机点)")
        try:
            result = gp_minimize(
                func=evaluate_grasp,
                dimensions=dimensions, 
                acq_func="EI",
                n_calls=N_CALLS_BO,
                n_initial_points=num_random_points_needed, 
                x0=collected_initial_points if collected_initial_points else None, 
                random_state=123,
                noise=1e-6 
             )
        except Exception as e_opt: print(f"\n贝叶斯优化过程中发生错误: {e_opt}"); traceback.print_exc(); sys.exit()
        print("\n贝叶斯优化结束。")

    # --- 步骤 3: 处理和显示结果 ---
    if result is not None:
        best_params_list = result.x; best_cost = result.fun
        best_params = dict(zip(param_names, best_params_list))

        print("\n使用最优参数重新评估最终状态...")
        final_r = best_params['r']; final_p1 = best_params['p1']; final_p2 = best_params['p2']; final_p3 = best_params['p3']
        final_angle_offset = best_params['angle_offset']
        final_cost_check = evaluate_grasp([final_r, final_p1, final_p2, final_p3, final_angle_offset])

        best_gii_from_eval = -final_cost_check if final_cost_check < 0 else 0.0
        print("\n找到的最优参数:")
        print(f"  r            = {best_params['r']:.4f}")
        print(f"  P1           = {best_params['p1']:.0f}")
        print(f"  P2           = {best_params['p2']:.0f}")
        print(f"  P3           = {best_params['p3']:.0f}")
        print(f"  angle_offset = {best_params['angle_offset']:.2f}°")
        print(f"优化找到的目标函数值 (成本): {best_cost:.4f}")
        print(f"重新评估确认的成本: {final_cost_check:.4f}")
        if best_gii_from_eval > 0: print(f"重新评估确认的最大 GII 值: {best_gii_from_eval:.4f}")
        else: print(f"重新评估确认的状态不产生有效 GII。")

        # --- 步骤 4: 使用最优参数进行最终可视化 ---
        print("\n使用最优参数生成最终可视化...")
        final_pressures = [best_params['p1'], best_params['p2'], best_params['p3']]
        final_finger_meshes = []; all_final_preds_valid = True; dot_prod_tolerance = 1e-6
        for i in range(3):
            displacements_matrix = predict_displacements_for_pressure(model_global, scaler_X_global, scaler_y_global, device_global, final_pressures[i])
            if displacements_matrix is None: all_final_preds_valid = False; break
            deformed_curve1_ref_unordered = initial_coords_ref_global + displacements_matrix; curve2_ref = initial_coords_ref_global + width_translation_vector_global; deformed_curve2_ref_unordered = curve2_ref + displacements_matrix
            sorted_deformed_curve1_ref = sort_points_spatially(deformed_curve1_ref_unordered); sorted_deformed_curve2_ref = sort_points_spatially(deformed_curve2_ref_unordered)
            if sorted_deformed_curve1_ref is None or sorted_deformed_curve2_ref is None: all_final_preds_valid = False; break
            sorted_deformed_vertices_ref = np.vstack((sorted_deformed_curve1_ref, sorted_deformed_curve2_ref));
            if faces_np_global is None or faces_np_global.size == 0: all_final_preds_valid = False; break
            try: deformed_mesh_ref = pv.PolyData(sorted_deformed_vertices_ref, faces=faces_np_global)
            except Exception: all_final_preds_valid = False; break
            base_angle_deg = [0, 120, 240][i]; current_angle_deg = base_angle_deg + final_angle_offset; angle_rad = np.radians(current_angle_deg)
            rot_angle_z_placing = angle_rad + np.pi / 2.0; rot_z_placing = create_rotation_matrix_z(rot_angle_z_placing)
            target_pos_on_circle = np.array([ final_r * np.cos(angle_rad), final_r * np.sin(angle_rad), 0.0 ]); T3_place = create_transformation_matrix(rot_z_placing, target_pos_on_circle)
            T_final = T3_place @ T2_rotate_global @ T1_translate_global;
            try:
                 final_transformed_mesh = deformed_mesh_ref.transform(T_final, inplace=False);
                 if final_transformed_mesh is None or final_transformed_mesh.n_points == 0: all_final_preds_valid = False; break
                 final_transformed_mesh.clean(inplace=True);
                 if final_transformed_mesh.n_cells > 0: final_transformed_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True, auto_orient_normals=True, non_manifold_traversal=False)
                 final_finger_meshes.append(final_transformed_mesh)
            except Exception: all_final_preds_valid = False; break

        if all_final_preds_valid and len(final_finger_meshes) == 3:
            final_object_point_status = ['Non-Contact'] * num_object_points_global_static; final_closest_contact_per_finger = [None] * 3
            final_finger_dot_products = [[] for _ in range(3)]; final_has_overlap = False; final_num_contact_points = 0
            for obj_pt_idx, obj_point in enumerate(object_points_global_static):
                closest_dist_for_this_pt = float('inf'); finger_idx_for_this_pt = -1
                normal_for_this_pt = None; pt_on_mesh_for_this_pt = None
                for finger_idx, finger_mesh in enumerate(final_finger_meshes):
                    if finger_mesh is None or finger_mesh.n_cells == 0 : continue
                    has_normals = 'Normals' in finger_mesh.cell_data and finger_mesh.cell_data['Normals'] is not None
                    try:
                        closest_cell_id, pt_on_mesh_candidate = finger_mesh.find_closest_cell(obj_point, return_closest_point=True)
                        if closest_cell_id < 0 or closest_cell_id >= finger_mesh.n_cells: continue
                        dist = LA.norm(obj_point - pt_on_mesh_candidate)
                        if dist < closest_dist_for_this_pt:
                            closest_dist_for_this_pt = dist; finger_idx_for_this_pt = finger_idx
                            pt_on_mesh_for_this_pt = pt_on_mesh_candidate; current_normal = None
                            if has_normals and closest_cell_id < len(finger_mesh.cell_data['Normals']): current_normal = finger_mesh.cell_normals[closest_cell_id]
                            normal_for_this_pt = current_normal
                        if dist < collision_threshold :
                             current_normal_for_gii = None
                             if has_normals and closest_cell_id < len(finger_mesh.cell_data['Normals']): current_normal_for_gii = finger_mesh.cell_normals[closest_cell_id]
                             if current_normal_for_gii is not None:
                                if final_closest_contact_per_finger[finger_idx] is None or dist < final_closest_contact_per_finger[finger_idx][0]:
                                       final_closest_contact_per_finger[finger_idx] = (dist, obj_pt_idx, finger_idx, closest_cell_id, pt_on_mesh_candidate, current_normal_for_gii)
                    except Exception: continue
                if finger_idx_for_this_pt != -1:
                    dist = closest_dist_for_this_pt; normal_closest = normal_for_this_pt; pt_on_mesh_closest = pt_on_mesh_for_this_pt
                    if dist < overlap_threshold: final_object_point_status[obj_pt_idx] = 'Overlap'; final_has_overlap = True
                    elif dist < collision_threshold:
                         final_object_point_status[obj_pt_idx] = 'Contact'; final_num_contact_points += 1
                         if normal_closest is not None and LA.norm(normal_closest) > 1e-9:
                              vector_to_point = obj_point - pt_on_mesh_closest
                              if LA.norm(vector_to_point) > 1e-9:
                                   dot_prod = np.dot(vector_to_point, normal_closest)
                                   final_finger_dot_products[finger_idx_for_this_pt].append(dot_prod)
            final_contacts_for_gii = [info for info in final_closest_contact_per_finger if info is not None]
            final_state_str = "No Contact"; final_finger_intersects = [False] * 3
            if final_has_overlap: final_state_str = "Overlap"
            else:
                for i in range(3):
                    if final_finger_dot_products[i]:
                        has_positive_dp = any(dp > dot_prod_tolerance for dp in final_finger_dot_products[i])
                        has_negative_dp = any(dp < -dot_prod_tolerance for dp in final_finger_dot_products[i])
                        if has_positive_dp and has_negative_dp: final_finger_intersects[i] = True
                if any(final_finger_intersects): final_state_str = "Intersection"
                elif final_num_contact_points > 0: final_state_str = "Contact"
            final_gii = None
            if final_state_str == "Contact":
                 if len(final_contacts_for_gii) >= 3: final_gii = calculate_gii_multi_contact(final_contacts_for_gii, object_centroid_global_static, friction_coefficient, eigenvalue_threshold)
            if final_state_str == "Overlap": final_gii_text_display = "Optimal GII: N/A (Overlap)"
            elif final_state_str == "Intersection": final_gii_text_display = "Optimal GII: N/A (Intersection)"
            elif final_state_str != "Contact" or len(final_contacts_for_gii) < 3: final_gii_text_display = f"Optimal GII: N/A ({final_state_str}, {len(final_contacts_for_gii)} GII Contacts)"
            elif final_gii is not None and final_gii > 1e-9: final_gii_text_display = f"Optimal GII: {final_gii:.4f}"
            else: final_gii_text_display = f"Optimal GII: N/A ({len(final_contacts_for_gii)} GII Contacts, GII Failed or Low)"

            plotter_final = pv.Plotter(window_size=[1000, 800], title="Optimal Grasp Configuration")
            plotter_final.add_mesh(tray, color=tray_color, opacity=0.5, name='tray')
            plotter_final.add_mesh(pv.PointSet(object_centroid_global_static), color='green', point_size=15, render_points_as_spheres=True, name='centroid')
            if object_mesh_global_static: plotter_final.add_mesh(object_mesh_global_static, style='wireframe', color='cyan', line_width=2, name='object_outline_final')
            if show_axes: plotter_final.add_axes_at_origin(labels_off=False)
            for i, mesh in enumerate(final_finger_meshes):
                if mesh is not None:
                    plotter_final.add_mesh(mesh, color=finger_color, style='surface', edge_color='grey', opacity=0.85, smooth_shading=True, name=f'finger_{i}')
                    if show_finger_normals and mesh.n_cells > 0 and 'Normals' in mesh.cell_data and mesh.cell_data['Normals'] is not None:
                        try:
                            cell_centers = mesh.cell_centers(); cell_normals = np.asarray(mesh.cell_data['Normals'])
                            if cell_centers.n_points == len(cell_normals): plotter_final.add_arrows(cent=cell_centers.points, direction=cell_normals, mag=finger_normal_vis_scale, color=finger_normal_color, name=f'finger_{i}_normals')
                        except Exception as e_norm_viz_final: print(f"Error visualizing normals for finger {i+1}: {e_norm_viz_final}")
            final_rgba_colors = np.array([color_map.get(status, color_map['Gray']) for status in final_object_point_status], dtype=np.uint8)
            final_point_cloud_polydata = pv.PolyData(object_points_global_static)
            if final_rgba_colors.shape == (num_object_points_global_static, 4):
                 final_point_cloud_polydata.point_data['colors'] = final_rgba_colors
                 plotter_final.add_mesh(final_point_cloud_polydata, scalars='colors', rgba=True, style='points', point_size=5, render_points_as_spheres=True, name='object_points')
            else: plotter_final.add_mesh(final_point_cloud_polydata, color=object_point_color, style='points', point_size=5, render_points_as_spheres=True, name='object_points')
            contact_marker_colors = ['red', 'green', 'cyan', 'purple', 'orange', 'pink']
            if final_state_str == "Contact" and len(final_contacts_for_gii) > 0:
                print("正在添加最终接触点可视化元素...")
                for k, contact_info in enumerate(final_contacts_for_gii):
                     if contact_info is None or len(contact_info) < 6: continue
                     pt_on_mesh = contact_info[4]; normal_finger = contact_info[5]
                     if pt_on_mesh is None or normal_finger is None: continue
                     plotter_final.add_mesh(pv.Sphere(center=pt_on_mesh, radius=contact_marker_radius), color=contact_marker_colors[k % len(contact_marker_colors)], name=f'contact_marker_{k}')
                     if LA.norm(normal_finger) > 1e-6:
                         norm_viz = normal_finger / LA.norm(normal_finger)
                         plotter_final.add_arrows(cent=pt_on_mesh, direction=norm_viz, mag=contact_normal_length, color=contact_normal_color, name=f'contact_normal_{k}')
                         try:
                             tangent_plane = pv.Plane(center=pt_on_mesh, direction=norm_viz, i_size=contact_plane_size, j_size=contact_plane_size, i_resolution=1, j_resolution=1)
                             plotter_final.add_mesh(tangent_plane, color='gray', opacity=0.4, style='surface', name=f'tangent_plane_{k}')
                         except Exception as e_plane: print(f"最终可视化绘制切面 {k+1} 出错: {e_plane}")

            final_params_text = f"r={best_params['r']:.2f}, P1={best_params['p1']:.0f}, P2={best_params['p2']:.0f}, P3={best_params['p3']:.0f}, offset={best_params['angle_offset']:.1f}°"
            plotter_final.add_text(f"Final State: {final_state_str}\n{final_gii_text_display}\n{final_params_text}", position="upper_edge", font_size=10, name='final_status')

            plotter_final.camera_position = 'iso'
            print("\n显示最优抓取配置。按 Q 键退出。")
            plotter_final.show()
        else: print("未能为最优参数生成所有有效的手指网格，无法进行最终可视化。")
    else: print("\n贝叶斯优化未成功运行，无法显示最优结果。")

    print("\n程序结束。")