# -*- coding: utf-8 -*-
import numpy as np
import pyvista as pv
import numpy.linalg as LA
import torch
import torch.nn as nn
import joblib
import sys
from scipy.spatial.distance import cdist
import time
import os
import open3d as o3d
import pandas as pd # 用于数据处理和Excel导出

# --- 打印版本信息 ---
try:
    print(f"Grasp Random Sampling Script for External Point Cloud")
    print(f"Open3D version: {o3d.__version__}")
    print(f"PyVista version: {pv.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Torch version: {torch.__version__}")
    print(f"Pandas version: {pd.__version__}")
except NameError: pass
except Exception as e: print(f"Error printing library versions: {e}")

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
INPUT_DIM = 1
NODE_COUNT = 63; OUTPUT_DIM = NODE_COUNT * 3
HIDDEN_LAYER_1 = 128; HIDDEN_LAYER_2 = 256; HIDDEN_LAYER_3 = 128

# --- 3. 文件路径定义 ---
# --- 请根据您的实际路径修改以下变量 ---
BASE_PATH = r"C:\Users\admin\Desktop\Figure\grasp_experiments\mug"
POINT_CLOUD_PATH = os.path.join(BASE_PATH, "mug.ply")
GRIPPER_POSE_PATH = os.path.join(BASE_PATH, "relative_gripper_to_object_pose.txt")
# -----------------------------------------
MODEL_PATH = 'best_mlp_model.pth'; X_SCALER_PATH = 'x_scaler.joblib'
Y_SCALER_PATH = 'y_scaler.joblib'; INITIAL_COORDS_PATH = 'initial_coordinates.txt'

# --- 4. 实验配置参数 (MODIFIED) ---
OUTPUT_DIR = "gii_random_sampling_results_sugar_box"
EXCEL_FILENAME = "gii_contour_data_sugar_box.xlsx"
tray_radius = 60.0; tray_height = 1.0; finger_width = 10.0

# --- 关键修改：设置固定的r值和随机采样数量 ---
FIXED_R_VALUE = 50 # 固定测试的半径r值
NUM_RANDOM_SAMPLES = 100 # 要生成的随机构型数量
# ----------------------------------------------------

# POINT_CLOUD_SCALE = 900.0
POINT_CLOUD_SCALE = 680.0

# MANUAL_TRANSLATION = [-30.0, -85.0, -10.0] 
# MANUAL_TRANSLATION = [6.0, 0.0, -35.0]
MANUAL_TRANSLATION = [-30.0, 20.0, -25.0]

# MANUAL_ROTATION_EULER = [-90.0, 0.0, -25.0] 
MANUAL_ROTATION_EULER = [0.0, 60.0, -30.0]

INITIAL_PRESSURE_FOR_CONTACT_SEARCH = 100.0
PRESSURE_STEP_FOR_CONTACT_SEARCH = 500.0
MAX_PRESSURE_FOR_CONTACT_SEARCH = 40000.0
collision_threshold = 1.0; overlap_threshold = 1e-3
friction_coefficient = 0.5; eigenvalue_threshold = 1e-6
P_BOUNDS_CLIP = (0.0, 40000.0)
SHOW_ONE_TIME_INITIAL_CONFIG_PREVIEW = True
DOT_PROD_TOLERANCE_LOCAL = 1e-6

# --- 美化参数 ---
finger_color_viz = '#ff7f0e'; tray_color_viz_pv = '#BDB7A4'
object_point_color_viz = '#1f77b4'; background_color_viz = '#EAEAEA'
text_color_viz = 'black'; font_family_viz = 'times'

# --- 5. 辅助函数 ---
def euler_to_rotation_matrix(rx, ry, rz):
    rx_rad, ry_rad, rz_rad = np.radians([rx, ry, rz])
    R_x = np.array([[1,0,0], [0,np.cos(rx_rad),-np.sin(rx_rad)], [0,np.sin(rx_rad),np.cos(rx_rad)]])
    R_y = np.array([[np.cos(ry_rad),0,np.sin(ry_rad)], [0,1,0], [-np.sin(ry_rad),0,np.cos(ry_rad)]])
    R_z = np.array([[np.cos(rz_rad),-np.sin(rz_rad),0], [np.sin(rz_rad),np.cos(rz_rad),0], [0,0,1]])
    return R_z @ R_y @ R_x

def create_rotation_matrix_z(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

def create_transformation_matrix_opt8(r_mat, t_vec):
    matrix=np.identity(4)
    if r_mat is not None: matrix[:3, :3] = r_mat
    if t_vec is not None: matrix[:3, 3] = t_vec.flatten()
    return matrix

def load_initial_coordinates(file_path, expected_nodes):
    try:
        coords = np.loadtxt(file_path, dtype=np.float32, usecols=(0, 1, 2))
        return coords if coords.shape == (expected_nodes, 3) else None
    except Exception as e:
        print(f"加载初始坐标出错: {e}"); return None

def create_faces_array(num_nodes_per_curve):
    faces = []
    for i in range(num_nodes_per_curve - 1):
        p1, p2 = i, i + 1
        p3, p4 = (i + 1) + num_nodes_per_curve, i + num_nodes_per_curve
        faces.append([4, p1, p2, p3, p4])
    return np.hstack(faces) if faces else np.array([], dtype=int)

def sort_points_spatially(points):
    # (此函数保持不变)
    if points is None: return None; points = np.asarray(points)
    if points.shape[0] < 2: return points
    num_points = points.shape[0]; sorted_indices = []; remaining_indices = list(range(num_points))
    current_index = np.argmin(np.sum(points, axis=1)); sorted_indices.append(current_index)
    if current_index in remaining_indices: remaining_indices.pop(remaining_indices.index(current_index))
    while remaining_indices:
        last_point = points[current_index,np.newaxis]; remaining_points_array = points[remaining_indices]
        if remaining_points_array.ndim == 1: remaining_points_array = remaining_points_array[np.newaxis,:]
        if remaining_points_array.shape[0] == 0: break
        distances = cdist(last_point,remaining_points_array)[0]
        if distances.size == 0: break
        nearest_neighbor_relative_index = np.argmin(distances)
        nearest_neighbor_absolute_index = remaining_indices[nearest_neighbor_relative_index]
        sorted_indices.append(nearest_neighbor_absolute_index); current_index = nearest_neighbor_absolute_index
        if nearest_neighbor_absolute_index in remaining_indices: remaining_indices.pop(remaining_indices.index(nearest_neighbor_absolute_index))
    return points[sorted_indices]

def get_orthogonal_vectors(normal_vector):
    # (此函数保持不变)
    n = np.asarray(normal_vector).astype(float) / LA.norm(normal_vector)
    if np.abs(n[0]) > 0.9: v_arbitrary = np.array([0., 1., 0.])
    else: v_arbitrary = np.array([1., 0., 0.])
    t1 = np.cross(n, v_arbitrary); t1 /= LA.norm(t1)
    t2 = np.cross(n, t1); t2 /= LA.norm(t2)
    return t1, t2

def load_prediction_components(model_path,x_scaler_path,y_scaler_path,input_dim,output_dim,h1,h2,h3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"ML使用设备: {device}")
    model = MLPRegression(input_dim,output_dim,h1,h2,h3)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device)); model.to(device); model.eval()
        scaler_X=joblib.load(x_scaler_path)
        scaler_y=joblib.load(y_scaler_path)
        return model,scaler_X,scaler_y,device
    except Exception as e:
        print(f"加载ML组件时出错: {e}"); return None,None,None,None

def predict_displacements_for_pressure(model,scaler_X,scaler_y,device,pressure_value):
    pressure_value=np.clip(pressure_value, P_BOUNDS_CLIP[0], P_BOUNDS_CLIP[1]); input_p=np.array([[pressure_value]],dtype=np.float32)
    input_p_scaled = scaler_X.transform(input_p)
    input_tensor=torch.tensor(input_p_scaled,dtype=torch.float32).to(device)
    with torch.no_grad():
        predicted_scaled_tensor=model(input_tensor)
        predicted_scaled=predicted_scaled_tensor.cpu().numpy()
        predicted_original_scale = scaler_y.inverse_transform(predicted_scaled)
    return predicted_original_scale.reshape(NODE_COUNT, 3)

def calculate_gii_multi_contact(contacts_info, object_centroid_for_gii, mu, eigenvalue_thresh, characteristic_length=1.0):
    # (此函数保持不变)
    if not contacts_info or len(contacts_info) < 2: return None
    all_wrenches = []
    if characteristic_length <= 1e-9: characteristic_length = 1.0
    for contact_details in contacts_info:
        pt_on_mesh = np.asarray(contact_details[4]); n_contact = np.asarray(contact_details[5])
        if LA.norm(n_contact) < 1e-9: continue
        n_contact /= LA.norm(n_contact)
        t1, t2 = get_orthogonal_vectors(n_contact)
        r_contact_vec = pt_on_mesh - object_centroid_for_gii
        d_list = [n_contact + mu * t1, n_contact - mu * t1, n_contact + mu * t2, n_contact - mu * t2]
        for d_force in d_list:
            torque = np.cross(r_contact_vec, d_force)
            wrench = np.concatenate((d_force, torque / characteristic_length))
            all_wrenches.append(wrench)
    if not all_wrenches: return None
    grasp_matrix_G = np.column_stack(all_wrenches)
    J = grasp_matrix_G @ grasp_matrix_G.T
    eigenvalues = LA.eigvalsh(J)
    positive_eigenvalues = eigenvalues[eigenvalues > eigenvalue_thresh]
    if not positive_eigenvalues.size: return 0.0
    lambda_min = np.min(positive_eigenvalues); lambda_max = np.max(positive_eigenvalues)
    return np.sqrt(lambda_min / lambda_max) if lambda_max > 1e-9 else 0.0

# --- 新增：随机角度生成函数 ---
def generate_random_angular_spacings():
    """ 生成一组符合约束的随机相对角间距 (theta_small, theta_mid) """
    # 在圆上随机选择三个点
    points = np.sort(np.random.rand(3) * 360)
    # 计算三个弧长
    delta1 = points[1] - points[0]
    delta2 = points[2] - points[1]
    delta3 = 360.0 - (points[2] - points[0])
    # 排序得到 (small, mid, large)
    thetas = sorted([delta1, delta2, delta3])
    return thetas[0], thetas[1]

# --- 全局变量 ---
initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global = [None]*5
object_points_global_static, object_centroid_global_static, num_object_points_global_static = None, None, 0
faces_np_global, width_translation_vector_global = None, None
T1_translate_global, T2_rotate_global = None, None
T_gripper_pose_global = None

# --- 可视化辅助函数 ---
def setup_pv_plotter(title, window_size=[800,600], off_screen=False):
    plotter = pv.Plotter(window_size=window_size, title=title, off_screen=off_screen)
    plotter.set_background(background_color_viz)
    return plotter

# --- 核心评估函数 (MODIFIED to accept continuous angles) ---
def evaluate_gii_for_angles(r_value, absolute_angles_deg):
    global initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global, \
           object_points_global_static, object_centroid_global_static, \
           faces_np_global, width_translation_vector_global, T1_translate_global, T2_rotate_global, \
           T_gripper_pose_global, friction_coefficient, eigenvalue_threshold, \
           collision_threshold, overlap_threshold, INITIAL_PRESSURE_FOR_CONTACT_SEARCH, \
           PRESSURE_STEP_FOR_CONTACT_SEARCH, MAX_PRESSURE_FOR_CONTACT_SEARCH
    
    default_return = 0.0
    # (压力迭代和接触检测逻辑与之前版本基本一致)
    current_pressures = np.full(3, INITIAL_PRESSURE_FOR_CONTACT_SEARCH, dtype=float)
    finger_contact_established = [False] * 3
    deformed_finger_meshes_at_contact = [None] * 3
    
    while not all(finger_contact_established):
        pressure_increased_this_step = False
        for i in range(3):
            if not finger_contact_established[i]:
                if current_pressures[i] < MAX_PRESSURE_FOR_CONTACT_SEARCH:
                    current_pressures[i] += PRESSURE_STEP_FOR_CONTACT_SEARCH
                    pressure_increased_this_step = True
                else: return default_return
        if not pressure_increased_this_step: break

        current_step_finger_meshes = [None] * 3
        for i in range(3):
            if finger_contact_established[i]:
                current_step_finger_meshes[i] = deformed_finger_meshes_at_contact[i]
                continue
            
            displacements = predict_displacements_for_pressure(model_global, scaler_X_global, scaler_y_global, device_global, current_pressures[i])
            if displacements is None: return default_return
            
            deformed_c1 = initial_coords_ref_global + displacements
            deformed_c2 = initial_coords_ref_global + width_translation_vector_global + displacements
            s_d_c1 = sort_points_spatially(deformed_c1); s_d_c2 = sort_points_spatially(deformed_c2)
            vertices = np.vstack((s_d_c1, s_d_c2))
            mesh_ref = pv.PolyData(vertices, faces=faces_np_global)
            
            # --- 关键修改：直接使用传入的连续角度 ---
            angle_deg_p = absolute_angles_deg[i]
            angle_rad_p = np.radians(angle_deg_p)
            # ----------------------------------------
            
            rot_z_p_mat = create_rotation_matrix_z(angle_rad_p + np.pi / 2.0)
            target_pos_p_vec = np.array([r_value * np.cos(angle_rad_p), r_value * np.sin(angle_rad_p), 0.0])
            T3_p_mat = create_transformation_matrix_opt8(rot_z_p_mat, target_pos_p_vec)
            T_finger_world_mat = T_gripper_pose_global @ T3_p_mat @ T2_rotate_global @ T1_translate_global
            
            mesh_world = mesh_ref.transform(T_finger_world_mat, inplace=False)
            mesh_world.compute_normals(cell_normals=True, point_normals=False, inplace=True, auto_orient_normals=True)
            current_step_finger_meshes[i] = mesh_world

        # (接触检测和GII计算逻辑保持不变)
        has_overall_overlap = False
        contact_made_this_iter = [False] * 3
        dot_products_this_iter = [[] for _ in range(3)]
        for obj_point in object_points_global_static:
            for finger_idx, finger_mesh in enumerate(current_step_finger_meshes):
                closest_cell_id, pt_on_mesh = finger_mesh.find_closest_cell(obj_point, return_closest_point=True)
                dist = LA.norm(obj_point - pt_on_mesh)
                if dist < overlap_threshold: has_overall_overlap = True; break
                if dist < collision_threshold:
                    contact_made_this_iter[finger_idx] = True
                    normal = finger_mesh.cell_normals[closest_cell_id]
                    vec_cto = obj_point - pt_on_mesh
                    dot_prod = np.dot(vec_cto / LA.norm(vec_cto), normal / LA.norm(normal))
                    dot_products_this_iter[finger_idx].append(dot_prod)
            if has_overall_overlap: break
        if has_overall_overlap: return default_return
        for i in range(3):
            if contact_made_this_iter[i] and dot_products_this_iter[i]:
                pos_dp = any(dp > DOT_PROD_TOLERANCE_LOCAL for dp in dot_products_this_iter[i])
                neg_dp = any(dp < -DOT_PROD_TOLERANCE_LOCAL for dp in dot_products_this_iter[i])
                if pos_dp and neg_dp: return default_return
        for i in range(3):
            if not finger_contact_established[i] and contact_made_this_iter[i]:
                finger_contact_established[i] = True
                deformed_finger_meshes_at_contact[i] = current_step_finger_meshes[i].copy()
    
    if not all(finger_contact_established): return default_return

    best_contact_info = [None] * 3
    for finger_idx, finger_mesh in enumerate(deformed_finger_meshes_at_contact):
        min_dist_this_finger = float('inf')
        for obj_point in object_points_global_static:
            closest_cell_id, pt_on_mesh = finger_mesh.find_closest_cell(obj_point, return_closest_point=True)
            dist = LA.norm(obj_point - pt_on_mesh)
            if dist < min_dist_this_finger:
                min_dist_this_finger = dist
                normal = finger_mesh.cell_normals[closest_cell_id]
                best_contact_info[finger_idx] = (dist, 0, finger_idx, closest_cell_id, pt_on_mesh, normal)
    
    valid_contacts = [info for info in best_contact_info if info is not None]
    if len(valid_contacts) < 3: return 0.0
    
    char_length = np.max(np.ptp(object_points_global_static, axis=0))
    gii = calculate_gii_multi_contact(valid_contacts, object_centroid_global_static,
                                      friction_coefficient, eigenvalue_threshold,
                                      characteristic_length=char_length)
    return gii if gii is not None else 0.0

# --- 主脚本 ---
if __name__ == '__main__':
    print("--- 脚本开始 ---")
    # --- 1. 加载通用资源 ---
    initial_coords_ref_global = load_initial_coordinates(INITIAL_COORDS_PATH, NODE_COUNT)
    model_global, scaler_X_global, scaler_y_global, device_global = load_prediction_components(MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH, INPUT_DIM, OUTPUT_DIM, HIDDEN_LAYER_1, HIDDEN_LAYER_2, HIDDEN_LAYER_3)
    if initial_coords_ref_global is None or model_global is None: sys.exit("错误：未能初始化手指模型或坐标。")
    faces_np_global = create_faces_array(NODE_COUNT)
    width_translation_vector_global = np.array([0, finger_width, 0])
    bottom_node_idx = np.argmin(initial_coords_ref_global[:,0]); ref_mid_pt = initial_coords_ref_global[bottom_node_idx] + width_translation_vector_global/2.0
    T1_translate_global = create_transformation_matrix_opt8(None, -ref_mid_pt)
    rot_ref_to_local = np.array([[0,1,0],[0,0,1],[1,0,0]]); T2_rotate_global = create_transformation_matrix_opt8(rot_ref_to_local, None)

    # --- 2. 加载指定的点云和姿态 ---
    try:
        pcd = o3d.io.read_point_cloud(POINT_CLOUD_PATH)
        object_points_global_static = np.asarray(pcd.points) * POINT_CLOUD_SCALE
        object_centroid_global_static = np.mean(object_points_global_static, axis=0)
        num_object_points_global_static = object_points_global_static.shape[0]
        print(f"成功加载并缩放点云: {os.path.basename(POINT_CLOUD_PATH)}, 点数: {num_object_points_global_static}")
    except Exception as e:
        sys.exit(f"错误: 加载点云文件 '{POINT_CLOUD_PATH}' 失败: {e}")

    try:
        T_gripper_pose_from_file = np.loadtxt(GRIPPER_POSE_PATH)
    except Exception as e:
        sys.exit(f"错误: 加载姿态文件 '{GRIPPER_POSE_PATH}' 失败: {e}")
    
    # --- 2.5 应用手动位姿调整 ---
    R_manual = euler_to_rotation_matrix(*MANUAL_ROTATION_EULER)
    t_manual = np.array(MANUAL_TRANSLATION)
    T_manual = create_transformation_matrix_opt8(R_manual, t_manual)
    T_gripper_pose_global = T_gripper_pose_from_file @ T_manual
    print(f"已应用手动平移: {MANUAL_TRANSLATION} 和旋转: {MANUAL_ROTATION_EULER}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    excel_file_path = os.path.join(OUTPUT_DIR, EXCEL_FILENAME)
    
    # --- 3. 初始场景预览 ---
    if SHOW_ONE_TIME_INITIAL_CONFIG_PREVIEW:
        print("\n--- 显示一次性初始构型预览 ---")
        ts_prev, tm_prev = generate_random_angular_spacings()
        abs_angles_prev = (0, ts_prev, ts_prev + tm_prev)
        plotter = setup_pv_plotter(f"Initial Setup: r={FIXED_R_VALUE:.2f}, angles=({abs_angles_prev[1]:.1f}, {abs_angles_prev[2]:.1f})")
        plotter.add_mesh(pv.PolyData(object_points_global_static), color=object_point_color_viz, point_size=5.0, render_points_as_spheres=True)
        
        preview_tray_mesh = pv.Cylinder(center=(0, 0, -tray_height/2.0), direction=(0, 0, 1), radius=tray_radius, height=tray_height, resolution=30)
        preview_tray_mesh.transform(T_gripper_pose_global, inplace=True)
        plotter.add_mesh(preview_tray_mesh, color=tray_color_viz_pv, opacity=0.3)

        for angle_deg in abs_angles_prev:
            undeformed_verts_ref = np.vstack((initial_coords_ref_global, initial_coords_ref_global + width_translation_vector_global))
            undeformed_m_ref = pv.PolyData(undeformed_verts_ref, faces=faces_np_global)
            angle_rad_prev = np.radians(angle_deg)
            rot_z_prev = create_rotation_matrix_z(angle_rad_prev + np.pi / 2.0)
            target_pos_prev = np.array([FIXED_R_VALUE * np.cos(angle_rad_prev), FIXED_R_VALUE * np.sin(angle_rad_prev), 0.0])
            T3_prev = create_transformation_matrix_opt8(rot_z_prev, target_pos_prev)
            T_finger_world_prev = T_gripper_pose_global @ T3_prev @ T2_rotate_global @ T1_translate_global
            undeformed_finger_world_prev = undeformed_m_ref.transform(T_finger_world_prev, inplace=False)
            plotter.add_mesh(undeformed_finger_world_prev, color=finger_color_viz, style='surface', opacity=0.8, show_edges=True, edge_color='gray')
        
        plotter.camera_position = 'iso'; plotter.camera.zoom(1.5)
        plotter.add_text(f"Initial Preview. Close this window to continue...", position="upper_left", font_size=10)
        plotter.show()
        print("--- 初始构型预览结束 ---")

    # --- 4. 主评估循环 (MODIFIED for random sampling) ---
    print(f"\n--- 开始对 r={FIXED_R_VALUE:.2f} 进行 {NUM_RANDOM_SAMPLES} 次随机采样 ---")
    start_time_total = time.time()
    results_for_excel = []
    
    for i in range(NUM_RANDOM_SAMPLES):
        print(f"  采样 {i+1}/{NUM_RANDOM_SAMPLES}...")
        
        theta_s, theta_m = generate_random_angular_spacings()
        # 将相对角度转换为绝对角度 (固定第一个手指在0度)
        absolute_angles = (0.0, theta_s, theta_s + theta_m)
        
        gii_value = evaluate_gii_for_angles(FIXED_R_VALUE, absolute_angles)
        
        if gii_value > 0:
            print(f"    有效抓取, GII = {gii_value:.4f}, thetas=({theta_s:.1f}, {theta_m:.1f})")
            results_for_excel.append({
                'r_value': FIXED_R_VALUE,
                'theta_small': theta_s,
                'theta_mid': theta_m,
                'gii_value': gii_value
            })
        else:
            print(f"    无效抓取或GII为0")

    end_time_total = time.time()
    print(f"\n--- 所有采样评估完成，总耗时: {end_time_total - start_time_total:.2f} 秒 ---")

    # --- 5. 将最终结果写入Excel ---
    if results_for_excel:
        final_df = pd.DataFrame(results_for_excel)
        print("\n最终结果 DataFrame 预览:")
        print(final_df.head())
        try:
            final_df.to_excel(excel_file_path, index=False, float_format="%.4f")
            print(f"\n数据已成功写入Excel: {os.path.abspath(excel_file_path)}")
        except Exception as e:
            print(f"\n保存Excel文件失败: {e}")
    else:
        print("没有有效的抓取结果可供写入。")

    print("\n脚本结束。")
    try: pv.close_all()
    except Exception: pass
