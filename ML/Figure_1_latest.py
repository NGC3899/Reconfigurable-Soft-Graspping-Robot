# 用于生成3*3热图的程序

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
from itertools import combinations # 用于生成所有手指位置组合
import pandas as pd # 用于数据处理和Excel导出

# --- 打印版本信息 ---
try:
    print(f"Grasp Enumeration Script for External Point Cloud")
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

# --- 3. 文件路径定义 (MODIFIED) ---
# --- 请根据您的实际路径修改以下变量 ---
BASE_PATH = r"C:\Users\admin\Desktop\Figure\grasp_experiments\sugar_box"
POINT_CLOUD_PATH = os.path.join(BASE_PATH, "Sugar_Box.ply")
GRIPPER_POSE_PATH = os.path.join(BASE_PATH, "relative_gripper_to_object_pose.txt")
# -----------------------------------------
MODEL_PATH = 'best_mlp_model.pth'; X_SCALER_PATH = 'x_scaler.joblib'
Y_SCALER_PATH = 'y_scaler.joblib'; INITIAL_COORDS_PATH = 'initial_coordinates.txt'

# --- 4. 实验配置参数 (MODIFIED) ---
OUTPUT_DIR = "gii_enumeration_results_sugar_box"
EXCEL_FILENAME = "gii_heatmap_data_sugar_box.xlsx"
tray_radius = 60.0; tray_height = 1.0; finger_width = 10.0
# --- 关键修改：将槽位数从6改为8，以实现45度间隔 ---
NUM_FINGER_SLOTS = 8 # 总共8个可选槽位 (360 / 45 = 8)
# ----------------------------------------------------
# 设置6个不同的r值进行测试
R_VALUES_TO_TEST = np.linspace(40.0, 50.0, 3) 
# 新增：点云缩放比例
POINT_CLOUD_SCALE = 900.0
# POINT_CLOUD_SCALE = 680.0

# --- 新增：手动调整夹爪位姿 ---
# 在加载的姿态矩阵基础上，进一步进行平移 (x, y, z)
MANUAL_TRANSLATION = [-30.0, -85.0, -10.0] 
# MANUAL_TRANSLATION = [-12.0, 0.0, -35.0]
# 在加载的姿态矩阵基础上，进一步绕 x, y, z 轴旋转 (单位: 度)
MANUAL_ROTATION_EULER = [-90.0, 0.0, -25.0] 
# MANUAL_ROTATION_EULER = [0.0, 0.0, 0.0] 
# ----------------------------------

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

# --- 5. 辅助函数 (大部分保持不变) ---
def euler_to_rotation_matrix(rx, ry, rz):
    """ 将欧拉角(度)转换为旋转矩阵 (ZYX顺序) """
    rx_rad, ry_rad, rz_rad = np.radians([rx, ry, rz])
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(rx_rad), -np.sin(rx_rad)],
                    [0, np.sin(rx_rad), np.cos(rx_rad)]])
    R_y = np.array([[np.cos(ry_rad), 0, np.sin(ry_rad)],
                    [0, 1, 0],
                    [-np.sin(ry_rad), 0, np.cos(ry_rad)]])
    R_z = np.array([[np.cos(rz_rad), -np.sin(rz_rad), 0],
                    [np.sin(rz_rad), np.cos(rz_rad), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x

def create_rotation_matrix(axis, angle_rad):
    axis = np.asarray(axis).astype(float); axis_norm = LA.norm(axis)
    if axis_norm < 1e-9 : return np.identity(3)
    axis /= axis_norm; a = np.cos(angle_rad / 2.0); b, c, d = -axis * np.sin(angle_rad / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d; bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)], [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)], [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
def create_rotation_matrix_z(a): return create_rotation_matrix([0,0,1], a)
def create_transformation_matrix_opt8(r_mat, t_vec):
    matrix=np.identity(4)
    if r_mat is not None: matrix[:3, :3] = r_mat
    if t_vec is not None: matrix[:3, 3] = t_vec.flatten()
    return matrix
def load_initial_coordinates(file_path, expected_nodes):
    try: coords = np.loadtxt(file_path, dtype=np.float32, usecols=(0, 1, 2))
    except Exception as e: print(f"加载初始坐标出错: {e}"); return None
    if coords.shape == (expected_nodes, 3): return coords
    else: print(f"错误: 坐标形状与预期不符."); return None
def create_faces_array(num_nodes_per_curve):
    faces = []; num_quads = num_nodes_per_curve - 1
    if num_quads <= 0: return np.array([], dtype=int)
    for i in range(num_quads): p1,p2=i,i+1; p3,p4=(i+1)+num_nodes_per_curve,i+num_nodes_per_curve; faces.append([4,p1,p2,p3,p4])
    return np.hstack(faces)
def sort_points_spatially(points):
    if points is None: return None; points = np.asarray(points)
    if points.shape[0] < 2: return points
    num_points = points.shape[0]; sorted_indices = []; remaining_indices = list(range(num_points))
    current_index = np.argmin(np.sum(points, axis=1)); sorted_indices.append(current_index)
    if current_index in remaining_indices: remaining_indices.pop(remaining_indices.index(current_index))
    while remaining_indices:
        last_point = points[current_index,np.newaxis]; remaining_points_array = points[remaining_indices]
        if remaining_points_array.ndim == 1: remaining_points_array = remaining_points_array[np.newaxis,:]
        if remaining_points_array.shape[0] == 0: break
        try: distances = cdist(last_point,remaining_points_array)[0]
        except Exception as e_cdist: print(f"Error cdist: {e_cdist}"); break
        if distances.size == 0: break
        nearest_neighbor_relative_index = np.argmin(distances)
        nearest_neighbor_absolute_index = remaining_indices[nearest_neighbor_relative_index]
        sorted_indices.append(nearest_neighbor_absolute_index); current_index = nearest_neighbor_absolute_index
        if nearest_neighbor_absolute_index in remaining_indices: remaining_indices.pop(remaining_indices.index(nearest_neighbor_absolute_index))
    if len(sorted_indices) != num_points: print(f"Warn sort: {len(sorted_indices)}/{num_points}.")
    return points[sorted_indices]
def get_orthogonal_vectors(normal_vector):
    n = np.asarray(normal_vector).astype(float); norm_n = LA.norm(n)
    if norm_n < 1e-9: raise ValueError("Normal vector is zero or near zero.")
    n /= norm_n
    if np.abs(n[0]) > 0.9: v_arbitrary = np.array([0., 1., 0.])
    else: v_arbitrary = np.array([1., 0., 0.])
    t1 = np.cross(n, v_arbitrary); norm_t1 = LA.norm(t1)
    if norm_t1 < 1e-9:
        v_arbitrary = np.array([0., 0., 1.]); t1 = np.cross(n, v_arbitrary); norm_t1 = LA.norm(t1)
        if norm_t1 < 1e-9: raise ValueError("Fallback t1 computation failed.")
    t1 /= norm_t1; t2_temp = np.cross(n, t1); norm_t2 = LA.norm(t2_temp)
    if norm_t2 < 1e-9: raise ValueError("Cannot compute tangent 2.")
    t2 = t2_temp / norm_t2; return t1, t2
def load_prediction_components(model_path,x_scaler_path,y_scaler_path,input_dim,output_dim,h1,h2,h3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"ML使用设备: {device}")
    model = MLPRegression(input_dim,output_dim,h1,h2,h3)
    try: model.load_state_dict(torch.load(model_path, map_location=device)); model.to(device); model.eval()
    except Exception as e: print(f"加载模型 {model_path} 出错: {e}"); return None,None,None,None
    try: scaler_X=joblib.load(x_scaler_path)
    except Exception as e: print(f"加载 X Scaler '{x_scaler_path}' 出错: {e}"); return None,None,None,None
    try: scaler_y=joblib.load(y_scaler_path)
    except Exception as e: print(f"加载 Y Scaler '{y_scaler_path}' 出错: {e}"); return None,None,None,None
    return model,scaler_X,scaler_y,device
def predict_displacements_for_pressure(model,scaler_X,scaler_y,device,pressure_value):
    if model is None: return None
    pressure_value=np.clip(pressure_value, P_BOUNDS_CLIP[0], P_BOUNDS_CLIP[1]); input_p=np.array([[pressure_value]],dtype=np.float32)
    input_p_scaled = scaler_X.transform(input_p)
    input_tensor=torch.tensor(input_p_scaled,dtype=torch.float32).to(device)
    with torch.no_grad():
        predicted_scaled_tensor=model(input_tensor)
        predicted_scaled=predicted_scaled_tensor.cpu().numpy()
        predicted_original_scale = scaler_y.inverse_transform(predicted_scaled)
    return predicted_original_scale.reshape(NODE_COUNT, 3)
def calculate_gii_multi_contact(contacts_info, object_centroid_for_gii, mu, eigenvalue_thresh, characteristic_length=1.0):
    if not contacts_info or len(contacts_info) < 2: return None
    all_wrenches = []
    if characteristic_length <= 1e-9: characteristic_length = 1.0
    for contact_details in contacts_info:
        if not isinstance(contact_details, (tuple, list)) or len(contact_details) < 6: continue
        pt_on_mesh = np.asarray(contact_details[4])
        n_contact = np.asarray(contact_details[5])
        if LA.norm(n_contact) < 1e-9: continue
        n_contact /= LA.norm(n_contact)
        try: t1, t2 = get_orthogonal_vectors(n_contact)
        except ValueError: continue
        r_contact_vec = pt_on_mesh - object_centroid_for_gii
        d_list = [n_contact + mu * t1, n_contact - mu * t1, n_contact + mu * t2, n_contact - mu * t2]
        for d_force in d_list:
            torque = np.cross(r_contact_vec, d_force)
            normalized_torque = torque / characteristic_length
            wrench = np.concatenate((d_force, normalized_torque))
            all_wrenches.append(wrench)
    if not all_wrenches: return None
    grasp_matrix_G = np.column_stack(all_wrenches)
    J = grasp_matrix_G @ grasp_matrix_G.T
    try:
        eigenvalues = LA.eigvalsh(J)
        positive_eigenvalues = eigenvalues[eigenvalues > eigenvalue_thresh]
        if not positive_eigenvalues.size: return 0.0
        lambda_min = np.min(positive_eigenvalues); lambda_max = np.max(positive_eigenvalues)
        if lambda_max < 1e-9: return 0.0
        return np.sqrt(lambda_min / lambda_max)
    except LA.LinAlgError: return None

# --- 全局变量 ---
initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global = [None]*5
object_points_global_static, object_centroid_global_static, num_object_points_global_static = None, None, 0
faces_np_global, width_translation_vector_global = None, None
T1_translate_global, T2_rotate_global = None, None
T_gripper_pose_global = None # 重命名以更清晰地表示夹爪姿态

# --- 可视化辅助函数 ---
def setup_pv_plotter(title, window_size=[800,600], off_screen=False):
    plotter = pv.Plotter(window_size=window_size, title=title, off_screen=off_screen)
    plotter.set_background(background_color_viz)
    return plotter

# --- 核心评估函数 (逻辑基本不变) ---
def evaluate_gii_for_config(r_value, finger_indices_tuple):
    global initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global, \
           object_points_global_static, object_centroid_global_static, num_object_points_global_static, \
           faces_np_global, width_translation_vector_global, T1_translate_global, T2_rotate_global, \
           T_gripper_pose_global, NUM_FINGER_SLOTS, friction_coefficient, eigenvalue_threshold, \
           collision_threshold, overlap_threshold, INITIAL_PRESSURE_FOR_CONTACT_SEARCH, \
           PRESSURE_STEP_FOR_CONTACT_SEARCH, MAX_PRESSURE_FOR_CONTACT_SEARCH
    
    default_return = 0.0
    if not all(v is not None for v in [initial_coords_ref_global, model_global, object_points_global_static, faces_np_global, width_translation_vector_global, T1_translate_global, T2_rotate_global, T_gripper_pose_global]): return default_return
    if num_object_points_global_static == 0 : return default_return
    if len(set(finger_indices_tuple)) < 3: return default_return
    
    # 压力迭代逻辑... (与原脚本相同)
    current_pressures = np.full(3, INITIAL_PRESSURE_FOR_CONTACT_SEARCH, dtype=float)
    final_contact_pressures = np.full(3, -1.0, dtype=float)
    finger_contact_established = [False, False, False]
    deformed_finger_meshes_at_contact = [None] * 3
    
    while not all(finger_contact_established):
        pressure_increased_this_step = False
        for i in range(3):
            if not finger_contact_established[i]:
                if current_pressures[i] < MAX_PRESSURE_FOR_CONTACT_SEARCH:
                    current_pressures[i] += PRESSURE_STEP_FOR_CONTACT_SEARCH
                    pressure_increased_this_step = True
                else:
                    return default_return # 达到压力上限仍未接触
        if not pressure_increased_this_step: break # 如果没有手指压力增加，则跳出

        current_step_finger_meshes = [None] * 3
        # ... 生成网格 ... (与原脚本相同)
        for i in range(3):
            if finger_contact_established[i] and deformed_finger_meshes_at_contact[i] is not None:
                current_step_finger_meshes[i] = deformed_finger_meshes_at_contact[i]; continue
            displacements = predict_displacements_for_pressure(model_global, scaler_X_global, scaler_y_global, device_global, current_pressures[i])
            if displacements is None: return default_return
            deformed_c1 = initial_coords_ref_global + displacements
            deformed_c2 = initial_coords_ref_global + width_translation_vector_global + displacements
            s_d_c1 = sort_points_spatially(deformed_c1); s_d_c2 = sort_points_spatially(deformed_c2)
            vertices = np.vstack((s_d_c1, s_d_c2))
            mesh_ref = pv.PolyData(vertices, faces=faces_np_global)
            
            pos_idx = finger_indices_tuple[i]
            angle_rad_p = np.radians(pos_idx * (360.0 / NUM_FINGER_SLOTS))
            rot_z_p_mat = create_rotation_matrix_z(angle_rad_p + np.pi / 2.0)
            target_pos_p_vec = np.array([r_value * np.cos(angle_rad_p), r_value * np.sin(angle_rad_p), 0.0])
            T3_p_mat = create_transformation_matrix_opt8(rot_z_p_mat, target_pos_p_vec)
            T_finger_world_mat = T_gripper_pose_global @ T3_p_mat @ T2_rotate_global @ T1_translate_global
            
            mesh_world = mesh_ref.transform(T_finger_world_mat, inplace=False)
            mesh_world.compute_normals(cell_normals=True, point_normals=False, inplace=True, auto_orient_normals=True)
            current_step_finger_meshes[i] = mesh_world

        # ... 接触检测 ... (与原脚本相同)
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
                if pos_dp and neg_dp: return default_return # 交叉穿透

        for i in range(3):
            if not finger_contact_established[i] and contact_made_this_iter[i]:
                finger_contact_established[i] = True
                final_contact_pressures[i] = current_pressures[i]
                deformed_finger_meshes_at_contact[i] = current_step_finger_meshes[i].copy()
    
    if not all(finger_contact_established): return default_return

    # ... GII 计算 ... (与原脚本相同)
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
    
    # 假设物体质心在点云几何中心
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
    if faces_np_global is None or faces_np_global.size == 0: sys.exit("错误：未能创建手指表面。")
    width_translation_vector_global = np.array([0, finger_width, 0])
    bottom_node_idx = np.argmin(initial_coords_ref_global[:,0]); ref_mid_pt = initial_coords_ref_global[bottom_node_idx] + width_translation_vector_global/2.0
    T1_translate_global = create_transformation_matrix_opt8(None, -ref_mid_pt)
    rot_ref_to_local = np.array([[0,1,0],[0,0,1],[1,0,0]]); T2_rotate_global = create_transformation_matrix_opt8(rot_ref_to_local, None)

    # --- 2. 加载指定的点云和姿态 (MODIFIED) ---
    try:
        pcd = o3d.io.read_point_cloud(POINT_CLOUD_PATH)
        object_points_global_static = np.asarray(pcd.points)
        if object_points_global_static.shape[0] == 0: raise ValueError("点云为空")
        
        # 应用缩放
        object_points_global_static *= POINT_CLOUD_SCALE
        print(f"已应用点云缩放比例: {POINT_CLOUD_SCALE}")
            
        object_centroid_global_static = np.mean(object_points_global_static, axis=0)
        num_object_points_global_static = object_points_global_static.shape[0]
        print(f"成功加载点云: {os.path.basename(POINT_CLOUD_PATH)}, 点数: {num_object_points_global_static}")
    except Exception as e:
        sys.exit(f"错误: 加载点云文件 '{POINT_CLOUD_PATH}' 失败: {e}")

    try:
        T_gripper_pose_from_file = np.loadtxt(GRIPPER_POSE_PATH)
        if T_gripper_pose_from_file.shape != (4, 4): raise ValueError("姿态矩阵形状不为 4x4")
        print(f"成功加载夹爪姿态矩阵: {os.path.basename(GRIPPER_POSE_PATH)}")
    except Exception as e:
        sys.exit(f"错误: 加载姿态文件 '{GRIPPER_POSE_PATH}' 失败: {e}")
    
    # --- 2.5 应用手动位姿调整 (NEW) ---
    print("\n--- 应用手动位姿调整 ---")
    R_manual = euler_to_rotation_matrix(*MANUAL_ROTATION_EULER)
    t_manual = np.array(MANUAL_TRANSLATION)
    T_manual = np.identity(4)
    T_manual[:3, :3] = R_manual
    T_manual[:3, 3] = t_manual
    
    # 将手动调整应用于加载的姿态矩阵
    T_gripper_pose_global = T_gripper_pose_from_file @ T_manual
    
    print(f"手动平移: {MANUAL_TRANSLATION}")
    print(f"手动旋转 (欧拉角 °): {MANUAL_ROTATION_EULER}")
    print("--- 手动调整应用完毕 ---\n")

    # --- 3. 生成所有手指组合 (MODIFIED) ---
    all_finger_combinations = list(combinations(range(NUM_FINGER_SLOTS), 3))
    print(f"将为每个r值测试 {len(all_finger_combinations)} 种手指位置组合。")
    print(f"测试的r值: {[f'{val:.2f}' for val in R_VALUES_TO_TEST]}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    excel_file_path = os.path.join(OUTPUT_DIR, EXCEL_FILENAME)
    
    # --- 4. 初始场景预览 (MODIFIED) ---
    if SHOW_ONE_TIME_INITIAL_CONFIG_PREVIEW and R_VALUES_TO_TEST.size > 0 and all_finger_combinations:
        print("\n--- 显示一次性初始构型预览 ---")
        preview_r = R_VALUES_TO_TEST[0]
        preview_combo = all_finger_combinations[0]
        plotter = setup_pv_plotter(f"Initial Setup: r={preview_r:.2f}, pos={preview_combo}")
        plotter.add_mesh(pv.PolyData(object_points_global_static), color=object_point_color_viz, point_size=5.0, render_points_as_spheres=True)
        
        preview_tray_mesh = pv.Cylinder(center=(0, 0, -tray_height/2.0), direction=(0, 0, 1), radius=tray_radius, height=tray_height, resolution=30)
        preview_tray_mesh.transform(T_gripper_pose_global, inplace=True)
        plotter.add_mesh(preview_tray_mesh, color=tray_color_viz_pv, opacity=0.3)

        for i_finger_prev in range(len(preview_combo)):
            undeformed_verts_ref = np.vstack((initial_coords_ref_global, initial_coords_ref_global + width_translation_vector_global))
            undeformed_m_ref = pv.PolyData(undeformed_verts_ref, faces=faces_np_global)
            pos_idx_prev = preview_combo[i_finger_prev]
            angle_rad_prev = np.radians(pos_idx_prev * (360.0 / NUM_FINGER_SLOTS))
            rot_z_prev = create_rotation_matrix_z(angle_rad_prev + np.pi / 2.0)
            target_pos_prev = np.array([preview_r * np.cos(angle_rad_prev), preview_r * np.sin(angle_rad_prev), 0.0])
            T3_prev = create_transformation_matrix_opt8(rot_z_prev, target_pos_prev)
            T_finger_world_prev = T_gripper_pose_global @ T3_prev @ T2_rotate_global @ T1_translate_global
            undeformed_finger_world_prev = undeformed_m_ref.transform(T_finger_world_prev, inplace=False)
            plotter.add_mesh(undeformed_finger_world_prev, color=finger_color_viz, style='surface', opacity=0.8, show_edges=True, edge_color='gray')
        
        plotter.camera_position = 'iso'; plotter.camera.zoom(1.5)
        # 修改文本提示，告知用户需要手动关闭窗口
        plotter.add_text(f"Initial Preview. Close this window to continue...", position="upper_left", font_size=10)
        # 使用阻塞式的 show() 方法，脚本会在此暂停直到窗口被关闭
        plotter.show()
        # 用户关闭窗口后，脚本会自动继续，不再需要 time.sleep 和 plotter.close()
        print("--- 初始构型预览结束 ---")

    # --- 5. 主评估循环与数据聚合 (MODIFIED) ---
    print("\n--- 开始主评估循环 ---")
    total_configs_to_evaluate = len(R_VALUES_TO_TEST) * len(all_finger_combinations)
    evaluated_configs_count = 0
    start_time_total = time.time()
    
    # 用于聚合所有r值的结果
    final_results_for_excel = []

    for r_val in R_VALUES_TO_TEST:
        print(f"\n===== 测试半径 r = {r_val:.2f} =====")
        # 数据聚合字典：键为 (theta_small, theta_mid)，值为 GII 列表
        gii_aggregation_dict = {}

        for finger_combo in all_finger_combinations:
            evaluated_configs_count += 1
            print(f"  配置 {evaluated_configs_count}/{total_configs_to_evaluate}: r={r_val:.2f}, pos={finger_combo}")
            
            gii_value = evaluate_gii_for_config(r_val, finger_combo)
            
            if gii_value > 0:
                print(f"    有效抓取, GII = {gii_value:.4f}")
                # 计算相对角间距
                angles_deg = np.array([idx * (360.0 / NUM_FINGER_SLOTS) for idx in finger_combo])
                angles_deg = np.sort(angles_deg)
                
                delta_theta1 = angles_deg[1] - angles_deg[0]
                delta_theta2 = angles_deg[2] - angles_deg[1]
                delta_theta3 = 360.0 - (angles_deg[2] - angles_deg[0])
                
                thetas = sorted([delta_theta1, delta_theta2, delta_theta3])
                theta_small, theta_mid = thetas[0], thetas[1]
                
                # 将结果存入聚合字典
                key = (round(theta_small, 2), round(theta_mid, 2))
                if key not in gii_aggregation_dict:
                    gii_aggregation_dict[key] = []
                gii_aggregation_dict[key].append(gii_value)
            else:
                print(f"    无效抓取或GII为0")

        # 对当前r值的聚合结果求平均
        for key, gii_list in gii_aggregation_dict.items():
            avg_gii = np.mean(gii_list)
            num_samples = len(gii_list)
            final_results_for_excel.append({
                'r_value': r_val,
                'theta_small': key[0],
                'theta_mid': key[1],
                'avg_gii': avg_gii,
                'num_samples': num_samples
            })

    end_time_total = time.time()
    print(f"\n--- 所有配置评估完成，总耗时: {end_time_total - start_time_total:.2f} 秒 ---")

    # --- 6. 将最终聚合结果写入Excel (MODIFIED) ---
    if final_results_for_excel:
        final_df = pd.DataFrame(final_results_for_excel)
        print("\n最终聚合结果 DataFrame 预览:")
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
