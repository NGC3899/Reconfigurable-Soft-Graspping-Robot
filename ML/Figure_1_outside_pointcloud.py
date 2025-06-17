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
import os
import copy
import open3d as o3d
import random
from itertools import combinations # 用于生成手指位置组合
import pandas as pd # 用于数据处理和Excel导出

# --- 打印版本信息 ---
try:
    print(f"Figure_1_External_Cloud_Excel - 使用外部点云和Figure_1算法生成Excel (v2 - 修正定义)")
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

# --- 3. 文件路径定义 (参考 Opt_10.txt) ---
MODEL_PATH = 'best_mlp_model.pth'; X_SCALER_PATH = 'x_scaler.joblib'
Y_SCALER_PATH = 'y_scaler.joblib'; INITIAL_COORDS_PATH = 'initial_coordinates.txt'

GRASP_OUTPUTS_BASE_PATH = r"C:\Users\admin\Desktop\grasp_outputs" # <--- 修改为您的路径
RELATIVE_POSE_FILENAME = "relative_gripper_to_object_pose_CONDITIONAL_REORIENT.txt"
HIGH_RES_OBJECT_DESKTOP_PATH = r"C:\Users\admin\Desktop" # <--- 修改为您的路径
HIGH_RES_OBJECT_FILENAME = "Bird_Model.ply" # 外部点云文件名

RELATIVE_POSE_FILE_PATH = os.path.join(GRASP_OUTPUTS_BASE_PATH, RELATIVE_POSE_FILENAME)
HIGH_RES_OBJECT_PLY_PATH = os.path.join(HIGH_RES_OBJECT_DESKTOP_PATH, HIGH_RES_OBJECT_FILENAME)


# --- 4. Case Study 配置参数 ---
CASE_STUDY_OUTPUT_DIR = "grasp_case_study_gii_results_external_object"
EXCEL_FILENAME = "gii_analysis_external_object.xlsx"
DEBUG_PLOT_SUBDIR = "debug_visualization_plots_external" 

tray_radius = 60.0; tray_height = 1.0; finger_width = 10.0
tray_center = np.array([0.0, 0.0, -tray_height / 2.0]) 

TARGET_POINT_COUNT_FOR_SIM = 1000 
OBJECT_SCALE_FACTOR = 0.8         

NUM_FINGER_SLOTS_CASE_STUDY = 10 
# R_BOUNDS = (tray_radius * 0.2, tray_radius * 1) # 添加 R_BOUNDS 定义
R_BOUNDS = (56.7394, tray_radius * 1)
R_VALUES_TO_TEST = None # 将在主函数中基于 R_BOUNDS 和物体尺寸设置

INITIAL_PRESSURE_FOR_CONTACT_SEARCH = 100.0 
PRESSURE_STEP_FOR_CONTACT_SEARCH = 500.0    
MAX_PRESSURE_FOR_CONTACT_SEARCH = 40000.0   
max_pressure = 40000.0 
P_BOUNDS_CLIP = (0.0, max_pressure) 

collision_threshold = 1.0; overlap_threshold = 1e-3 
friction_coefficient = 0.5; eigenvalue_threshold = 1e-6
SHOW_ONE_TIME_INITIAL_CONFIG_PREVIEW = True
DOT_PROD_TOLERANCE_LOCAL = 1e-6 
CHARACTERISTIC_LENGTH_FOR_GII = 1.0 # 将被动态计算

# --- 美化参数 ---
finger_color_viz = '#ff7f0e'; tray_color_viz_pv = '#BDB7A4'
object_point_color_viz = '#1f77b4'; background_color_viz = '#EAEAEA' 
object_obb_color_pv = 'green' 
object_obb_color_o3d = (0.1, 0.9, 0.1) 
text_color_viz = 'black'; font_family_viz = 'times'
contact_normal_glyph_color = 'red'      
contact_point_glyph_color = 'magenta'   
contact_normal_glyph_scale = 5.0        


# --- 5. 辅助函数 ---
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

def transform_points_opt8(points, matrix): 
    if points.ndim == 1: points = points.reshape(1,-1)
    if points.shape[1] != 3: raise ValueError("Points shape error for transform_points_opt8")
    if matrix.shape != (4, 4): raise ValueError("Matrix shape error for transform_points_opt8")
    num_pts = points.shape[0]; h_points = np.hstack((points, np.ones((num_pts, 1))))
    t_h = h_points @ matrix.T; w = t_h[:, 3, np.newaxis]
    return np.divide(t_h[:, :3], w, out=np.zeros_like(t_h[:, :3]), where=w!=0)

def load_transformation_matrix_from_txt(file_path): 
    if not os.path.exists(file_path): print(f"错误: 变换矩阵文件 '{file_path}' 未找到."); return None
    try:
        matrix = np.loadtxt(file_path)
        if matrix.shape == (4, 4): print(f"成功从 '{file_path}' 加载变换矩阵."); return matrix
        else: print(f"错误: 从 '{file_path}' 加载的矩阵形状不是 (4, 4)，而是 {matrix.shape}."); return None
    except Exception as e: print(f"加载变换矩阵 '{file_path}' 时出错: {e}"); return None

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
        if norm_t1 < 1e-9:
            if np.allclose(np.abs(n), [1.,0.,0.]): t1 = np.array([0.,1.,0.])
            elif np.allclose(np.abs(n), [0.,1.,0.]): t1 = np.array([1.,0.,0.])
            elif np.allclose(np.abs(n), [0.,0.,1.]): t1 = np.array([1.,0.,0.])
            else: raise ValueError("Fallback t1 computation failed for a complex normal case.")
            norm_t1 = LA.norm(t1)
            if norm_t1 < 1e-9: raise ValueError("Ultimate fallback for t1 is zero or near zero.")
    t1 /= norm_t1; t2_temp = np.cross(n, t1); norm_t2 = LA.norm(t2_temp)
    if norm_t2 < 1e-9: raise ValueError("Cannot compute tangent 2 (t2_temp is zero or near zero).")
    t2 = t2_temp / norm_t2; return t1, t2

def get_rotation_matrix_between_vectors(vec1, vec2): # 确保此函数已定义
    a = vec1 / np.linalg.norm(vec1); b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b); c = np.dot(a, b); s = np.linalg.norm(v)
    if s < 1e-9: return np.identity(3) if c > 0 else create_rotation_matrix(np.array([1.0,0,0]) if np.abs(a[0])<0.9 else np.array([0,1.0,0]), np.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r_mat = np.identity(3) + vx + vx @ vx * ((1 - c) / (s ** 2)); return r_mat

def load_prediction_components(model_path,x_scaler_path,y_scaler_path,input_dim,output_dim,h1,h2,h3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"ML使用设备: {device}")
    model = MLPRegression(input_dim,output_dim,h1,h2,h3)
    try: model.load_state_dict(torch.load(model_path, map_location=device)); model.to(device); model.eval()
    except Exception as e: print(f"加载模型 {model_path} 出错: {e}"); return None,None,None,None
    scaler_X,scaler_y=None,None
    try: scaler_X=joblib.load(x_scaler_path); print(f"X Scaler 加载成功。")
    except FileNotFoundError: print(f"警告: 未找到 X scaler '{x_scaler_path}'。"); scaler_X = None
    except Exception as e: print(f"加载 X Scaler '{x_scaler_path}' 出错: {e}"); return None,None,None,None
    try: scaler_y=joblib.load(y_scaler_path); print(f"Y Scaler 加载成功。")
    except FileNotFoundError: print(f"警告: 未找到 Y scaler '{y_scaler_path}'。"); scaler_y = None
    except Exception as e: print(f"加载 Y Scaler '{y_scaler_path}' 出错: {e}"); return None,None,None,None
    return model,scaler_X,scaler_y,device

def predict_displacements_for_pressure(model,scaler_X,scaler_y,device,pressure_value): 
    if model is None: return None
    pressure_value=np.clip(pressure_value, P_BOUNDS_CLIP[0], P_BOUNDS_CLIP[1]); input_p=np.array([[pressure_value]],dtype=np.float32)
    if scaler_X:
        try: input_p_scaled = scaler_X.transform(input_p)
        except Exception: return None
    else: input_p_scaled = input_p
    input_tensor=torch.tensor(input_p_scaled,dtype=torch.float32).to(device); predicted_original_scale=None
    with torch.no_grad():
        try:
            predicted_scaled_tensor=model(input_tensor); predicted_scaled=predicted_scaled_tensor.cpu().numpy()
            if scaler_y:
                try: predicted_original_scale = scaler_y.inverse_transform(predicted_scaled)
                except Exception: return None
            else: predicted_original_scale = predicted_scaled
        except Exception: return None
    if predicted_original_scale is not None:
        if predicted_original_scale.shape[1] != OUTPUT_DIM: return None
        return predicted_original_scale.reshape(NODE_COUNT, 3)
    return None

def generate_finger_position_combinations_fixed_zero(num_slots): 
    if num_slots < 3: return []
    combos = []
    for combo_pair in combinations(range(1, num_slots), 2): 
        combos.append(tuple(sorted((0,) + combo_pair))) 
    return combos

def calculate_gii_multi_contact(contacts_info, object_centroid_for_gii, mu, eigenvalue_thresh,
                                normal_sign_multipliers=(1, 1, 1), 
                                characteristic_length=1.0): 
    if not contacts_info or len(contacts_info) < 2: return None
    all_wrenches = []; valid_contacts_for_gii = 0
    extended_multipliers = list(normal_sign_multipliers)
    while len(extended_multipliers) < 3: extended_multipliers.append(1) 
    if characteristic_length <= 1e-9: characteristic_length = 1.0
    
    for contact_details in contacts_info:
        if not isinstance(contact_details, (tuple, list)) or len(contact_details) < 6: continue
        pt_on_mesh = np.asarray(contact_details[4])
        original_normal_at_contact = np.asarray(contact_details[5])
        finger_idx = contact_details[2] 
        
        if pt_on_mesh.size != 3 or original_normal_at_contact.size != 3: continue
        
        multiplier = extended_multipliers[finger_idx] if 0 <= finger_idx < len(extended_multipliers) else 1
        effective_normal_finger = original_normal_at_contact * multiplier 
        
        norm_mag_effective = LA.norm(effective_normal_finger)
        if norm_mag_effective < 1e-9: continue
        
        n_contact = - (effective_normal_finger / norm_mag_effective) 

        try: t1, t2 = get_orthogonal_vectors(n_contact) 
        except ValueError: continue
        
        r_contact_vec = pt_on_mesh - object_centroid_for_gii
        d_list = [n_contact + mu * t1, n_contact - mu * t1, n_contact + mu * t2, n_contact - mu * t2]
        
        for d_force in d_list:
            if LA.norm(d_force) < 1e-9: continue
            torque = np.cross(r_contact_vec, d_force)
            normalized_torque = torque / characteristic_length 
            wrench = np.concatenate((d_force, normalized_torque))
            all_wrenches.append(wrench)
        valid_contacts_for_gii += 1
        
    if valid_contacts_for_gii < 2 or not all_wrenches: return None
    try: grasp_matrix_G = np.column_stack(all_wrenches)
    except ValueError: return None
    if grasp_matrix_G.shape[0] != 6: return None
    
    J = grasp_matrix_G @ grasp_matrix_G.T
    try:
        eigenvalues = LA.eigvalsh(J)
        positive_eigenvalues = eigenvalues[eigenvalues > eigenvalue_thresh]
        if not positive_eigenvalues.size: return 0.0
        lambda_min = np.min(positive_eigenvalues)
        lambda_max = np.max(positive_eigenvalues)
        if lambda_max < eigenvalue_thresh: return 0.0 
        return np.sqrt(lambda_min / lambda_max) if lambda_max > 1e-9 else 0.0
    except LA.LinAlgError: return None

# --- 全局变量 ---
initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global = [None]*5
object_points_global_static, object_centroid_global_static, num_object_points_global_static = None, None, 0
object_mesh_global_static_pv = None 
world_obb_object_global_o3d = None 
pv_obb_object_mesh_global = None 
faces_np_global, width_translation_vector_global = None, None
T1_translate_global, T2_rotate_global = None, None
T_pose_for_tray_display_and_finger_placement_global = None 
preview_tray_mesh = None 

# --- 可视化辅助函数 (来自 Figure_1.txt) ---
def setup_pv_plotter(title, window_size=[800,600], off_screen=False):
    plotter_theme = pv.themes.DocumentTheme()
    plotter_theme.font.family = font_family_viz; plotter_theme.font.color = text_color_viz
    plotter_theme.font.size = 10; plotter_theme.font.label_size = 8
    plotter_theme.background = pv.Color(background_color_viz)
    plotter = pv.Plotter(window_size=window_size, theme=plotter_theme, title=title, off_screen=off_screen)
    if not off_screen:
        plotter.enable_anti_aliasing('msaa', multi_samples=4)
    plotter.remove_all_lights(); plotter.enable_lightkit()
    return plotter

def evaluate_gii_for_case_study_with_pressure_iteration(r_value, finger_indices_tuple, normal_sign_multipliers=(1,1,1)):
    global initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global, \
           object_points_global_static, object_centroid_global_static, num_object_points_global_static, \
           faces_np_global, width_translation_vector_global, T1_translate_global, T2_rotate_global, \
           T_pose_for_tray_display_and_finger_placement_global, NUM_FINGER_SLOTS_CASE_STUDY, \
           friction_coefficient, eigenvalue_threshold, collision_threshold, overlap_threshold, \
           INITIAL_PRESSURE_FOR_CONTACT_SEARCH, PRESSURE_STEP_FOR_CONTACT_SEARCH, MAX_PRESSURE_FOR_CONTACT_SEARCH, \
           NODE_COUNT, CHARACTERISTIC_LENGTH_FOR_GII
           
    default_return = 0.0, np.full(3, -1.0, dtype=float), [None, None, None], []
    if not all(v is not None for v in [initial_coords_ref_global, model_global, object_points_global_static, faces_np_global, width_translation_vector_global, T1_translate_global, T2_rotate_global, T_pose_for_tray_display_and_finger_placement_global]): 
        print("评估函数：一个或多个全局变量未初始化。")
        return default_return
    if num_object_points_global_static == 0 : return default_return
    if len(set(finger_indices_tuple)) < 3: return default_return 

    current_pressures = np.full(3, INITIAL_PRESSURE_FOR_CONTACT_SEARCH, dtype=float)
    final_contact_pressures = np.full(3, -1.0, dtype=float)
    finger_contact_established = [False, False, False]
    deformed_finger_meshes_at_contact = [None] * 3
    max_total_iterations = int(MAX_PRESSURE_FOR_CONTACT_SEARCH / PRESSURE_STEP_FOR_CONTACT_SEARCH) * 3 + 20 
    current_total_iterations = 0

    while True: 
        current_total_iterations += 1
        if current_total_iterations > max_total_iterations: return default_return 

        pressure_increased_this_step = False
        for i in range(3):
            if not finger_contact_established[i]:
                if current_pressures[i] < MAX_PRESSURE_FOR_CONTACT_SEARCH:
                    current_pressures[i] += PRESSURE_STEP_FOR_CONTACT_SEARCH
                    current_pressures[i] = min(current_pressures[i], MAX_PRESSURE_FOR_CONTACT_SEARCH)
                    pressure_increased_this_step = True
        
        current_step_finger_meshes = [None] * 3 
        all_meshes_generated_ok = True
        for i in range(3): 
            if finger_contact_established[i] and deformed_finger_meshes_at_contact[i] is not None:
                current_step_finger_meshes[i] = deformed_finger_meshes_at_contact[i]; continue 

            pressure_to_eval = current_pressures[i]
            displacements = predict_displacements_for_pressure(model_global, scaler_X_global, scaler_y_global, device_global, pressure_to_eval)
            if displacements is None: all_meshes_generated_ok = False; break
            
            deformed_c1 = initial_coords_ref_global + displacements
            deformed_c2 = initial_coords_ref_global + width_translation_vector_global + displacements
            s_d_c1 = sort_points_spatially(deformed_c1); s_d_c2 = sort_points_spatially(deformed_c2)
            if s_d_c1 is None or s_d_c2 is None: all_meshes_generated_ok = False; break
            
            vertices = np.vstack((s_d_c1, s_d_c2))
            if faces_np_global is None or faces_np_global.size == 0 : all_meshes_generated_ok = False; break
            try: mesh_ref = pv.PolyData(vertices, faces=faces_np_global)
            except Exception: all_meshes_generated_ok = False; break
            
            pos_idx = finger_indices_tuple[i]
            angle_deg_p = pos_idx * (360.0 / NUM_FINGER_SLOTS_CASE_STUDY) 
            angle_rad_p = np.radians(angle_deg_p)
            rot_z_p_mat = create_rotation_matrix_z(angle_rad_p + np.pi / 2.0)
            target_pos_p_vec = np.array([r_value * np.cos(angle_rad_p), r_value * np.sin(angle_rad_p), 0.0])
            T3_p_mat = create_transformation_matrix_opt8(rot_z_p_mat, target_pos_p_vec)
            
            T_finger_world_mat = T_pose_for_tray_display_and_finger_placement_global @ T3_p_mat @ T2_rotate_global @ T1_translate_global
            
            try:
                mesh_world = mesh_ref.transform(T_finger_world_mat, inplace=False)
                if mesh_world is None or mesh_world.n_points == 0: all_meshes_generated_ok = False; break
                mesh_world.clean(inplace=True)
                if mesh_world.n_cells > 0:
                    mesh_world.compute_normals(cell_normals=True, point_normals=False, inplace=True, auto_orient_normals=True, non_manifold_traversal=False) 
                current_step_finger_meshes[i] = mesh_world
            except Exception: all_meshes_generated_ok = False; break
        
        if not all_meshes_generated_ok: return default_return 

        for i_update_mesh in range(3):
            if current_step_finger_meshes[i_update_mesh] is not None:
                deformed_finger_meshes_at_contact[i_update_mesh] = current_step_finger_meshes[i_update_mesh].copy()

        has_overall_overlap_this_iteration = False
        contact_made_by_finger_this_iteration = [False] * 3
        finger_dot_products_this_iteration = [[] for _ in range(3)]

        if object_points_global_static is not None:
            for obj_pt_idx, obj_point in enumerate(object_points_global_static):
                closest_dist_this_obj_pt_to_any_finger = float('inf')
                winning_finger_for_this_obj_pt = -1
                winning_pt_on_mesh_this_obj_pt = None
                winning_normal_this_obj_pt = None
                for finger_idx, finger_mesh in enumerate(current_step_finger_meshes): 
                    if finger_mesh is None or finger_mesh.n_cells == 0: continue
                    has_cell_normals = ('Normals' in finger_mesh.cell_data and 
                                        finger_mesh.cell_data['Normals'] is not None and 
                                        len(finger_mesh.cell_data['Normals']) == finger_mesh.n_cells)
                    try:
                        closest_cell_id, pt_on_mesh_candidate = finger_mesh.find_closest_cell(obj_point, return_closest_point=True)
                        if closest_cell_id < 0 or closest_cell_id >= finger_mesh.n_cells: continue
                        dist = LA.norm(obj_point - pt_on_mesh_candidate)
                        if dist < closest_dist_this_obj_pt_to_any_finger:
                            closest_dist_this_obj_pt_to_any_finger = dist
                            winning_finger_for_this_obj_pt = finger_idx
                            winning_pt_on_mesh_this_obj_pt = pt_on_mesh_candidate
                            if has_cell_normals: winning_normal_this_obj_pt = finger_mesh.cell_normals[closest_cell_id]
                        if dist < collision_threshold: 
                            contact_made_by_finger_this_iteration[finger_idx] = True
                    except RuntimeError: continue 
                    except Exception: continue
                
                if winning_finger_for_this_obj_pt != -1:
                    if closest_dist_this_obj_pt_to_any_finger < overlap_threshold: 
                        has_overall_overlap_this_iteration = True; break 
                    elif closest_dist_this_obj_pt_to_any_finger < collision_threshold:
                        if winning_normal_this_obj_pt is not None and LA.norm(winning_normal_this_obj_pt) > 1e-9:
                            vec_cto = obj_point - winning_pt_on_mesh_this_obj_pt
                            if LA.norm(vec_cto) > 1e-9:
                                dot_prod = np.dot(vec_cto / LA.norm(vec_cto), winning_normal_this_obj_pt / LA.norm(winning_normal_this_obj_pt))
                                finger_dot_products_this_iteration[winning_finger_for_this_obj_pt].append(dot_prod)
            if has_overall_overlap_this_iteration: return default_return 

        finger_intersects_this_iteration = [False] * 3
        if not has_overall_overlap_this_iteration:
            for i in range(3):
                if contact_made_by_finger_this_iteration[i] and finger_dot_products_this_iteration[i]:
                    pos_dp = any(dp > DOT_PROD_TOLERANCE_LOCAL for dp in finger_dot_products_this_iteration[i])
                    neg_dp = any(dp < -DOT_PROD_TOLERANCE_LOCAL for dp in finger_dot_products_this_iteration[i])
                    if pos_dp and neg_dp: finger_intersects_this_iteration[i] = True
        if any(finger_intersects_this_iteration): return default_return

        for i in range(3):
            if not finger_contact_established[i] and contact_made_by_finger_this_iteration[i]:
                finger_contact_established[i] = True
                final_contact_pressures[i] = current_pressures[i]
        
        if all(finger_contact_established): break 
        if not pressure_increased_this_step and not all(finger_contact_established): return default_return
        for i in range(3):
            if not finger_contact_established[i] and current_pressures[i] >= MAX_PRESSURE_FOR_CONTACT_SEARCH:
                return default_return

    if not all(finger_contact_established): return default_return 

    best_contact_info_for_gii_final = [None] * 3
    min_dist_per_finger_final = [float('inf')] * 3
    if object_points_global_static is not None:
        for finger_idx_gii, finger_mesh_gii in enumerate(deformed_finger_meshes_at_contact): 
            if finger_mesh_gii is None or finger_mesh_gii.n_cells == 0: continue
            has_cell_normals = ('Normals' in finger_mesh_gii.cell_data and 
                                finger_mesh_gii.cell_data['Normals'] is not None and 
                                len(finger_mesh_gii.cell_data['Normals']) == finger_mesh_gii.n_cells)
            if not has_cell_normals: continue

            for obj_pt_idx, obj_point in enumerate(object_points_global_static):
                try:
                    closest_cell_id_gii, pt_on_mesh_gii = finger_mesh_gii.find_closest_cell(obj_point, return_closest_point=True)
                    if closest_cell_id_gii < 0 or closest_cell_id_gii >= finger_mesh_gii.n_cells: continue
                    dist_gii = LA.norm(obj_point - pt_on_mesh_gii)
                    if dist_gii < collision_threshold and dist_gii < min_dist_per_finger_final[finger_idx_gii]:
                        normal_gii_candidate = finger_mesh_gii.cell_normals[closest_cell_id_gii]
                        if normal_gii_candidate is not None and LA.norm(normal_gii_candidate) > 1e-9:
                            min_dist_per_finger_final[finger_idx_gii] = dist_gii
                            best_contact_info_for_gii_final[finger_idx_gii] = (dist_gii, obj_pt_idx, finger_idx_gii, closest_cell_id_gii, pt_on_mesh_gii, normal_gii_candidate)
                except Exception: continue
    
    valid_gii_contacts_final_state = [info for info in best_contact_info_for_gii_final if info is not None]
    
    if len(valid_gii_contacts_final_state) < 2: 
        return 0.0, final_contact_pressures, deformed_finger_meshes_at_contact, [] 

    if object_centroid_global_static is None: 
        return 0.0, final_contact_pressures, deformed_finger_meshes_at_contact, valid_gii_contacts_final_state

    gii = calculate_gii_multi_contact(valid_gii_contacts_final_state, object_centroid_global_static,
                                      friction_coefficient, eigenvalue_threshold,
                                      normal_sign_multipliers=normal_sign_multipliers, 
                                      characteristic_length=CHARACTERISTIC_LENGTH_FOR_GII) 
    
    final_gii_value = gii if (gii is not None and gii > 1e-9) else 0.0
    return final_gii_value, final_contact_pressures, deformed_finger_meshes_at_contact, valid_gii_contacts_final_state


# --- 主脚本 ---
if __name__ == '__main__':
    print("--- Script Start ---")
    # --- 1. 初始化模型和手指参考坐标 ---
    initial_coords_ref_global = load_initial_coordinates(INITIAL_COORDS_PATH, NODE_COUNT)
    model_global,scaler_X_global,scaler_y_global,device_global = load_prediction_components(MODEL_PATH,X_SCALER_PATH,Y_SCALER_PATH,INPUT_DIM,OUTPUT_DIM,HIDDEN_LAYER_1,HIDDEN_LAYER_2,HIDDEN_LAYER_3)
    if initial_coords_ref_global is None or model_global is None: sys.exit("错误：未能初始化手指模型或坐标。")
    faces_np_global = create_faces_array(NODE_COUNT)
    if faces_np_global is None or faces_np_global.size == 0: sys.exit("错误：未能创建手指表面。")
    width_translation_vector_global = np.array([0, finger_width, 0])
    bottom_node_idx = np.argmin(initial_coords_ref_global[:,0]); ref_mid_pt = initial_coords_ref_global[bottom_node_idx] + width_translation_vector_global/2.0
    T1_translate_global = create_transformation_matrix_opt8(None,-ref_mid_pt)
    rot_ref_to_local = np.array([[0,1,0],[0,0,1],[1,0,0]]); T2_rotate_global = create_transformation_matrix_opt8(rot_ref_to_local,None)

    # --- 2. 加载外部点云和位姿 (参考 Opt_10.txt 逻辑) ---
    T_gn_gripper_TO_gn_object_meters = load_transformation_matrix_from_txt(RELATIVE_POSE_FILE_PATH)
    if T_gn_gripper_TO_gn_object_meters is None: sys.exit("错误：未能加载物体相对于GraspNet夹爪的位姿。")

    T_original_tray_pose_world = create_transformation_matrix_opt8(np.identity(3), tray_center) 
    T_gn_object_TO_gn_gripper_meters = np.linalg.inv(T_gn_gripper_TO_gn_object_meters)
    T_imported_obj_relative_to_tray_mm = copy.deepcopy(T_gn_object_TO_gn_gripper_meters)
    T_imported_obj_relative_to_tray_mm[:3, 3] *= 1000.0 
    T_original_object_pose_world = T_original_tray_pose_world @ T_imported_obj_relative_to_tray_mm

    # "位姿对调" 逻辑 (与 Opt_10_GA_ExternalObject 一致)
    T_object_target_world_pose = T_original_tray_pose_world 
    _T_tray_ref_before_fix = T_original_object_pose_world   
    
    angle_rad_fix = np.pi / 2 
    cos_fix = np.cos(angle_rad_fix); sin_fix = np.sin(angle_rad_fix)
    T_local_orientation_fix_for_tray_geometry = np.array([
        [cos_fix, 0, sin_fix, 0], [0, 1, 0, 0], [-sin_fix,0, cos_fix, 0], [0, 0, 0, 1]
    ])
    T_actual_tray_geometry_world = _T_tray_ref_before_fix @ T_local_orientation_fix_for_tray_geometry
    T_pose_for_tray_display_and_finger_placement_global = T_actual_tray_geometry_world 

    T_tray_axes_vis_world = _T_tray_ref_before_fix 
    T_object_axes_vis_world = T_object_target_world_pose

    print(f"\n加载高精度物体点云: {HIGH_RES_OBJECT_PLY_PATH}")
    if not os.path.exists(HIGH_RES_OBJECT_PLY_PATH): sys.exit(f"错误: {HIGH_RES_OBJECT_PLY_PATH} 未找到")
    try:
        high_res_object_o3d = o3d.io.read_point_cloud(HIGH_RES_OBJECT_PLY_PATH)
        if not high_res_object_o3d.has_points(): sys.exit(f"错误: {HIGH_RES_OBJECT_PLY_PATH} 为空")
    except Exception as e: sys.exit(f"加载 {HIGH_RES_OBJECT_PLY_PATH} 错: {e}")

    current_high_res_points_mm_orig_frame = np.asarray(high_res_object_o3d.points)
    if OBJECT_SCALE_FACTOR != 1.0: 
        current_high_res_points_mm_orig_frame = current_high_res_points_mm_orig_frame * OBJECT_SCALE_FACTOR
    
    centroid_high_res_mm = np.mean(current_high_res_points_mm_orig_frame, axis=0)
    points_high_res_centered_mm = current_high_res_points_mm_orig_frame - centroid_high_res_mm
    
    points_high_res_centered_aligned_mm = points_high_res_centered_mm 
    R_local_align = np.eye(3) 
    if points_high_res_centered_mm.shape[0] > 0:
        local_pcd_for_alignment = o3d.geometry.PointCloud()
        local_pcd_for_alignment.points = o3d.utility.Vector3dVector(points_high_res_centered_mm)
        try:
            local_obb = local_pcd_for_alignment.get_oriented_bounding_box() 
            longest_extent_idx = np.argmax(local_obb.extent)
            local_longest_axis_vec = local_obb.R[:, longest_extent_idx] 
            target_local_x_axis = np.array([1.0, 0.0, 0.0]) 
            R_local_align_calc = get_rotation_matrix_between_vectors(local_longest_axis_vec, target_local_x_axis)
            if R_local_align_calc is not None:
                R_local_align = R_local_align_calc 
                T_local_align_homogeneous = create_transformation_matrix_opt8(R_local_align, None)
                points_high_res_centered_aligned_mm = transform_points_opt8(points_high_res_centered_mm, T_local_align_homogeneous)
        except Exception as e_obb_align: print(f"  局部OBB对齐过程中出错: {e_obb_align}。将使用原始居中点云。"); 

    object_points_transformed_full_mm = transform_points_opt8(points_high_res_centered_aligned_mm, T_object_target_world_pose)

    final_sampled_points_mm = None; sampled_colors_uint8_pv = None; sampled_colors_float_o3d = None
    original_hr_colors_float_o3d = np.asarray(high_res_object_o3d.colors) if high_res_object_o3d.has_colors() else None
    
    if len(object_points_transformed_full_mm) > TARGET_POINT_COUNT_FOR_SIM:
        indices = np.random.choice(len(object_points_transformed_full_mm), TARGET_POINT_COUNT_FOR_SIM, replace=False)
        final_sampled_points_mm = object_points_transformed_full_mm[indices]
        if original_hr_colors_float_o3d is not None and indices.size > 0 and \
           original_hr_colors_float_o3d.shape[0] == len(current_high_res_points_mm_orig_frame):
            try:
                sampled_colors_float_o3d = original_hr_colors_float_o3d[indices] 
                if sampled_colors_float_o3d.ndim == 2 and sampled_colors_float_o3d.shape[1] == 3:
                    sampled_colors_uint8_pv = (sampled_colors_float_o3d * 255).astype(np.uint8)
                else: sampled_colors_uint8_pv = None; sampled_colors_float_o3d = None
            except IndexError:
                print("警告：颜色采样时索引不匹配，颜色可能不正确。")
                sampled_colors_uint8_pv = None; sampled_colors_float_o3d = None
        else:
             sampled_colors_uint8_pv = None; sampled_colors_float_o3d = None
    elif len(object_points_transformed_full_mm) > 0 :
        final_sampled_points_mm = object_points_transformed_full_mm
        if original_hr_colors_float_o3d is not None and original_hr_colors_float_o3d.shape[0] == final_sampled_points_mm.shape[0]: 
            if original_hr_colors_float_o3d.ndim == 2 and original_hr_colors_float_o3d.shape[1] == 3:
                sampled_colors_float_o3d = original_hr_colors_float_o3d; sampled_colors_uint8_pv = (original_hr_colors_float_o3d * 255).astype(np.uint8)
            else: sampled_colors_uint8_pv = None; sampled_colors_float_o3d = None
    else: sys.exit(f"错误: 变换后的高精度点云不含点.")
    
    object_points_global_static = final_sampled_points_mm
    object_mesh_global_static_pv = pv.PolyData(object_points_global_static) 
    if sampled_colors_uint8_pv is not None: object_mesh_global_static_pv.point_data['colors'] = sampled_colors_uint8_pv
    
    object_centroid_global_static = np.mean(object_points_global_static, axis=0) 
    num_object_points_global_static = object_points_global_static.shape[0]
    print(f"\n已加载并处理 '{HIGH_RES_OBJECT_FILENAME}'。用于仿真点数: {num_object_points_global_static}，质心: {object_centroid_global_static.round(3)}")

    if num_object_points_global_static > 0:
        min_coords = np.min(object_points_global_static, axis=0); max_coords = np.max(object_points_global_static, axis=0)
        aabb_diag_length = LA.norm(max_coords - min_coords)
        CHARACTERISTIC_LENGTH_FOR_GII = aabb_diag_length / 2.0 
        if CHARACTERISTIC_LENGTH_FOR_GII < 1e-6: CHARACTERISTIC_LENGTH_FOR_GII = 1.0; print(f"警告: 计算得到的AABB对角线长度非常小。特征长度回退到 {CHARACTERISTIC_LENGTH_FOR_GII}")
        print(f"动态计算的特征长度 (AABB对角线一半) 用于GII归一化: {CHARACTERISTIC_LENGTH_FOR_GII:.3f} mm")
    else: CHARACTERISTIC_LENGTH_FOR_GII = 1.0; print(f"警告: 物体点云为空，无法动态计算特征长度。使用默认值: {CHARACTERISTIC_LENGTH_FOR_GII}")

    if num_object_points_global_static > 0:
        o3d_temp_pcd_for_obb = o3d.geometry.PointCloud(); o3d_temp_pcd_for_obb.points = o3d.utility.Vector3dVector(object_points_global_static)
        world_obb_object_global_o3d = o3d_temp_pcd_for_obb.get_oriented_bounding_box(); world_obb_object_global_o3d.color = object_obb_color_o3d
        pv_obb_object_mesh_global = pv.Cube(center=(0,0,0), x_length=world_obb_object_global_o3d.extent[0], y_length=world_obb_object_global_o3d.extent[1], z_length=world_obb_object_global_o3d.extent[2])
        T_pv_obb_transform = np.eye(4); T_pv_obb_transform[:3,:3] = world_obb_object_global_o3d.R; T_pv_obb_transform[:3,3] = world_obb_object_global_o3d.center
        pv_obb_object_mesh_global.transform(T_pv_obb_transform, inplace=True)
    else: world_obb_object_global_o3d = None; pv_obb_object_mesh_global = None;

    preview_tray_mesh = pv.Cylinder(center=(0, 0, -tray_height/2.0), direction=(0, 0, 1), radius=tray_radius, height=tray_height, resolution=30)
    preview_tray_mesh = preview_tray_mesh.transform(T_actual_tray_geometry_world, inplace=False) 
    tray_pv_global = preview_tray_mesh.copy() # 用于调试可视化

    # --- 3. Figure_1 的案例分析和Excel导出逻辑 ---
    if object_points_global_static is not None and num_object_points_global_static > 0:
        T_world_to_tray_ref = np.linalg.inv(T_pose_for_tray_display_and_finger_placement_global)
        object_centroid_in_tray_frame = transform_points_opt8(object_centroid_global_static, T_world_to_tray_ref)
        if world_obb_object_global_o3d:
            obj_extent_xy = np.sqrt(world_obb_object_global_o3d.extent[0]**2 + world_obb_object_global_o3d.extent[1]**2) / 2
            min_r_heuristic = obj_extent_xy * 0.7 
        else:
            min_r_heuristic = tray_radius * 0.2 
        # min_r_val_fig1 = np.clip(min_r_heuristic, R_BOUNDS[0] * 0.8, R_BOUNDS[1] * 0.8) 
        min_r_val_fig1 = R_BOUNDS[0]
        max_r_val_fig1 = R_BOUNDS[1] 
        R_VALUES_TO_TEST = np.linspace(min_r_val_fig1, max_r_val_fig1, 2) 
        print(f"用于Excel生成的R值范围: {R_VALUES_TO_TEST.round(2)}")
    else:
        R_VALUES_TO_TEST = np.linspace(R_BOUNDS[0], R_BOUNDS[1] * 0.9, 3) # Fallback
        print(f"警告: 未能基于物体尺寸确定R值范围，使用默认范围: {R_VALUES_TO_TEST.round(2)}")


    all_finger_combinations = generate_finger_position_combinations_fixed_zero(NUM_FINGER_SLOTS_CASE_STUDY)
    print(f"将为每个r值测试 {len(all_finger_combinations)} 种手指位置组合 (0号位固定)。")
    
    os.makedirs(CASE_STUDY_OUTPUT_DIR, exist_ok=True)
    excel_file_path = os.path.join(CASE_STUDY_OUTPUT_DIR, EXCEL_FILENAME)
    column_names = ['r_value'] + [f'slot_{i}_avg_gii' for i in range(NUM_FINGER_SLOTS_CASE_STUDY)]
    all_excel_rows_so_far = []

    if SHOW_ONE_TIME_INITIAL_CONFIG_PREVIEW and R_VALUES_TO_TEST is not None and R_VALUES_TO_TEST.size > 0 and all_finger_combinations:
        print("\n--- 显示一次性初始构型预览 ---")
        preview_r = R_VALUES_TO_TEST[0]
        preview_combo = all_finger_combinations[0] if all_finger_combinations else (0,1,2) 
        
        plotter_preview = setup_pv_plotter(f"初始设置预览: r={preview_r:.2f}, pos={preview_combo}", off_screen=False)
        if preview_tray_mesh: plotter_preview.add_mesh(preview_tray_mesh, color=tray_color_viz_pv, opacity=0.3, name="preview_tray_mesh")
        
        if object_mesh_global_static_pv: # 使用加载的点云
            if 'colors' in object_mesh_global_static_pv.point_data:
                plotter_preview.add_mesh(object_mesh_global_static_pv, scalars='colors', rgb=True, style='points', point_size=3)
            else:
                plotter_preview.add_mesh(object_mesh_global_static_pv, color=object_point_color_viz, style='points', point_size=3)
        
        for i_finger_prev in range(len(preview_combo)):
            undeformed_verts_ref = np.vstack((initial_coords_ref_global, initial_coords_ref_global + width_translation_vector_global))
            undeformed_m_ref = pv.PolyData(undeformed_verts_ref, faces=faces_np_global)
            pos_idx_prev = preview_combo[i_finger_prev]
            angle_deg_prev = pos_idx_prev * (360.0 / NUM_FINGER_SLOTS_CASE_STUDY)
            angle_rad_prev = np.radians(angle_deg_prev)
            rot_z_prev = create_rotation_matrix_z(angle_rad_prev + np.pi / 2.0)
            target_pos_prev = np.array([preview_r * np.cos(angle_rad_prev), preview_r * np.sin(angle_rad_prev), 0.0])
            T3_prev = create_transformation_matrix_opt8(rot_z_prev, target_pos_prev)
            T_finger_world_prev = T_pose_for_tray_display_and_finger_placement_global @ T3_prev @ T2_rotate_global @ T1_translate_global
            undeformed_finger_world_prev = undeformed_m_ref.transform(T_finger_world_prev, inplace=False)
            plotter_preview.add_mesh(undeformed_finger_world_prev, color=finger_color_viz, style='surface', opacity=0.8, smooth_shading=True, show_edges=True, edge_color='gray', line_width=0.5)
        
        plotter_preview.camera_position = 'iso'; plotter_preview.camera.zoom(1.2)
        if object_centroid_global_static is not None: plotter_preview.set_focus(object_centroid_global_static)
        plotter_preview.add_text(f"初始预览: r={preview_r:.2f}, pos={preview_combo}", position="upper_left", font_size=10)
        plotter_preview.show(title="初始构型预览", auto_close=False) 
        plotter_preview.close()
        print("--- 初始构型预览结束 ---")

    print("--- 开始主评估循环 (Figure_1 逻辑) ---")
    total_configs_to_evaluate = len(R_VALUES_TO_TEST) * len(all_finger_combinations)
    evaluated_configs_count = 0
    start_time_total = time.time()

    for r_val_iter in R_VALUES_TO_TEST:
        print(f"\n===== 测试半径 r = {r_val_iter:.2f} =====")
        gii_lists_for_current_r = {slot_idx: [] for slot_idx in range(NUM_FINGER_SLOTS_CASE_STUDY)}
        
        for finger_combo_indices_iter in all_finger_combinations:
            evaluated_configs_count += 1
            print(f"  配置 {evaluated_configs_count}/{total_configs_to_evaluate}: r={r_val_iter:.2f}, pos={finger_combo_indices_iter}")
            
            eval_results_main = evaluate_gii_for_case_study_with_pressure_iteration(r_val_iter, finger_combo_indices_iter) 
            gii_value, pressures_at_contact, _, _ = eval_results_main 

            contact_pressures_str = "N/A"
            if pressures_at_contact is not None and hasattr(pressures_at_contact, 'size') and pressures_at_contact.size == 3:
                 contact_pressures_str = f"[{pressures_at_contact[0]:.0f}, {pressures_at_contact[1]:.0f}, {pressures_at_contact[2]:.0f}]"
            
            if gii_value > 0: 
                print(f"    GII = {gii_value:.4f} (最终接触压力: {contact_pressures_str})")
                for i_finger_in_combo in range(len(finger_combo_indices_iter)):
                    slot_idx_in_combo = finger_combo_indices_iter[i_finger_in_combo]
                    if 0 <= slot_idx_in_combo < NUM_FINGER_SLOTS_CASE_STUDY:
                         gii_lists_for_current_r[slot_idx_in_combo].append(gii_value)
            else:
                print(f"    无效抓取或GII为0 (GII={gii_value:.4f}, 最终尝试压力: {contact_pressures_str})")
        
        current_r_row_data = {'r_value': r_val_iter}
        for slot_idx_report in range(NUM_FINGER_SLOTS_CASE_STUDY):
            gii_list_for_slot = gii_lists_for_current_r[slot_idx_report]
            avg_gii_report = np.mean(gii_list_for_slot) if gii_list_for_slot else 0.0
            current_r_row_data[f'slot_{slot_idx_report}_avg_gii'] = avg_gii_report
        all_excel_rows_so_far.append(current_r_row_data)
        
        df_to_write = pd.DataFrame(all_excel_rows_so_far, columns=column_names)
        try:
            df_to_write.to_excel(excel_file_path, index=False, float_format="%.4f")
            print(f"  数据已更新到Excel: {os.path.abspath(excel_file_path)} (r={r_val_iter:.2f} 完成)")
        except Exception as e: print(f"  保存Excel文件失败 (r={r_val_iter:.2f}): {e}")

    end_time_total = time.time()
    print(f"\n--- 所有配置评估完成，总耗时: {end_time_total - start_time_total:.2f} 秒 ---")
    print("\n最终结果DataFrame预览:")
    if all_excel_rows_so_far:
        final_df_results = pd.DataFrame(all_excel_rows_so_far, columns=column_names)
        print(final_df_results)
    else: print("没有结果可供预览。")
    
    print("\n脚本结束。")
    try:
        if 'preview_tray_mesh' in globals() and preview_tray_mesh is not None: del preview_tray_mesh; preview_tray_mesh = None
    except Exception: pass
    try: pv.close_all()
    except Exception: pass
    import gc
    gc.collect()
