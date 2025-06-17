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
# import vtk

# --- 打印版本信息 ---
try:
    print(f"Case_Study_GII_Analysis_Script (Corrected Definitions + Wrench Norm)")
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
MODEL_PATH = 'best_mlp_model.pth'; X_SCALER_PATH = 'x_scaler.joblib'
Y_SCALER_PATH = 'y_scaler.joblib'; INITIAL_COORDS_PATH = 'initial_coordinates.txt'

# --- 4. Case Study 配置参数 ---
CASE_STUDY_OUTPUT_DIR = "grasp_case_study_gii_results_final_debug"
EXCEL_FILENAME = "gii_analysis_final_debug.xlsx"
DEBUG_PLOT_SUBDIR = "debug_visualization_plots" # 定义 DEBUG_PLOT_SUBDIR
tray_radius = 60.0; tray_height = 1.0; finger_width = 10.0
SPHERE_OBJECT_RADIUS = 25.0
SPHERE_OBJECT_CENTER_OFFSET_Z = (SPHERE_OBJECT_RADIUS + 35.0)
SPHERE_N_POINTS = 500
NUM_FINGER_SLOTS_CASE_STUDY = 10
min_r_val = SPHERE_OBJECT_RADIUS + finger_width * 0.1
# min_r_val = SPHERE_OBJECT_RADIUS + 9.1428
# min_r_val = 45
max_r_val = tray_radius
# max_r_val = 50
R_VALUES_TO_TEST = np.linspace(min_r_val, max_r_val, 10)
INITIAL_PRESSURE_FOR_CONTACT_SEARCH = 100.0
PRESSURE_STEP_FOR_CONTACT_SEARCH = 500.0
MAX_PRESSURE_FOR_CONTACT_SEARCH = 40000.0
collision_threshold = 1.0; overlap_threshold = 1e-3
friction_coefficient = 0.5; eigenvalue_threshold = 1e-6
P_BOUNDS_CLIP = (0.0, 40000.0)
SHOW_ONE_TIME_INITIAL_CONFIG_PREVIEW = True
PREVIEW_DURATION = 0.5
DOT_PROD_TOLERANCE_LOCAL = 1e-6 # 确保定义

# --- 美化参数 ---
finger_color_viz = '#ff7f0e'; tray_color_viz_pv = '#BDB7A4'
object_point_color_viz = '#1f77b4'; background_color_viz = '#EAEAEA'
text_color_viz = 'black'; font_family_viz = 'times'
contact_normal_glyph_color = 'red'       # 定义 contact_normal_glyph_color
contact_point_glyph_color = 'magenta'    # 定义 contact_point_glyph_color
contact_normal_glyph_scale = 5.0         # 定义 contact_normal_glyph_scale


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
    if points.shape[1] != 3: raise ValueError("Points shape error")
    if matrix.shape != (4, 4): raise ValueError("Matrix shape error")
    num_pts = points.shape[0]; h_points = np.hstack((points, np.ones((num_pts, 1))))
    t_h = h_points @ matrix.T; w = t_h[:, 3, np.newaxis]
    return np.divide(t_h[:, :3], w, out=np.zeros_like(t_h[:, :3]), where=w!=0)
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
def create_sphere_point_cloud(radius, center, n_points=500):
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    pcd = sphere_mesh.sample_points_uniformly(number_of_points=n_points)
    pcd.translate(center, relative=False)
    return np.asarray(pcd.points)
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
        n_contact = effective_normal_finger / norm_mag_effective
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
        print(f"DEBUG: Number of positive eigenvalues > {eigenvalue_thresh}: {len(positive_eigenvalues)}")
        if not positive_eigenvalues.size: return 0.0
        lambda_min = np.min(positive_eigenvalues)
        lambda_max = np.max(positive_eigenvalues)
        if lambda_max < 1e-9: return 0.0
        return np.sqrt(lambda_min / lambda_max)
    except LA.LinAlgError: return None

# --- 全局变量 ---
initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global = [None]*5
object_points_global_static, object_centroid_global_static, num_object_points_global_static = None, None, 0
faces_np_global, width_translation_vector_global = None, None
T1_translate_global, T2_rotate_global = None, None
T_pose_for_tray_display_and_finger_placement_global = None
preview_tray_mesh = None

# --- 可视化辅助函数 ---
def setup_pv_plotter(title, window_size=[800,600], off_screen=False):
    plotter_theme = pv.themes.DocumentTheme()
    plotter_theme.font.family = font_family_viz; plotter_theme.font.color = text_color_viz
    plotter_theme.font.size = 10; plotter_theme.font.label_size = 8
    plotter_theme.background = pv.Color(background_color_viz)
    plotter = pv.Plotter(window_size=window_size, theme=plotter_theme, title=title, off_screen=off_screen)
    if not off_screen:
        plotter.enable_anti_aliasing('msaa', multi_samples=4)
        plotter.enable_parallel_projection()
        plotter.specular_power = 5.0
    plotter.remove_all_lights(); plotter.enable_lightkit()
    if plotter.renderer.cube_axes_actor is not None:
        actor = plotter.renderer.cube_axes_actor; font_family_vtk_int = 1
        if hasattr(actor, 'GetXAxisCaptionActor2D'):
            for i in range(3):
                cap_prop = actor.GetCaptionTextProperty(i) if hasattr(actor, 'GetCaptionTextProperty') else None
                if cap_prop: cap_prop.SetFontFamily(font_family_vtk_int); cap_prop.SetFontSize(8); cap_prop.SetColor(0,0,0); cap_prop.SetBold(0)
                else:
                    title_prop = actor.GetTitleTextProperty(i) if hasattr(actor, 'GetTitleTextProperty') else None
                    if title_prop: title_prop.SetFontFamily(font_family_vtk_int); title_prop.SetFontSize(8); title_prop.SetColor(0,0,0); title_prop.SetBold(0)
        actor.SetXTitle(""); actor.SetYTitle(""); actor.SetZTitle("")
        if hasattr(actor, 'GetProperty'): actor.GetProperty().SetLineWidth(0.5)
    return plotter

def evaluate_gii_for_case_study_with_pressure_iteration(r_value, finger_indices_tuple, normal_sign_multipliers=(1,1,1)):
    global initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global, \
           object_points_global_static, object_centroid_global_static, num_object_points_global_static, \
           faces_np_global, width_translation_vector_global, T1_translate_global, T2_rotate_global, \
           T_pose_for_tray_display_and_finger_placement_global, NUM_FINGER_SLOTS_CASE_STUDY, \
           friction_coefficient, eigenvalue_threshold, collision_threshold, overlap_threshold, \
           INITIAL_PRESSURE_FOR_CONTACT_SEARCH, PRESSURE_STEP_FOR_CONTACT_SEARCH, MAX_PRESSURE_FOR_CONTACT_SEARCH, \
           NODE_COUNT, SPHERE_OBJECT_RADIUS
    default_return = 0.0, np.full(3, -1.0, dtype=float), [None, None, None], []
    if not all(v is not None for v in [initial_coords_ref_global, model_global, object_points_global_static, faces_np_global, width_translation_vector_global, T1_translate_global, T2_rotate_global, T_pose_for_tray_display_and_finger_placement_global]): return default_return
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
                elif current_pressures[i] >= MAX_PRESSURE_FOR_CONTACT_SEARCH: return default_return
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
                    has_cell_normals = ('Normals' in finger_mesh.cell_data and finger_mesh.cell_data['Normals'] is not None and len(finger_mesh.cell_data['Normals']) == finger_mesh.n_cells)
                    try:
                        closest_cell_id, pt_on_mesh_candidate = finger_mesh.find_closest_cell(obj_point, return_closest_point=True)
                        if closest_cell_id < 0 or closest_cell_id >= finger_mesh.n_cells: continue
                        dist = LA.norm(obj_point - pt_on_mesh_candidate)
                        if dist < closest_dist_this_obj_pt_to_any_finger:
                            closest_dist_this_obj_pt_to_any_finger = dist
                            winning_finger_for_this_obj_pt = finger_idx
                            winning_pt_on_mesh_this_obj_pt = pt_on_mesh_candidate
                            if has_cell_normals: winning_normal_this_obj_pt = finger_mesh.cell_normals[closest_cell_id]
                        if dist < collision_threshold: contact_made_by_finger_this_iteration[finger_idx] = True
                    except RuntimeError: continue
                    except Exception: continue
                if winning_finger_for_this_obj_pt != -1:
                    if closest_dist_this_obj_pt_to_any_finger < overlap_threshold: has_overall_overlap_this_iteration = True; break
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
                    pos_dp = any(dp > DOT_PROD_TOLERANCE_LOCAL for dp in finger_dot_products_this_iteration[i]) # Uses global DOT_PROD_TOLERANCE_LOCAL
                    neg_dp = any(dp < -DOT_PROD_TOLERANCE_LOCAL for dp in finger_dot_products_this_iteration[i])
                    if pos_dp and neg_dp: finger_intersects_this_iteration[i] = True
        if any(finger_intersects_this_iteration): return default_return
        for i in range(3):
            if not finger_contact_established[i] and contact_made_by_finger_this_iteration[i]:
                finger_contact_established[i] = True
                final_contact_pressures[i] = current_pressures[i]
                deformed_finger_meshes_at_contact[i] = current_step_finger_meshes[i].copy()
        if all(finger_contact_established): break
        if not pressure_increased_this_step and not all(finger_contact_established): return default_return
    if not all(finger_contact_established): return default_return
    best_contact_info_for_gii_final = [None] * 3
    min_dist_per_finger_final = [float('inf')] * 3
    if object_points_global_static is not None:
        for finger_idx_gii, finger_mesh_gii in enumerate(deformed_finger_meshes_at_contact):
            if finger_mesh_gii is None or finger_mesh_gii.n_cells == 0: continue
            has_cell_normals = ('Normals' in finger_mesh_gii.cell_data and finger_mesh_gii.cell_data['Normals'] is not None and len(finger_mesh_gii.cell_data['Normals']) == finger_mesh_gii.n_cells)
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
    if len(valid_gii_contacts_final_state) < 2: return 0.0, final_contact_pressures, deformed_finger_meshes_at_contact, []
    if object_centroid_global_static is None: return 0.0, final_contact_pressures, deformed_finger_meshes_at_contact, valid_gii_contacts_final_state
    gii = calculate_gii_multi_contact(valid_gii_contacts_final_state, object_centroid_global_static,
                                      friction_coefficient, eigenvalue_threshold,
                                      normal_sign_multipliers,
                                      characteristic_length=SPHERE_OBJECT_RADIUS) # Pass characteristic_length
    final_gii_value = gii if (gii is not None and gii > 1e-9) else 0.0
    return final_gii_value, final_contact_pressures, deformed_finger_meshes_at_contact, valid_gii_contacts_final_state

# --- 主脚本 ---
if __name__ == '__main__':
    print("--- Script Start ---")
    initial_coords_ref_global = load_initial_coordinates(INITIAL_COORDS_PATH, NODE_COUNT)
    model_global,scaler_X_global,scaler_y_global,device_global = load_prediction_components(MODEL_PATH,X_SCALER_PATH,Y_SCALER_PATH,INPUT_DIM,OUTPUT_DIM,HIDDEN_LAYER_1,HIDDEN_LAYER_2,HIDDEN_LAYER_3)
    if initial_coords_ref_global is None or model_global is None: sys.exit("错误：未能初始化手指模型或坐标。")
    faces_np_global = create_faces_array(NODE_COUNT)
    if faces_np_global is None or faces_np_global.size == 0: sys.exit("错误：未能创建手指表面。")
    width_translation_vector_global = np.array([0, finger_width, 0])
    bottom_node_idx = np.argmin(initial_coords_ref_global[:,0]); ref_mid_pt = initial_coords_ref_global[bottom_node_idx] + width_translation_vector_global/2.0
    T1_translate_global = create_transformation_matrix_opt8(None,-ref_mid_pt)
    rot_ref_to_local = np.array([[0,1,0],[0,0,1],[1,0,0]]); T2_rotate_global = create_transformation_matrix_opt8(rot_ref_to_local,None)
    T_pose_for_tray_display_and_finger_placement_global = np.identity(4)
    sphere_world_center = np.array([0, 0, SPHERE_OBJECT_CENTER_OFFSET_Z])
    object_points_global_static = create_sphere_point_cloud(SPHERE_OBJECT_RADIUS, sphere_world_center, n_points=SPHERE_N_POINTS)
    if object_points_global_static is None or object_points_global_static.shape[0] == 0: sys.exit("错误:未能生成球形物体点云")
    object_centroid_global_static = np.mean(object_points_global_static, axis=0)
    num_object_points_global_static = object_points_global_static.shape[0]
    print(f"已生成球形物体，中心: {object_centroid_global_static.round(2)}, 点数: {num_object_points_global_static}")
    preview_tray_mesh = pv.Cylinder(center=(0, 0, -tray_height/2.0), direction=(0, 0, 1), radius=tray_radius, height=tray_height, resolution=30)
    all_finger_combinations = generate_finger_position_combinations_fixed_zero(NUM_FINGER_SLOTS_CASE_STUDY)
    print(f"将为每个r值测试 {len(all_finger_combinations)} 种手指位置组合 (0号位固定)。")
    print(f"测试的r值: {R_VALUES_TO_TEST}")
    os.makedirs(CASE_STUDY_OUTPUT_DIR, exist_ok=True)
    excel_file_path = os.path.join(CASE_STUDY_OUTPUT_DIR, EXCEL_FILENAME)
    column_names = ['r_value'] + [f'slot_{i}_avg_gii' for i in range(NUM_FINGER_SLOTS_CASE_STUDY)]
    all_excel_rows_so_far = []

    if SHOW_ONE_TIME_INITIAL_CONFIG_PREVIEW and R_VALUES_TO_TEST.size > 0 and all_finger_combinations:
        print("\n--- 显示一次性初始构型预览 ---")
        preview_r = R_VALUES_TO_TEST[0]
        preview_combo = all_finger_combinations[0] if all_finger_combinations else (0,1,2) # Default if no combos
        preview_plotter_one_time = setup_pv_plotter(f"Initial Setup Preview: r={preview_r:.2f}, pos={preview_combo}", off_screen=False)
        preview_plotter_one_time.add_mesh(preview_tray_mesh, color=tray_color_viz_pv, opacity=0.3, name="preview_tray_mesh")
        sphere_viz_preview = pv.Sphere(radius=SPHERE_OBJECT_RADIUS, center=sphere_world_center if sphere_world_center is not None else [0,0,SPHERE_OBJECT_CENTER_OFFSET_Z])
        preview_plotter_one_time.add_mesh(sphere_viz_preview, color=object_point_color_viz, style='surface', opacity=0.5)
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
            preview_plotter_one_time.add_mesh(undeformed_finger_world_prev, color=finger_color_viz, style='surface', opacity=0.8, smooth_shading=True, show_edges=True, edge_color='gray', line_width=0.5)
            del undeformed_m_ref, undeformed_finger_world_prev
        preview_plotter_one_time.camera_position = 'iso'; preview_plotter_one_time.camera.zoom(1.5)
        if object_centroid_global_static is not None: preview_plotter_one_time.set_focus(object_centroid_global_static)
        preview_plotter_one_time.add_text(f"Initial Preview: r={preview_r:.2f}, pos={preview_combo}\nShowing for {PREVIEW_DURATION:.1f}s", position="upper_left", font_size=10)
        preview_plotter_one_time.show(title="Initial Configuration Preview")
        preview_plotter_one_time.close(); del preview_plotter_one_time
        print("--- 初始构型预览结束 ---")

    DEBUG_NORMAL_SIGNS = False #  <--- 设置为 True 来运行调试并可视化法向量符号组合
                            #  <--- 设置为 False 以运行标准分析流程
    if DEBUG_NORMAL_SIGNS:
        print("\n--- 调试法向量符号组合 (交互式3D可视化) ---")
        # debug_output_plot_dir = os.path.join(CASE_STUDY_OUTPUT_DIR, DEBUG_PLOT_SUBDIR) # Not saving files in this mode
        # os.makedirs(debug_output_plot_dir, exist_ok=True)
        # print(f"调试图像将保存在(如果启用保存): {os.path.abspath(debug_output_plot_dir)}")

        debug_r_value = R_VALUES_TO_TEST[0] if len(R_VALUES_TO_TEST) > 0 else (SPHERE_OBJECT_RADIUS + finger_width * 0.5)
        debug_finger_combo = all_finger_combinations[0] if len(all_finger_combinations) > 0 else (0, int(NUM_FINGER_SLOTS_CASE_STUDY/3), int(2*NUM_FINGER_SLOTS_CASE_STUDY/3))
        print(f"测试配置: r_value = {debug_r_value:.2f}, finger_combo = {debug_finger_combo}")
        sign_options = [1, -1]
        print("符号组合格式: (手指0乘子, 手指1乘子, 手指2乘子)")
        for s0 in sign_options:
            for s1 in sign_options:
                for s2 in sign_options:
                    current_signs_tuple = (s0, s1, s2)
                    print(f"  尝试法向量符号乘子 = {current_signs_tuple}")
                    eval_results = evaluate_gii_for_case_study_with_pressure_iteration(debug_r_value, debug_finger_combo, normal_sign_multipliers=current_signs_tuple)
                    gii_value_debug, pressures_debug, final_finger_meshes_debug, final_contacts_info_debug = eval_results
                    pressures_str_debug = "N/A"
                    if pressures_debug is not None and hasattr(pressures_debug, 'size') and pressures_debug.size ==3: pressures_str_debug = f"[{pressures_debug[0]:.0f}, {pressures_debug[1]:.0f}, {pressures_debug[2]:.0f}]"
                    if gii_value_debug is not None: print(f"    --> GII = {gii_value_debug:.6f}, 最终接触压力 = {pressures_str_debug}")
                    else: print(f"    --> GII 计算失败或返回None (对于此符号组合)。")
                    if final_finger_meshes_debug and isinstance(final_finger_meshes_debug, list) and all(f is not None for f in final_finger_meshes_debug) and final_contacts_info_debug:
                        plotter_debug = setup_pv_plotter(title=f"r={debug_r_value:.2f}, signs={current_signs_tuple}, GII={gii_value_debug:.4f}", off_screen=False)
                        sphere_viz = pv.Sphere(radius=SPHERE_OBJECT_RADIUS, center=sphere_world_center if sphere_world_center is not None else [0,0,SPHERE_OBJECT_CENTER_OFFSET_Z] )
                        plotter_debug.add_mesh(sphere_viz, color=object_point_color_viz, style='surface', opacity=0.3)
                        for finger_mesh in final_finger_meshes_debug:
                            if finger_mesh: plotter_debug.add_mesh(finger_mesh, color=finger_color_viz, style='surface', opacity=0.7, show_edges=True, edge_color='gray')
                        contact_points_for_glyph = []; effective_normals_for_glyph = []
                        extended_multipliers_debug = list(current_signs_tuple);
                        while len(extended_multipliers_debug) < 3: extended_multipliers_debug.append(1)
                        for contact_detail in final_contacts_info_debug:
                            if contact_detail and len(contact_detail) == 6:
                                contact_pt = np.asarray(contact_detail[4]); original_normal = np.asarray(contact_detail[5]); finger_idx_from_contact = contact_detail[2]
                                multiplier = extended_multipliers_debug[finger_idx_from_contact] if 0 <= finger_idx_from_contact < 3 else 1
                                effective_normal = original_normal * multiplier; norm_of_effective_normal = LA.norm(effective_normal)
                                if norm_of_effective_normal > 1e-9:
                                    contact_points_for_glyph.append(contact_pt)
                                    effective_normals_for_glyph.append(effective_normal / norm_of_effective_normal)
                        if contact_points_for_glyph:
                            contact_points_pv = pv.pyvista_ndarray(contact_points_for_glyph); normals_pv = pv.pyvista_ndarray(effective_normals_for_glyph)
                            glyph_points_pd = pv.PolyData(contact_points_pv); glyph_points_pd['Normals'] = normals_pv
                            arrow_geom = pv.Arrow(tip_length=0.25, tip_radius=0.1, tip_resolution=10, shaft_radius=0.05, shaft_resolution=10)
                            arrow_glyphs = glyph_points_pd.glyph(orient="Normals", scale=False, factor=contact_normal_glyph_scale, geom=arrow_geom)
                            plotter_debug.add_mesh(arrow_glyphs, color=contact_normal_glyph_color)
                            plotter_debug.add_points(contact_points_pv, color=contact_point_glyph_color, point_size=8, render_points_as_spheres=True)
                        plotter_debug.add_text(f"Signs: {current_signs_tuple} GII: {gii_value_debug:.6f}\nPressures: {pressures_str_debug}", position="upper_left", font_size=10)
                        plotter_debug.camera_position = 'yz'; plotter_debug.camera.elevation = 20; plotter_debug.camera.azimuth = -70; plotter_debug.camera.zoom(1.3)
                        print(f"      显示配置: signs={current_signs_tuple}. 请关闭3D窗口以继续下一个...")
                        plotter_debug.show()
                        plotter_debug.close()
                    else: print(f"    由于缺少网格或接触数据，跳过可视化 (符号组合 {current_signs_tuple}).")
        print("--- 调试法向量符号组合结束 ---\n")
        if True: print("法向量调试和可视化完成。脚本将退出。"); sys.exit()

    print("--- 开始主评估循环 (如果未因调试而退出) ---")
    total_configs_to_evaluate = len(R_VALUES_TO_TEST) * len(all_finger_combinations)
    evaluated_configs_count = 0
    start_time_total = time.time()
    for r_val_iter in R_VALUES_TO_TEST:
        print(f"\n===== 测试半径 r = {r_val_iter:.2f} =====")
        gii_lists_for_current_r = {slot_idx: [] for slot_idx in range(NUM_FINGER_SLOTS_CASE_STUDY)}
        for finger_combo_indices_iter in all_finger_combinations:
            evaluated_configs_count += 1
            print(f"  配置 {evaluated_configs_count}/{total_configs_to_evaluate}: r={r_val_iter:.2f}, pos={finger_combo_indices_iter}")
            # For main run, use default normal_sign_multipliers=(1,1,1) (or your determined best fixed signs)
            eval_results_main = evaluate_gii_for_case_study_with_pressure_iteration(r_val_iter, finger_combo_indices_iter) # Default signs
            gii_value, pressures_at_contact, _, _ = eval_results_main # Unpack, use only first two for main run

            contact_pressures_str = "N/A"
            if pressures_at_contact is not None and hasattr(pressures_at_contact, 'size') and pressures_at_contact.size == 3:
                 contact_pressures_str = f"[{pressures_at_contact[0]:.0f}, {pressures_at_contact[1]:.0f}, {pressures_at_contact[2]:.0f}]"
            
            # Logic for storing GII for averaging (from your original Figure_1.txt this was only for gii_value > 0)
            # If you want to include 0s for failures in the average, this logic needs to change.
            # Sticking to the "only positive GIIs for average" implied by Figure_1.txt for now for this part.
            if gii_value > 0:
                print(f"    GII = {gii_value:.4f} (最终接触压力: {contact_pressures_str})")
                for i_finger_in_combo in range(len(finger_combo_indices_iter)):
                    slot_idx_in_combo = finger_combo_indices_iter[i_finger_in_combo]
                    if 0 <= slot_idx_in_combo < NUM_FINGER_SLOTS_CASE_STUDY:
                         gii_lists_for_current_r[slot_idx_in_combo].append(gii_value)
            else:
                print(f"    无效抓取或GII为0 (GII={gii_value:.4f}, 最终尝试压力: {contact_pressures_str})")
                # If you want to include 0s in the average, you would append gii_value (which is 0) here:
                # for i_finger_in_combo in range(len(finger_combo_indices_iter)):
                #    slot_idx_in_combo = finger_combo_indices_iter[i_finger_in_combo]
                #    if 0 <= slot_idx_in_combo < NUM_FINGER_SLOTS_CASE_STUDY:
                #         gii_lists_for_current_r[slot_idx_in_combo].append(gii_value) # Appending 0

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
    print("\nCase Study 脚本结束。")
    try:
        if 'preview_tray_mesh' in globals() and preview_tray_mesh is not None: del preview_tray_mesh; preview_tray_mesh = None
    except Exception: pass
    try: pv.close_all()
    except Exception: pass
    import gc
    gc.collect()