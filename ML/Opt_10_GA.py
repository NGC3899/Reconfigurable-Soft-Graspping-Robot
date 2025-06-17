# -*- coding: utf-8 -*-
# 本脚本使用遗传算法的方式进行构型推荐程序的目标函数求解，使用真实物体点云 + GraspNet预测的位姿
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
import open3d as o3d # 确保 Open3D 已导入
import random # For selecting random distinct positions
import itertools # 导入 itertools 用于生成排列组合

# --- 调试可视化开关 ---
DEBUG_VISUALIZE_FAILED_GRASPS = False
FAILED_GRASP_VIS_COUNT = 0
MAX_FAILED_GRASP_VIS = 5

# --- 打印版本信息 ---
try:
    print(f"Opt_10_GA_Imported_Object - 修正接触检测参考系")
    print(f"Open3D version: {o3d.__version__}")
    print(f"PyVista version: {pv.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Torch version: {torch.__version__}")
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
INPUT_DIM = 1; NODE_COUNT = 63; OUTPUT_DIM = NODE_COUNT * 3
HIDDEN_LAYER_1 = 128; HIDDEN_LAYER_2 = 256; HIDDEN_LAYER_3 = 128

# --- 3. 文件路径定义 ---
MODEL_PATH = 'best_mlp_model.pth'; X_SCALER_PATH = 'x_scaler.joblib'
Y_SCALER_PATH = 'y_scaler.joblib'; INITIAL_COORDS_PATH = 'initial_coordinates.txt'

GRASP_OUTPUTS_BASE_PATH = r"C:\Users\admin\Desktop\grasp_outputs" # <--- 修改为您的路径
RELATIVE_POSE_FILENAME = "relative_gripper_to_object_pose_CONDITIONAL_REORIENT.txt"
HIGH_RES_OBJECT_DESKTOP_PATH = r"C:\Users\admin\Desktop" # <--- 修改为您的路径
HIGH_RES_OBJECT_FILENAME = "Graphene_Bottle.ply" # 示例，您会从外部读取

EXTERNAL_OBJECT_TRANSFORM_PATH = os.path.join(GRASP_OUTPUTS_BASE_PATH, RELATIVE_POSE_FILENAME)
EXTERNAL_OBJECT_PLY_PATH = os.path.join(HIGH_RES_OBJECT_DESKTOP_PATH, HIGH_RES_OBJECT_FILENAME)


# --- 4. 配置参数 ---
tray_radius = 60.0; tray_height = 1.0; finger_width = 10.0
tray_center_original_world = np.array([0.0, 0.0, -tray_height / 2.0]) 
TARGET_POINT_COUNT_FOR_SIM = 2000; OBJECT_SCALE_FACTOR = 0.8 
ALIGN_OBJECT_OBB_TO_X = True 

show_axes = True
finger_color = 'lightcoral'; tray_color_runtime = 'tan'; object_point_color_runtime = 'blue'
object_obb_color_pv = 'green'; object_obb_color_o3d = (0.1, 0.9, 0.1)
finger_color_fig1_init_undeformed = '#ff7f0e'; tray_color_fig1_init = '#BDB7A4'
object_color_fig1_init_points = '#1f77b4'; background_color_fig1_init = '#EAEAEA'
text_color_fig1_init = 'black'; font_family_fig1_init = 'times'
SHOW_INITIAL_SETUP_PREVIEW = True

collision_threshold = 1.0; overlap_threshold = 1e-4
friction_coefficient = 0.5; eigenvalue_threshold = 1e-6
max_pressure = 40000.0
PRESSURE_STEP_EVAL_GRASP = 500.0; INITIAL_PRESSURE_EVAL_GRASP = 100.0
R_BOUNDS = (tray_radius * 0.3, tray_radius * 0.95) 
N_FINGER_SLOTS = 9; DOT_PROD_TOLERANCE_LOCAL = 1e-6
CHARACTERISTIC_LENGTH_FOR_GII = 30.0 

COST_MESH_FAILURE = 9.0; COST_OVERLAP = 5.0; COST_INTERSECTION = 4.0
COST_MAX_PRESSURE_NO_CONTACT = 6.0; COST_NO_CONTACT_OR_ITER_LIMIT = 3.0
COST_LOW_GII_OR_FEW_CONTACTS = 2.0

all_valid_finger_combinations_global = list(itertools.combinations(range(N_FINGER_SLOTS), 3))
num_valid_combinations = len(all_valid_finger_combinations_global)
print(f"总共有 {N_FINGER_SLOTS} 个手指槽位，生成了 {num_valid_combinations} 种唯一的（无序）三手指位置组合。")

GA_POPULATION_SIZE = 5; GA_NUM_GENERATIONS = 4
GA_CX_PROB = 0.7; GA_MUT_PROB = 0.2; GA_ELITISM_COUNT = 2

# --- 5. 辅助函数 (与之前版本基本一致) ---
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
    except FileNotFoundError: print(f"错误：找不到文件 {file_path}"); return None
    except Exception as e: print(f"加载初始坐标时出错: {e}"); return None
    if coords.shape == (expected_nodes, 3): print(f"成功加载 {coords.shape[0]} 个初始节点坐标。"); return coords
    else: print(f"错误：坐标形状 {coords.shape} 与预期 ({expected_nodes},3) 不符。"); return None
def create_faces_array(num_nodes_per_curve):
    faces = []; num_quads = num_nodes_per_curve - 1
    if num_quads <= 0: return np.array([], dtype=int)
    for i in range(num_quads): p1,p2=i,i+1; p3,p4=(i+1)+num_nodes_per_curve,i+num_nodes_per_curve; faces.append([4,p1,p2,p3,p4])
    return np.hstack(faces)
def sort_points_spatially(points):
    if points is None: return None;
    points = np.asarray(points)
    if points.shape[0] < 2: return points
    num_points = points.shape[0]; sorted_indices = []; remaining_indices = list(range(num_points))
    start_node_index = np.argmin(points[:,0]+points[:,1]+points[:,2]); current_index = start_node_index
    sorted_indices.append(current_index)
    if current_index in remaining_indices: remaining_indices.pop(remaining_indices.index(current_index))
    while remaining_indices:
        last_point = points[current_index,np.newaxis]; remaining_points_array = points[remaining_indices]
        if remaining_points_array.ndim == 1: remaining_points_array = remaining_points_array[np.newaxis,:]
        if remaining_points_array.shape[0] == 0: break
        try: distances = cdist(last_point,remaining_points_array)[0]
        except Exception as e_cdist: print(f"Error during cdist: {e_cdist}"); break
        if distances.size == 0: break
        nearest_neighbor_relative_index = np.argmin(distances); nearest_neighbor_absolute_index = remaining_indices[nearest_neighbor_relative_index]
        sorted_indices.append(nearest_neighbor_absolute_index); current_index = nearest_neighbor_absolute_index
        if nearest_neighbor_absolute_index in remaining_indices: remaining_indices.pop(remaining_indices.index(nearest_neighbor_absolute_index))
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
            if np.abs(n[0]) > 0.9: t1 = np.array([0.,1.,0.])
            elif np.abs(n[1]) > 0.9: t1 = np.array([1.,0.,0.])
            else: t1 = np.array([1.,0.,0.])
            norm_t1 = LA.norm(t1)
            if norm_t1 < 1e-9: raise ValueError("Fallback t1 is zero after multiple attempts.")
    t1 /= norm_t1; t2_temp = np.cross(n, t1); norm_t2 = LA.norm(t2_temp)
    if norm_t2 < 1e-9: raise ValueError("Cannot compute tangent 2.")
    t2 = t2_temp / norm_t2; return t1, t2
def load_prediction_components(model_path,x_scaler_path,y_scaler_path,input_dim,output_dim,h1,h2,h3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"ML使用设备: {device}")
    model = MLPRegression(input_dim,output_dim,h1,h2,h3)
    try: model.load_state_dict(torch.load(model_path, map_location=device)); model.to(device); model.eval()
    except Exception as e: print(f"加载模型 {model_path} 时出错: {e}"); return None,None,None,None
    scaler_X,scaler_y=None,None
    try: scaler_X=joblib.load(x_scaler_path); print(f"X Scaler 加载成功。")
    except FileNotFoundError: print(f"警告: 未找到 X scaler '{x_scaler_path}'。")
    except Exception as e: print(f"加载 X Scaler '{x_scaler_path}' 时出错: {e}"); return None,None,None,None
    try: scaler_y=joblib.load(y_scaler_path); print(f"Y Scaler 加载成功。")
    except FileNotFoundError: print(f"警告: 未找到 Y scaler '{y_scaler_path}'。")
    except Exception as e: print(f"加载 Y Scaler '{y_scaler_path}' 时出错: {e}"); return None,None,None,None
    return model,scaler_X,scaler_y,device
def predict_displacements_for_pressure(model,scaler_X,scaler_y,device,pressure_value):
    if model is None: print("模型未加载。"); return None
    pressure_value=np.clip(pressure_value, 0.0, max_pressure)
    input_p=np.array([[pressure_value]],dtype=np.float32)
    if scaler_X:
        try: input_p_scaled = scaler_X.transform(input_p)
        except Exception as e: print(f"X scaler 标准化出错: {e}"); return None
    else: input_p_scaled = input_p
    input_tensor=torch.tensor(input_p_scaled,dtype=torch.float32).to(device); predicted_original_scale=None
    with torch.no_grad():
        try:
            predicted_scaled_tensor=model(input_tensor); predicted_scaled=predicted_scaled_tensor.cpu().numpy()
            if scaler_y:
                try: predicted_original_scale = scaler_y.inverse_transform(predicted_scaled)
                except Exception as e: print(f"Y scaler 反标准化出错: {e}"); return None
            else: predicted_original_scale = predicted_scaled
        except Exception as e: print(f"模型预测出错: {e}"); return None
    if predicted_original_scale is not None:
        if predicted_original_scale.shape[1] != OUTPUT_DIM: print(f"错误：模型输出维度错误"); return None
        return predicted_original_scale.reshape(NODE_COUNT, 3)
    else: return None
def calculate_gii_multi_contact(contacts_info,object_centroid_for_gii,mu,eigenvalue_thresh, characteristic_length=1.0):
    if not contacts_info or len(contacts_info) < 2: return None
    all_wrenches = []; valid_contacts_for_gii = 0
    if characteristic_length <= 1e-9: characteristic_length = 1.0
    for contact in contacts_info:
        if isinstance(contact, (tuple, list)) and len(contact) >= 6:
            pt_on_mesh = contact[4]; normal_finger = contact[5]
        else: continue
        if pt_on_mesh is None or normal_finger is None: continue
        pt_on_mesh = np.asarray(pt_on_mesh); normal_finger = np.asarray(normal_finger)
        if pt_on_mesh.size != 3 or normal_finger.size != 3: continue
        norm_mag = LA.norm(normal_finger)
        if norm_mag < 1e-6: continue
        n_contact_finger_surface_normal = normal_finger / norm_mag
        try: t1, t2 = get_orthogonal_vectors(n_contact_finger_surface_normal)
        except ValueError: continue
        r_contact = pt_on_mesh - object_centroid_for_gii
        force_normal_dir = -n_contact_finger_surface_normal
        d_list = [force_normal_dir + mu * t1, force_normal_dir - mu * t1, force_normal_dir + mu * t2, force_normal_dir - mu * t2]
        for d_force in d_list:
            if LA.norm(d_force) < 1e-9: continue
            torque = np.cross(r_contact, d_force); normalized_torque = torque / characteristic_length
            wrench = np.concatenate((d_force, normalized_torque)); all_wrenches.append(wrench)
        valid_contacts_for_gii += 1
    if valid_contacts_for_gii < 2 or not all_wrenches: return None
    try: grasp_matrix_G = np.column_stack(all_wrenches)
    except ValueError: return None
    if grasp_matrix_G.shape[0] != 6: return None
    J = grasp_matrix_G @ grasp_matrix_G.T
    try:
        eigenvalues = LA.eigvalsh(J); positive_eigenvalues = eigenvalues[eigenvalues > eigenvalue_thresh]
        if not positive_eigenvalues.size: return 0.0
        lambda_min = np.min(positive_eigenvalues); lambda_max = np.max(positive_eigenvalues)
        if lambda_max < eigenvalue_thresh: return 0.0
        return np.sqrt(lambda_min / lambda_max) if lambda_max > 1e-9 else 0.0
    except LA.LinAlgError: return None
def get_rotation_matrix_between_vectors(vec1, vec2):
    a = vec1 / np.linalg.norm(vec1); b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b); c = np.dot(a, b); s = np.linalg.norm(v)
    if s < 1e-9: return np.identity(3) if c > 0 else create_rotation_matrix(np.array([1.0,0,0]) if np.abs(a[0])<0.9 else np.array([0,1.0,0]), np.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r_mat = np.identity(3) + vx + vx @ vx * ((1 - c) / (s ** 2)); return r_mat
def visualize_poses_with_open3d(tray_geometry_transform_mm, tray_axes_transform_mm, object_axes_transform_mm, object_points_world_mm, object_colors_rgb_float, object_obb_world, tray_radius_mm, tray_height_mm, window_title="Open3D Relative Pose Visualization"):
    print(f"\n--- 打开 Open3D 窗口: {window_title} ---")
    geometries = []
    o3d_tray_canonical_centered = o3d.geometry.TriangleMesh.create_cylinder(radius=tray_radius_mm, height=tray_height_mm, resolution=20, split=4)
    o3d_tray_canonical_centered.translate(np.array([0,0,-tray_height_mm/2.0]))
    o3d_tray_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_tray_canonical_centered)
    o3d_tray_wireframe.transform(tray_geometry_transform_mm); o3d_tray_wireframe.paint_uniform_color(pv.Color(tray_color_fig1_init).float_rgb if SHOW_INITIAL_SETUP_PREVIEW else pv.Color(tray_color_runtime).float_rgb)
    geometries.append(o3d_tray_wireframe)
    o3d_object_pcd = o3d.geometry.PointCloud()
    if object_points_world_mm is not None and object_points_world_mm.shape[0] > 0:
        o3d_object_pcd.points = o3d.utility.Vector3dVector(object_points_world_mm)
        if object_colors_rgb_float is not None and len(object_colors_rgb_float) == len(object_points_world_mm): o3d_object_pcd.colors = o3d.utility.Vector3dVector(object_colors_rgb_float)
        else: o3d_object_pcd.paint_uniform_color(pv.Color(object_color_fig1_init_points).float_rgb if SHOW_INITIAL_SETUP_PREVIEW else pv.Color(object_point_color_runtime).float_rgb)
        geometries.append(o3d_object_pcd)
    if object_obb_world is not None: geometries.append(object_obb_world)
    tray_axes_size = tray_radius_mm * 0.7
    o3d_tray_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=tray_axes_size, origin=[0,0,0]); o3d_tray_axes.transform(tray_axes_transform_mm); geometries.append(o3d_tray_axes)
    if object_points_world_mm is not None and object_points_world_mm.shape[0] > 0:
        temp_object_pcd_for_bbox = o3d.geometry.PointCloud(); temp_object_pcd_for_bbox.points = o3d.utility.Vector3dVector(object_points_world_mm)
        bbox_object = temp_object_pcd_for_bbox.get_axis_aligned_bounding_box(); diag_len = LA.norm(bbox_object.get_max_bound() - bbox_object.get_min_bound())
        object_axes_size = diag_len * 0.25 if diag_len > 1e-2 else tray_axes_size * 0.8
    else: object_axes_size = tray_axes_size * 0.8
    object_axes_size = max(object_axes_size, 5.0)
    o3d_obj_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=object_axes_size, origin=[0,0,0]); o3d_obj_axes.transform(object_axes_transform_mm); geometries.append(o3d_obj_axes)
    world_axes_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=tray_radius_mm * 1.2, origin=[0,0,0]); geometries.append(world_axes_o3d)
    o3d.visualization.draw_geometries(geometries, window_name=window_title, width=800, height=600)
    print(f"--- 关闭 Open3D 窗口: {window_title} ---")

# 全局变量定义
initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global = [None]*5
object_points_global_static, object_centroid_global_static, object_mesh_global_static_pv = [None]*3
world_obb_object_global = None; pv_obb_object_mesh_global = None
num_object_points_global_static, faces_np_global, width_translation_vector_global = 0, None, None
T1_translate_global, T2_rotate_global = None, None
# T_disk_ref_in_world_global = None # Replaced by T_pose_for_tray_display_and_finger_placement_global for clarity
T_pose_for_tray_display_and_finger_placement_global = None # Used in evaluate_grasp
tray_pv_mesh_global = None
sampled_object_colors_float_o3d_global = None

def setup_pv_plotter(title, window_size=[900,700], off_screen=False):
    plotter_theme = pv.themes.DocumentTheme(); plotter_theme.font.family = font_family_fig1_init
    plotter_theme.font.color = text_color_fig1_init; plotter_theme.font.size = 10
    plotter_theme.background = pv.Color(background_color_fig1_init)
    plotter = pv.Plotter(window_size=window_size, theme=plotter_theme, title=title, off_screen=off_screen)
    if not off_screen: plotter.enable_anti_aliasing('msaa'); plotter.enable_parallel_projection()
    plotter.remove_all_lights(); plotter.enable_lightkit()
    return plotter

def visualize_failed_ga_step(r_param, chosen_indices_param, pressures_param, finger_meshes_param, failure_reason_str, generation, individual_idx):
    global DEBUG_VISUALIZE_FAILED_GRASPS, FAILED_GRASP_VIS_COUNT, MAX_FAILED_GRASP_VIS, object_mesh_global_static_pv, tray_pv_mesh_global, pv_obb_object_mesh_global, show_axes
    if not DEBUG_VISUALIZE_FAILED_GRASPS or FAILED_GRASP_VIS_COUNT >= MAX_FAILED_GRASP_VIS:
        if FAILED_GRASP_VIS_COUNT == MAX_FAILED_GRASP_VIS and DEBUG_VISUALIZE_FAILED_GRASPS: print("已达到最大失败可视化次数，后续失败将不再显示。"); FAILED_GRASP_VIS_COUNT +=1
        return
    FAILED_GRASP_VIS_COUNT += 1; plotter_title = f"GA失败调试 Gen {generation} Ind {individual_idx} #{FAILED_GRASP_VIS_COUNT}: {failure_reason_str}"
    plotter_debug = setup_pv_plotter(title=plotter_title)
    if tray_pv_mesh_global: plotter_debug.add_mesh(tray_pv_mesh_global, color=tray_color_runtime, opacity=0.5, name='tray_debug') # This tray uses T_actual_tray_geometry_world
    if object_mesh_global_static_pv: # This object uses T_object_target_world_pose
        if 'colors' in object_mesh_global_static_pv.point_data and object_mesh_global_static_pv.point_data['colors'] is not None: plotter_debug.add_mesh(object_mesh_global_static_pv, scalars='colors', rgba=True, style='points', point_size=3, name='obj_debug_imported')
        else: plotter_debug.add_mesh(object_mesh_global_static_pv, color=object_point_color_runtime, style='points', point_size=3, name='obj_debug_imported')
    if pv_obb_object_mesh_global: plotter_debug.add_mesh(pv_obb_object_mesh_global, style='wireframe', color=object_obb_color_pv, line_width=2, name='obj_obb_debug')
    if show_axes: plotter_debug.add_axes(interactive=True) # Uses world axes + actor-specific if enabled
    if finger_meshes_param: # These fingers are transformed using T_pose_for_tray_display_and_finger_placement_global
        for i, mesh in enumerate(finger_meshes_param):
            if mesh: plotter_debug.add_mesh(mesh, color=finger_color, style='surface', opacity=0.75, name=f'finger_debug_{i}')
            else: print(f"  调试可视化警告: 手指 {i} 的网格为 None。")
    params_text = f"r={r_param:.3f}, 位置=({chosen_indices_param[0]},{chosen_indices_param[1]},{chosen_indices_param[2]})\n"; pressures_text = f"压力=[{pressures_param[0]:.0f}, {pressures_param[1]:.0f}, {pressures_param[2]:.0f}]\n"; reason_text = f"原因: {failure_reason_str}"
    plotter_debug.add_text(params_text + pressures_text + reason_text, position="upper_left", font_size=10, color=text_color_fig1_init)
    plotter_debug.camera_position = 'xy'; print(f"\n(PyVista) 显示GA失败步骤 Gen {generation} Ind {individual_idx} #{FAILED_GRASP_VIS_COUNT}。原因: {failure_reason_str}。请关闭窗口以继续...")
    plotter_debug.show(cpos='xy'); plotter_debug.close()

# evaluate_grasp function (from user, adapted for GA context)
def evaluate_grasp(r, pos_idx1, pos_idx2, pos_idx3, generation_info="N/A"):
    global initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global, \
           object_points_global_static, object_centroid_global_static, num_object_points_global_static, \
           faces_np_global, width_translation_vector_global, T1_translate_global, T2_rotate_global, \
           T_pose_for_tray_display_and_finger_placement_global, N_FINGER_SLOTS, \
           CHARACTERISTIC_LENGTH_FOR_GII, collision_threshold, overlap_threshold, \
           INITIAL_PRESSURE_EVAL_GRASP, PRESSURE_STEP_EVAL_GRASP, max_pressure, \
           friction_coefficient, eigenvalue_threshold, \
           COST_MESH_FAILURE, COST_OVERLAP, COST_INTERSECTION, \
           COST_MAX_PRESSURE_NO_CONTACT, COST_NO_CONTACT_OR_ITER_LIMIT, COST_LOW_GII_OR_FEW_CONTACTS, \
           DOT_PROD_TOLERANCE_LOCAL, DEBUG_VISUALIZE_FAILED_GRASPS

    if T_pose_for_tray_display_and_finger_placement_global is None:
        print(f"错误: T_pose_for_tray_display_and_finger_placement_global 未在 evaluate_grasp 中初始化! (Gen: {generation_info})")
        return 20.0, np.full(3, INITIAL_PRESSURE_EVAL_GRASP, dtype=float), [None]*3, None

    chosen_indices = [int(pos_idx1), int(pos_idx2), int(pos_idx3)]
    current_call_params_str = f"r={r:.3f}, 位置=({chosen_indices[0]},{chosen_indices[1]},{chosen_indices[2]})"

    return_pressures = np.full(3, INITIAL_PRESSURE_EVAL_GRASP, dtype=float)
    return_meshes = [None] * 3
    return_contacts = None
    
    cost_to_return = 20.0 
    gii_value_to_print = "N/A" 
    failure_reason_for_vis = ""
    deformed_finger_meshes_at_contact = [None] * 3 

    try:
        current_pressures = np.full(3, INITIAL_PRESSURE_EVAL_GRASP, dtype=float)
        finger_contact_established = [False] * 3
        
        max_pressure_iterations = int((max_pressure - INITIAL_PRESSURE_EVAL_GRASP) / PRESSURE_STEP_EVAL_GRASP) * 3 + 25 
        current_pressure_iter = 0

        while True:
            current_pressure_iter += 1
            if current_pressure_iter > max_pressure_iterations:
                cost_to_return = COST_NO_CONTACT_OR_ITER_LIMIT
                failure_reason_for_vis = "迭代超限"
                print(f"  GA评估 ({generation_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                if DEBUG_VISUALIZE_FAILED_GRASPS:
                    gen_num_viz, ind_idx_viz = "N/A", "N/A"
                    if isinstance(generation_info, str) and "-" in generation_info:
                        parts = generation_info.split("-")
                        if len(parts) == 2: gen_num_viz, ind_idx_viz = parts[0], parts[1]
                    visualize_failed_ga_step(r, chosen_indices, current_pressures, deformed_finger_meshes_at_contact, failure_reason_for_vis, gen_num_viz, ind_idx_viz)
                return (cost_to_return, current_pressures, deformed_finger_meshes_at_contact, None)

            pressure_changed_this_step = False
            for i in range(3):
                if not finger_contact_established[i] and current_pressures[i] < max_pressure:
                    current_pressures[i] += PRESSURE_STEP_EVAL_GRASP
                    current_pressures[i] = min(current_pressures[i], max_pressure)
                    pressure_changed_this_step = True
            
            current_step_finger_meshes = [None] * 3 
            mesh_generation_successful_this_iter = True
            for i in range(3):
                displacements = predict_displacements_for_pressure(model_global, scaler_X_global, scaler_y_global, device_global, current_pressures[i])
                if displacements is None: mesh_generation_successful_this_iter = False; break 
                deformed_curve1_ref_unordered = initial_coords_ref_global + displacements
                curve2_ref = initial_coords_ref_global + width_translation_vector_global
                deformed_curve2_ref_unordered = curve2_ref + displacements
                sorted_deformed_curve1_ref = sort_points_spatially(deformed_curve1_ref_unordered)
                sorted_deformed_curve2_ref = sort_points_spatially(deformed_curve2_ref_unordered)
                if sorted_deformed_curve1_ref is None or sorted_deformed_curve2_ref is None: mesh_generation_successful_this_iter = False; break 
                sorted_deformed_vertices_ref = np.vstack((sorted_deformed_curve1_ref, sorted_deformed_curve2_ref))
                if faces_np_global is None or faces_np_global.size == 0: mesh_generation_successful_this_iter = False; break 
                try: deformed_mesh_ref = pv.PolyData(sorted_deformed_vertices_ref, faces=faces_np_global)
                except Exception: mesh_generation_successful_this_iter = False; break 
                
                current_finger_actual_pos_index = chosen_indices[i]
                current_angle_deg = current_finger_actual_pos_index * (360.0 / N_FINGER_SLOTS)
                angle_rad_bo = np.radians(current_angle_deg)
                rot_angle_z_placing = angle_rad_bo + np.pi / 2.0
                rot_z_placing = create_rotation_matrix_z(rot_angle_z_placing)
                target_pos_on_circle = np.array([ r * np.cos(angle_rad_bo), r * np.sin(angle_rad_bo), 0.0 ])
                T3_place = create_transformation_matrix_opt8(rot_z_placing, target_pos_on_circle)
                T_transform_finger_relative_to_tray_origin = T3_place @ T2_rotate_global @ T1_translate_global
                # T_pose_for_tray_display_and_finger_placement_global is used here
                if T_pose_for_tray_display_and_finger_placement_global is None: mesh_generation_successful_this_iter = False; break 
                T_final_finger_world = T_pose_for_tray_display_and_finger_placement_global @ T_transform_finger_relative_to_tray_origin
                try:
                    final_transformed_finger_mesh = deformed_mesh_ref.transform(T_final_finger_world, inplace=False)
                    if final_transformed_finger_mesh is None or final_transformed_finger_mesh.n_points == 0: mesh_generation_successful_this_iter = False; break 
                    final_transformed_finger_mesh.clean(inplace=True) 
                    if final_transformed_finger_mesh.n_cells > 0: 
                        final_transformed_finger_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True, auto_orient_normals=True, non_manifold_traversal=False, split_vertices=False) 
                    current_step_finger_meshes[i] = final_transformed_finger_mesh
                    deformed_finger_meshes_at_contact[i] = final_transformed_finger_mesh.copy() 
                except Exception: mesh_generation_successful_this_iter = False; break 
            
            if not mesh_generation_successful_this_iter:
                cost_to_return = COST_MESH_FAILURE
                failure_reason_for_vis = "网格生成失败"
                print(f"  GA评估 ({generation_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                if DEBUG_VISUALIZE_FAILED_GRASPS:
                    gen_num_viz, ind_idx_viz = "N/A", "N/A"
                    if isinstance(generation_info, str) and "-" in generation_info:
                        parts = generation_info.split("-")
                        if len(parts) == 2: gen_num_viz, ind_idx_viz = parts[0], parts[1]
                    visualize_failed_ga_step(r, chosen_indices, current_pressures, deformed_finger_meshes_at_contact, failure_reason_for_vis, gen_num_viz, ind_idx_viz)
                return (cost_to_return, current_pressures, deformed_finger_meshes_at_contact, None)

            for i_mesh_update in range(3):
                if current_step_finger_meshes[i_mesh_update] is not None:
                    deformed_finger_meshes_at_contact[i_mesh_update] = current_step_finger_meshes[i_mesh_update].copy()

            has_overlap_this_iter = False
            contact_made_by_finger_this_iter_flags = [False] * 3
            finger_dot_products_this_iter = [[] for _ in range(3)] 
            
            if object_points_global_static is not None and num_object_points_global_static > 0:
                for obj_pt_idx, obj_point in enumerate(object_points_global_static): # obj_point is in world coords
                    closest_dist_for_obj_pt = float('inf'); winning_finger_for_obj_pt = -1
                    winning_normal_for_obj_pt = None; winning_pt_on_mesh_for_obj_pt = None
                    for finger_idx, finger_mesh in enumerate(current_step_finger_meshes): # finger_mesh is in world coords
                        if finger_mesh is None or finger_mesh.n_cells == 0 : continue
                        has_normals = 'Normals' in finger_mesh.cell_data and finger_mesh.cell_data['Normals'] is not None and len(finger_mesh.cell_data['Normals']) == finger_mesh.n_cells
                        try:
                            closest_cell_id, pt_on_mesh_candidate = finger_mesh.find_closest_cell(obj_point, return_closest_point=True)
                            if closest_cell_id < 0 or closest_cell_id >= finger_mesh.n_cells: continue 
                            dist = LA.norm(obj_point - pt_on_mesh_candidate)
                            if dist < closest_dist_for_obj_pt:
                                closest_dist_for_obj_pt = dist; winning_finger_for_obj_pt = finger_idx
                                winning_pt_on_mesh_for_obj_pt = pt_on_mesh_candidate
                                if has_normals: winning_normal_for_obj_pt = finger_mesh.cell_normals[closest_cell_id] 
                                else: winning_normal_for_obj_pt = None 
                            if dist < collision_threshold: contact_made_by_finger_this_iter_flags[finger_idx] = True
                        except RuntimeError: continue 
                        except Exception: continue 
                    if winning_finger_for_obj_pt != -1: 
                        if closest_dist_for_obj_pt < overlap_threshold: has_overlap_this_iter = True; break 
                        elif closest_dist_for_obj_pt < collision_threshold: 
                            if winning_normal_for_obj_pt is not None and LA.norm(winning_normal_for_obj_pt) > 1e-9:
                                vector_to_point = obj_point - winning_pt_on_mesh_for_obj_pt
                                if LA.norm(vector_to_point) > 1e-9: 
                                    dot_prod = np.dot(vector_to_point / LA.norm(vector_to_point), winning_normal_for_obj_pt / LA.norm(winning_normal_for_obj_pt))
                                    finger_dot_products_this_iter[winning_finger_for_obj_pt].append(dot_prod)
                if has_overlap_this_iter:
                    cost_to_return = COST_OVERLAP
                    failure_reason_for_vis = "重叠"
                    print(f"  GA评估 ({generation_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                    if DEBUG_VISUALIZE_FAILED_GRASPS:
                        gen_num_viz, ind_idx_viz = "N/A", "N/A"
                        if isinstance(generation_info, str) and "-" in generation_info:
                            parts = generation_info.split("-")
                            if len(parts) == 2: gen_num_viz, ind_idx_viz = parts[0], parts[1]
                        visualize_failed_ga_step(r, chosen_indices, current_pressures, current_step_finger_meshes, failure_reason_for_vis, gen_num_viz, ind_idx_viz)
                    return (cost_to_return, current_pressures, current_step_finger_meshes, None)

            finger_intersects_this_iter = [False] * 3 
            if not has_overlap_this_iter: 
                for i in range(3):
                    if contact_made_by_finger_this_iter_flags[i] and finger_dot_products_this_iter[i]: 
                        has_pos_dp = any(dp > DOT_PROD_TOLERANCE_LOCAL for dp in finger_dot_products_this_iter[i])
                        has_neg_dp = any(dp < -DOT_PROD_TOLERANCE_LOCAL for dp in finger_dot_products_this_iter[i])
                        if has_pos_dp and has_neg_dp: finger_intersects_this_iter[i] = True 
            if any(finger_intersects_this_iter):
                cost_to_return = COST_INTERSECTION
                failure_reason_for_vis = "穿透"
                print(f"  GA评估 ({generation_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                if DEBUG_VISUALIZE_FAILED_GRASPS:
                    gen_num_viz, ind_idx_viz = "N/A", "N/A"
                    if isinstance(generation_info, str) and "-" in generation_info:
                        parts = generation_info.split("-")
                        if len(parts) == 2: gen_num_viz, ind_idx_viz = parts[0], parts[1]
                    visualize_failed_ga_step(r, chosen_indices, current_pressures, current_step_finger_meshes, failure_reason_for_vis, gen_num_viz, ind_idx_viz)
                return (cost_to_return, current_pressures, current_step_finger_meshes, None)
            
            all_fingers_established_this_iter_round = True
            for i in range(3):
                if contact_made_by_finger_this_iter_flags[i]: 
                    if not finger_contact_established[i]: 
                        finger_contact_established[i] = True
                if not finger_contact_established[i]: 
                    all_fingers_established_this_iter_round = False

            if all_fingers_established_this_iter_round:
                return_pressures = current_pressures.copy()
                return_meshes = [m.copy() if m else None for m in deformed_finger_meshes_at_contact] 
                
                final_gii_contacts_collected = []
                min_dist_for_final_gii = [float('inf')] * 3 
                if object_points_global_static is not None and num_object_points_global_static > 0:
                    for f_idx, f_mesh in enumerate(deformed_finger_meshes_at_contact): 
                        if f_mesh is None or f_mesh.n_cells == 0: continue
                        has_f_normals = 'Normals' in f_mesh.cell_data and f_mesh.cell_data['Normals'] is not None and len(f_mesh.cell_data['Normals']) == f_mesh.n_cells
                        if not has_f_normals: continue 
                        
                        best_contact_for_this_finger = None
                        for obj_pt_idx_f, obj_pt_f in enumerate(object_points_global_static):
                            try:
                                c_cell_id, pt_on_m_cand = f_mesh.find_closest_cell(obj_pt_f, return_closest_point=True)
                                if c_cell_id < 0 or c_cell_id >= f_mesh.n_cells: continue
                                d_f = LA.norm(obj_pt_f - pt_on_m_cand)
                                if d_f < collision_threshold and d_f < min_dist_for_final_gii[f_idx]: 
                                    norm_cand = f_mesh.cell_normals[c_cell_id] 
                                    if norm_cand is not None and LA.norm(norm_cand) > 1e-9:
                                        min_dist_for_final_gii[f_idx] = d_f 
                                        best_contact_for_this_finger = (d_f, obj_pt_idx_f, f_idx, c_cell_id, pt_on_m_cand, norm_cand)
                            except RuntimeError: continue 
                            except Exception: continue
                        if best_contact_for_this_finger: final_gii_contacts_collected.append(best_contact_for_this_finger)
                
                return_contacts = final_gii_contacts_collected
                if len(final_gii_contacts_collected) >= 2: 
                    gii = calculate_gii_multi_contact(final_gii_contacts_collected, object_centroid_global_static, friction_coefficient, eigenvalue_threshold, characteristic_length=CHARACTERISTIC_LENGTH_FOR_GII)
                    if gii is not None and gii > 1e-9: 
                        cost_to_return = -gii
                        gii_value_to_print = gii
                        print(f"  GA评估 ({generation_info}): {current_call_params_str} -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print:.4f}")
                        return (cost_to_return, return_pressures, return_meshes, return_contacts)
                    else: 
                        cost_to_return = COST_LOW_GII_OR_FEW_CONTACTS
                        gii_value_to_print = gii if gii is not None else "计算失败"
                        failure_reason_for_vis = f"低GII ({gii_value_to_print if isinstance(gii_value_to_print, str) else gii_value_to_print:.4f})"
                        print(f"  GA评估 ({generation_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print if isinstance(gii_value_to_print, str) else gii_value_to_print:.4f}")
                        if DEBUG_VISUALIZE_FAILED_GRASPS:
                            gen_num_viz, ind_idx_viz = "N/A", "N/A"
                            if isinstance(generation_info, str) and "-" in generation_info:
                                parts = generation_info.split("-")
                                if len(parts) == 2: gen_num_viz, ind_idx_viz = parts[0], parts[1]
                            visualize_failed_ga_step(r, chosen_indices, return_pressures, return_meshes, failure_reason_for_vis, gen_num_viz, ind_idx_viz)
                        return (cost_to_return, return_pressures, return_meshes, return_contacts) 
                else: 
                    cost_to_return = COST_LOW_GII_OR_FEW_CONTACTS
                    failure_reason_for_vis = "接触点不足 GII"
                    print(f"  GA评估 ({generation_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                    if DEBUG_VISUALIZE_FAILED_GRASPS:
                        gen_num_viz, ind_idx_viz = "N/A", "N/A"
                        if isinstance(generation_info, str) and "-" in generation_info:
                            parts = generation_info.split("-")
                            if len(parts) == 2: gen_num_viz, ind_idx_viz = parts[0], parts[1]
                        visualize_failed_ga_step(r, chosen_indices, return_pressures, return_meshes, failure_reason_for_vis, gen_num_viz, ind_idx_viz)
                    return (cost_to_return, return_pressures, return_meshes, return_contacts) 

            for i in range(3):
                if not finger_contact_established[i] and current_pressures[i] >= max_pressure:
                    cost_to_return = COST_MAX_PRESSURE_NO_CONTACT
                    failure_reason_for_vis = f"手指 {i} 最大压力无接触"
                    print(f"  GA评估 ({generation_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                    if DEBUG_VISUALIZE_FAILED_GRASPS:
                        gen_num_viz, ind_idx_viz = "N/A", "N/A"
                        if isinstance(generation_info, str) and "-" in generation_info:
                            parts = generation_info.split("-")
                            if len(parts) == 2: gen_num_viz, ind_idx_viz = parts[0], parts[1]
                        visualize_failed_ga_step(r, chosen_indices, current_pressures, deformed_finger_meshes_at_contact, failure_reason_for_vis, gen_num_viz, ind_idx_viz)
                    return (cost_to_return, current_pressures, deformed_finger_meshes_at_contact, None) 
            
            if not pressure_changed_this_step and not all(finger_contact_established):
                cost_to_return = COST_MAX_PRESSURE_NO_CONTACT 
                failure_reason_for_vis = "停滞无接触"
                print(f"  GA评估 ({generation_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                if DEBUG_VISUALIZE_FAILED_GRASPS:
                    gen_num_viz, ind_idx_viz = "N/A", "N/A"
                    if isinstance(generation_info, str) and "-" in generation_info:
                        parts = generation_info.split("-")
                        if len(parts) == 2: gen_num_viz, ind_idx_viz = parts[0], parts[1]
                    visualize_failed_ga_step(r, chosen_indices, current_pressures, deformed_finger_meshes_at_contact, failure_reason_for_vis, gen_num_viz, ind_idx_viz)
                return (cost_to_return, current_pressures, deformed_finger_meshes_at_contact, None) 
        
    except Exception as e_eval_grasp:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"DEBUG_GA: EXCEPTION in evaluate_grasp for {current_call_params_str} (GenInfo: {generation_info}): {e_eval_grasp}")
        traceback.print_exc()
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        cost_to_return = 20.0 
        failure_reason_for_vis = "evaluate_grasp内部异常"
        current_pressures_fallback = current_pressures if 'current_pressures' in locals() else np.full(3, INITIAL_PRESSURE_EVAL_GRASP, dtype=float)
        deformed_meshes_fallback = deformed_finger_meshes_at_contact 
        
        print(f"  GA评估 ({generation_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
        if DEBUG_VISUALIZE_FAILED_GRASPS:
            gen_num_viz, ind_idx_viz = "N/A", "N/A"
            if isinstance(generation_info, str) and "-" in generation_info:
                parts = generation_info.split("-")
                if len(parts) == 2: gen_num_viz, ind_idx_viz = parts[0], parts[1]
            visualize_failed_ga_step(r, chosen_indices, current_pressures_fallback, 
                                     deformed_meshes_fallback, 
                                     failure_reason_for_vis, gen_num_viz, ind_idx_viz)
        return (cost_to_return, None, None, None)


# --- 主脚本 ---
if __name__ == '__main__':
    # --- A. 参照 Opt_10_BO.txt 的位姿设定逻辑 ---
    T_object_in_ref_gripper_meters = load_transformation_matrix_from_txt(EXTERNAL_OBJECT_TRANSFORM_PATH)
    if T_object_in_ref_gripper_meters is None: sys.exit(f"错误：未能从 {EXTERNAL_OBJECT_TRANSFORM_PATH} 加载物体相对于参考夹爪的位姿。")
    print(f"\n已加载“物体在参考夹爪中的位姿” (T_object_in_ref_gripper_meters):\n{T_object_in_ref_gripper_meters.round(3)}")
    T_original_tray_pose_world = create_transformation_matrix_opt8(np.identity(3), tray_center_original_world)
    print(f"定义的“原始托盘”世界位姿 (T_original_tray_pose_world):\n{T_original_tray_pose_world.round(3)}")
    T_ref_gripper_in_object_meters = np.linalg.inv(T_object_in_ref_gripper_meters)
    T_object_relative_to_original_tray_mm = copy.deepcopy(T_ref_gripper_in_object_meters)
    T_object_relative_to_original_tray_mm[:3, 3] *= 1000.0
    print(f"计算得到的“物体相对于原始托盘的位姿” (T_object_relative_to_original_tray_mm):\n{T_object_relative_to_original_tray_mm.round(3)}")
    T_original_object_pose_world = T_original_tray_pose_world @ T_object_relative_to_original_tray_mm
    print(f"计算得到的“物体在其原始场景中的世界位姿” (T_original_object_pose_world):\n{T_original_object_pose_world.round(3)}")

    # --- B. 应用 Opt_10_BO.txt 的“位姿对调”逻辑 ---
    # 物体点云的目标世界位姿 (即物体出现在托盘的原始位置)
    T_object_target_world_pose = T_original_tray_pose_world 
    
    # 托盘/手指放置的参考系原点 (在物体原始GraspNet场景中的位置)
    _T_tray_ref_before_fix = T_original_object_pose_world 
    angle_rad_fix = np.pi / 2.0; cos_fix = np.cos(angle_rad_fix); sin_fix = np.sin(angle_rad_fix)
    T_local_orientation_fix_for_tray_geometry = np.array([[cos_fix,0,sin_fix,0],[0,1,0,0],[-sin_fix,0,cos_fix,0],[0,0,0,1]])
    # 实际显示的圆盘几何体的世界位姿 (也是手指放置的参考系)
    T_actual_tray_geometry_world = _T_tray_ref_before_fix @ T_local_orientation_fix_for_tray_geometry 
    
    # 将 T_actual_tray_geometry_world 赋值给 evaluate_grasp 和初始可视化使用的手指放置参考变量
    T_pose_for_tray_display_and_finger_placement_global = T_actual_tray_geometry_world
    
    # 用于可视化坐标轴的变换
    T_tray_axes_vis_world = T_actual_tray_geometry_world # 托盘/手指放置参考系的坐标轴
    T_object_axes_vis_world = T_object_target_world_pose # 物体的坐标轴与其目标世界位姿一致

    print(f"\n应用位姿对调与修正逻辑后：")
    print(f"  物体点云的目标世界位姿 (T_object_target_world_pose):\n{T_object_target_world_pose.round(3)}")
    print(f"  实际圆盘几何体/手指放置参考系的世界位姿 (T_actual_tray_geometry_world):\n{T_actual_tray_geometry_world.round(3)}")
    print(f"  手指放置的全局变换 (T_pose_for_tray_display_and_finger_placement_global) 已设为 T_actual_tray_geometry_world.")
    print(f"  托盘/手指参考坐标轴可视化位姿 (T_tray_axes_vis_world):\n{T_tray_axes_vis_world.round(3)}")
    print(f"  物体坐标轴可视化位姿 (T_object_axes_vis_world):\n{T_object_axes_vis_world.round(3)}")
    
    # --- C. 加载和处理高精度物体点云 ---
    print(f"\n加载高精度物体点云: {EXTERNAL_OBJECT_PLY_PATH}")
    if not os.path.exists(EXTERNAL_OBJECT_PLY_PATH): sys.exit(f"错误: 物体点云文件 '{EXTERNAL_OBJECT_PLY_PATH}' 未找到。")
    try:
        external_object_o3d = o3d.io.read_point_cloud(EXTERNAL_OBJECT_PLY_PATH)
        if not external_object_o3d.has_points(): sys.exit(f"错误: {EXTERNAL_OBJECT_PLY_PATH} 为空或不是有效的点云文件。")
    except Exception as e: sys.exit(f"加载 {EXTERNAL_OBJECT_PLY_PATH} 时出错: {e}")
    object_points_raw_mm = np.asarray(external_object_o3d.points)
    object_colors_raw_float = np.asarray(external_object_o3d.colors) if external_object_o3d.has_colors() else None
    print(f"加载的 '{os.path.basename(EXTERNAL_OBJECT_PLY_PATH)}' 含 {len(object_points_raw_mm)} 点。")

    if OBJECT_SCALE_FACTOR != 1.0: print(f"  应用缩放因子 {OBJECT_SCALE_FACTOR}。"); object_points_raw_mm *= OBJECT_SCALE_FACTOR
    centroid_raw_mm = np.mean(object_points_raw_mm, axis=0); points_centered_raw_mm = object_points_raw_mm - centroid_raw_mm
    print(f"  点云原始质心 (缩放后): {centroid_raw_mm.round(3)}。已居中。")
    T_local_alignment_for_obb = np.identity(4); points_centered_aligned_mm = points_centered_raw_mm
    if ALIGN_OBJECT_OBB_TO_X and points_centered_raw_mm.shape[0] > 0:
        print("  开始局部OBB对齐..."); local_pcd_for_alignment = o3d.geometry.PointCloud(); local_pcd_for_alignment.points = o3d.utility.Vector3dVector(points_centered_raw_mm)
        try:
            local_obb = local_pcd_for_alignment.get_oriented_bounding_box(); longest_extent_idx = np.argmax(local_obb.extent)
            local_longest_axis_vec = local_obb.R[:, longest_extent_idx]; target_local_x_axis = np.array([1.0, 0.0, 0.0])
            print(f"    局部OBB最长轴: {local_longest_axis_vec.round(3)}")
            R_local_align = get_rotation_matrix_between_vectors(local_longest_axis_vec, target_local_x_axis)
            if R_local_align is not None:
                T_local_alignment_for_obb = create_transformation_matrix_opt8(R_local_align, None)
                points_centered_aligned_mm = transform_points_opt8(points_centered_raw_mm, T_local_alignment_for_obb)
                print(f"    已局部对齐。局部对齐变换:\n{T_local_alignment_for_obb.round(3)}")
            else: print("    局部对齐旋转计算失败或无需旋转。")
        except Exception as e_obb_align: print(f"    局部OBB对齐出错: {e_obb_align}。")
    elif not ALIGN_OBJECT_OBB_TO_X: print("  跳过局部OBB对齐。")
    
    # 物体点云变换到其最终目标世界位姿
    object_points_transformed_full_mm = transform_points_opt8(points_centered_aligned_mm, T_object_target_world_pose)
    
    print(f"\n对定位后的点云进行抽稀至约 {TARGET_POINT_COUNT_FOR_SIM} 点...")
    final_sampled_points_mm = None; sampled_colors_uint8_pv = None
    if len(object_points_transformed_full_mm) > TARGET_POINT_COUNT_FOR_SIM:
        temp_pcd_for_sampling = o3d.geometry.PointCloud(); temp_pcd_for_sampling.points = o3d.utility.Vector3dVector(object_points_transformed_full_mm)
        if object_colors_raw_float is not None and len(object_colors_raw_float) == len(points_centered_aligned_mm): 
             if len(object_colors_raw_float) == len(object_points_transformed_full_mm): temp_pcd_for_sampling.colors = o3d.utility.Vector3dVector(object_colors_raw_float)
             else: print(f"  警告: 原始颜色数量与变换后点数量不匹配，采样时不使用颜色。")
        if temp_pcd_for_sampling.has_points():
            num_pts_sample_from = len(temp_pcd_for_sampling.points)
            if num_pts_sample_from > 0:
                sampling_ratio = TARGET_POINT_COUNT_FOR_SIM / num_pts_sample_from
                if 0 < sampling_ratio < 1.0:
                    sampled_pcd_o3d = temp_pcd_for_sampling.random_down_sample(sampling_ratio)
                    final_sampled_points_mm = np.asarray(sampled_pcd_o3d.points)
                    if sampled_pcd_o3d.has_colors(): sampled_object_colors_float_o3d_global = np.asarray(sampled_pcd_o3d.colors)
                else: final_sampled_points_mm = np.asarray(temp_pcd_for_sampling.points); sampled_object_colors_float_o3d_global = np.asarray(temp_pcd_for_sampling.colors) if temp_pcd_for_sampling.has_colors() else None
            else: final_sampled_points_mm = object_points_transformed_full_mm
        else: final_sampled_points_mm = object_points_transformed_full_mm
        print(f"  采样后点云数量: {len(final_sampled_points_mm if final_sampled_points_mm is not None else [])}")
    elif len(object_points_transformed_full_mm) > 0 :
        final_sampled_points_mm = object_points_transformed_full_mm
        if object_colors_raw_float is not None and len(object_colors_raw_float) == len(final_sampled_points_mm): sampled_object_colors_float_o3d_global = object_colors_raw_float
        print(f"  点云点数: {len(final_sampled_points_mm)} (无需抽稀).")
    else: sys.exit(f"错误: 变换后的高精度点云不含点.")

    if sampled_object_colors_float_o3d_global is not None:
        if sampled_object_colors_float_o3d_global.ndim == 2 and sampled_object_colors_float_o3d_global.shape[1] == 3: sampled_colors_uint8_pv = (sampled_object_colors_float_o3d_global * 255).astype(np.uint8)
        else: print(f"警告: 采样点云颜色维度非RGB。"); sampled_colors_uint8_pv = None; sampled_object_colors_float_o3d_global = None

    object_points_global_static = final_sampled_points_mm # These are object points in world coordinates
    object_mesh_global_static_pv = pv.PolyData(object_points_global_static) if object_points_global_static is not None else None
    if sampled_colors_uint8_pv is not None and object_mesh_global_static_pv is not None and len(sampled_colors_uint8_pv) == object_mesh_global_static_pv.n_points:
        object_mesh_global_static_pv.point_data['colors'] = sampled_colors_uint8_pv; print("物体颜色已赋给PyVista对象。")
    else: print(f"物体无颜色或不匹配，PyVista用默认色。")
    
    if object_points_global_static is not None and object_points_global_static.shape[0] > 0:
        object_centroid_global_static = np.mean(object_points_global_static, axis=0)
        num_object_points_global_static = object_points_global_static.shape[0]
        print(f"\n已处理 '{os.path.basename(EXTERNAL_OBJECT_PLY_PATH)}'。仿真点数: {num_object_points_global_static}，最终世界质心: {object_centroid_global_static.round(3)}")
    else: object_centroid_global_static = np.array([0,0,0]); num_object_points_global_static = 0; print(f"\n警告: 处理后的物体点云为空或无效。")

    if num_object_points_global_static > 0:
        o3d_temp_pcd_for_obb = o3d.geometry.PointCloud(); o3d_temp_pcd_for_obb.points = o3d.utility.Vector3dVector(object_points_global_static)
        world_obb_object_global = o3d_temp_pcd_for_obb.get_oriented_bounding_box(); world_obb_object_global.color = object_obb_color_o3d
        print(f"  物体OBB (世界坐标系): 中心={world_obb_object_global.center.round(3)}, 范围={world_obb_object_global.extent.round(3)}")
        pv_obb_object_mesh_global = pv.Cube(center=world_obb_object_global.center, x_length=world_obb_object_global.extent[0], y_length=world_obb_object_global.extent[1], z_length=world_obb_object_global.extent[2])
        pv_obb_object_mesh_global.orientation = world_obb_object_global.R.flatten()
        if world_obb_object_global.extent is not None and all(e > 1e-6 for e in world_obb_object_global.extent): CHARACTERISTIC_LENGTH_FOR_GII = np.mean(world_obb_object_global.extent); print(f"  更新GII特征长度: {CHARACTERISTIC_LENGTH_FOR_GII:.3f}")
        else: print(f"  警告: OBB范围无效，GII特征长度保持为 {CHARACTERISTIC_LENGTH_FOR_GII:.3f}")
    else: world_obb_object_global = None; pv_obb_object_mesh_global = None; print("  物体点云为空，无法计算OBB。")
    
    # --- D. 初始状态可视化 ---
    if SHOW_INITIAL_SETUP_PREVIEW:
        print("\n--- 开始初始状态可视化 (参照Opt_10_BO风格) ---")
        if object_points_global_static is not None and object_points_global_static.shape[0] > 0:
            preview_r_init = (R_BOUNDS[0] + R_BOUNDS[1]) / 2.0
            preview_combo_init = all_valid_finger_combinations_global[0] if all_valid_finger_combinations_global else (0, int(N_FINGER_SLOTS/3), int(2*N_FINGER_SLOTS/3))
            plotter_initial_pv = setup_pv_plotter(f"初始设置: r={preview_r_init:.2f}, pos={preview_combo_init}")
            
            # 托盘几何体 (使用 T_actual_tray_geometry_world)
            # Initialize tray_pv_mesh_global here if not already, or ensure it's correctly transformed for visualization
            if tray_pv_mesh_global is None:
                 tray_pv_mesh_global = pv.Cylinder(center=(0, 0, -tray_height/2.0), direction=(0, 0, 1), radius=tray_radius, height=tray_height, resolution=30)
                 tray_pv_mesh_global.transform(T_actual_tray_geometry_world, inplace=True) # Make sure it's set for later GA viz
                 plotter_initial_pv.add_mesh(tray_pv_mesh_global.copy(), color=tray_color_fig1_init, opacity=0.3, name="initial_tray_mesh") # Use a copy for initial viz if main one is in-place
            else: # If already transformed (e.g. in section E), use a copy or re-transform a base mesh
                 temp_tray_for_init_viz = pv.Cylinder(center=(0, 0, -tray_height/2.0), direction=(0, 0, 1), radius=tray_radius, height=tray_height, resolution=30)
                 plotter_initial_pv.add_mesh(temp_tray_for_init_viz.transform(T_actual_tray_geometry_world, inplace=False), color=tray_color_fig1_init, opacity=0.3, name="initial_tray_mesh")

            # 物体点云 (使用 T_object_target_world_pose)
            if object_mesh_global_static_pv:
                if 'colors' in object_mesh_global_static_pv.point_data and object_mesh_global_static_pv.point_data['colors'] is not None:
                    plotter_initial_pv.add_mesh(object_mesh_global_static_pv, scalars='colors', rgba=True, style='points', point_size=5, name='initial_object_points_colored')
                else: plotter_initial_pv.add_mesh(object_mesh_global_static_pv, color=object_color_fig1_init_points, style='points', point_size=5, name='initial_object_points_default')
            if pv_obb_object_mesh_global: plotter_initial_pv.add_mesh(pv_obb_object_mesh_global, style='wireframe', color=object_obb_color_pv, line_width=1.5, name='initial_object_obb')

            # 示例未变形手指 (现在相对于 T_pose_for_tray_display_and_finger_placement_global)
            if initial_coords_ref_global is not None and faces_np_global is not None and width_translation_vector_global is not None:
                undeformed_verts_ref = np.vstack((initial_coords_ref_global, initial_coords_ref_global + width_translation_vector_global))
                undeformed_m_ref = pv.PolyData(undeformed_verts_ref, faces=faces_np_global)
                temp_T1_init, temp_T2_init = T1_translate_global, T2_rotate_global
                if temp_T1_init is None or temp_T2_init is None: 
                    temp_bottom_node_idx_init = np.argmin(initial_coords_ref_global[:,0]); temp_ref_mid_pt_init = initial_coords_ref_global[temp_bottom_node_idx_init] + width_translation_vector_global/2.0
                    temp_T1_init = create_transformation_matrix_opt8(None,-temp_ref_mid_pt_init)
                    temp_rot_ref_init = np.array([[0,1,0],[0,0,1],[1,0,0]]); temp_T2_init = create_transformation_matrix_opt8(temp_rot_ref_init,None)
                for i_finger_prev in range(len(preview_combo_init)):
                    pos_idx_prev = preview_combo_init[i_finger_prev]; angle_deg_prev = pos_idx_prev * (360.0 / N_FINGER_SLOTS); angle_rad_prev = np.radians(angle_deg_prev)
                    rot_z_prev = create_rotation_matrix_z(angle_rad_prev + np.pi / 2.0); target_pos_prev = np.array([preview_r_init * np.cos(angle_rad_prev), preview_r_init * np.sin(angle_rad_prev), 0.0])
                    T3_prev = create_transformation_matrix_opt8(rot_z_prev, target_pos_prev)
                    
                    # 示例手指现在也应该围绕 T_pose_for_tray_display_and_finger_placement_global 放置
                    T_finger_world_prev = T_pose_for_tray_display_and_finger_placement_global @ T3_prev @ temp_T2_init @ temp_T1_init
                    undeformed_finger_world_prev = undeformed_m_ref.transform(T_finger_world_prev, inplace=False)
                    plotter_initial_pv.add_mesh(undeformed_finger_world_prev, color=finger_color_fig1_init_undeformed, style='surface', opacity=0.8, smooth_shading=True, show_edges=True, edge_color='dimgray', line_width=0.5)
            
            if show_axes: plotter_initial_pv.add_axes(interactive=True, line_width=3)
            plotter_initial_pv.add_text(f"初始设置预览: r={preview_r_init:.2f}, pos={preview_combo_init}\n关闭窗口以继续", position="upper_left", font_size=10, color=text_color_fig1_init)
            plotter_initial_pv.camera_position = 'iso'; plotter_initial_pv.camera.zoom(1.2)
            if object_centroid_global_static is not None: plotter_initial_pv.set_focus(object_centroid_global_static)
            print("\n(PyVista) 显示初始物体、托盘和示例未变形手指。请关闭窗口以继续。")
            plotter_initial_pv.show(title="初始设置预览"); plotter_initial_pv.close()
            
            visualize_poses_with_open3d(
                tray_geometry_transform_mm=T_actual_tray_geometry_world, # Visual tray
                tray_axes_transform_mm=T_tray_axes_vis_world,          # Axes for tray/finger ref
                object_axes_transform_mm=T_object_axes_vis_world,      # Axes for object
                object_points_world_mm=object_points_global_static,    # Object points
                object_colors_rgb_float=sampled_object_colors_float_o3d_global, 
                object_obb_world=world_obb_object_global,
                tray_radius_mm=tray_radius, tray_height_mm=tray_height, 
                window_title="Open3D - 初始物体和托盘设置 (修正后Opt10BO逻辑)"
            )
        else: print("初始状态可视化：物体点云为空，跳过可视化。")
        print("--- 初始状态可视化结束 ---")

    # --- E. 初始化模型和手指参考 ---
    initial_coords_ref_global = load_initial_coordinates(INITIAL_COORDS_PATH, NODE_COUNT)
    model_global,scaler_X_global,scaler_y_global,device_global = load_prediction_components(MODEL_PATH,X_SCALER_PATH,Y_SCALER_PATH,INPUT_DIM,OUTPUT_DIM,HIDDEN_LAYER_1,HIDDEN_LAYER_2,HIDDEN_LAYER_3)
    if initial_coords_ref_global is None or model_global is None: sys.exit("错误：未能初始化手指模型或坐标。")
    faces_np_global = create_faces_array(NODE_COUNT)
    if faces_np_global is None or faces_np_global.size == 0: sys.exit("错误：未能创建手指表面。")
    width_translation_vector_global = np.array([0, finger_width, 0])
    bottom_node_idx = np.argmin(initial_coords_ref_global[:,0]); ref_mid_pt = initial_coords_ref_global[bottom_node_idx] + width_translation_vector_global/2.0
    T1_translate_global = create_transformation_matrix_opt8(None,-ref_mid_pt); rot_ref_to_local = np.array([[0,1,0],[0,0,1],[1,0,0]]); T2_rotate_global = create_transformation_matrix_opt8(rot_ref_to_local,None)
    
    # 确保 tray_pv_mesh_global (用于GA失败可视化) 已使用 T_actual_tray_geometry_world 进行变换
    if tray_pv_mesh_global is None: 
         tray_pv_mesh_global = pv.Cylinder(center=(0,0, -tray_height/2.0), direction=(0,0,1), radius=tray_radius, height=tray_height, resolution=30)
         tray_pv_mesh_global.transform(T_actual_tray_geometry_world, inplace=True)

    # --- F. 遗传算法 ---
    print(f"\n开始遗传算法优化 (群体大小: {GA_POPULATION_SIZE}, 代数: {GA_NUM_GENERATIONS})")
    print(f"目标物体: 从 {os.path.basename(EXTERNAL_OBJECT_PLY_PATH)} 导入, 最终世界质心: {object_centroid_global_static.round(2) if object_centroid_global_static is not None else 'N/A'}")
    if DEBUG_VISUALIZE_FAILED_GRASPS: print(f"失败步骤可视化已启用。最多显示 {MAX_FAILED_GRASP_VIS} 次。")
    population = []
    for _ in range(GA_POPULATION_SIZE): r_gene = random.uniform(R_BOUNDS[0], R_BOUNDS[1]); combo_idx_gene = random.randint(0, num_valid_combinations - 1); population.append([r_gene, combo_idx_gene])
    best_overall_gii = -float('inf'); best_overall_chromosome = None; best_overall_pressures = None; best_overall_meshes = None; best_overall_contacts = None
    for gen in range(GA_NUM_GENERATIONS):
        print(f"\n--- 第 {gen + 1}/{GA_NUM_GENERATIONS} 代 ---"); fitness_values = []; evaluated_population_details = []
        for i, chromosome in enumerate(population):
            r_val = chromosome[0]; combo_idx_val = int(chromosome[1]); pos_indices = all_valid_finger_combinations_global[combo_idx_val]; pos_idx1, pos_idx2, pos_idx3 = pos_indices[0], pos_indices[1], pos_indices[2]
            generation_debug_info = f"{gen+1}-{i+1}"; cost, pressures, meshes, contacts = evaluate_grasp(r_val, pos_idx1, pos_idx2, pos_idx3, generation_info=generation_debug_info)
            fitness = -cost; fitness_values.append(fitness); evaluated_population_details.append({"chromosome": chromosome, "fitness": fitness, "cost": cost, "pressures": pressures, "meshes": meshes, "contacts": contacts, "r": r_val, "pos_indices": pos_indices})
            if fitness > best_overall_gii: best_overall_gii = fitness; best_overall_chromosome = chromosome; best_overall_pressures = pressures; best_overall_meshes = meshes; best_overall_contacts = contacts; print(f"  ** 新的最优解! Gen {gen+1}, Ind {i+1}, GII = {-best_overall_gii:.4f}, r={r_val:.3f}, pos={pos_indices} **")
        valid_fitness_values = [f for f in fitness_values if f > -float('inf')]; min_fitness_adj = min(valid_fitness_values) if valid_fitness_values else 0; adjusted_fitness_values = [(f - min_fitness_adj + 1e-6) if f > -float('inf') else 1e-7 for f in fitness_values]; total_adjusted_fitness = sum(adjusted_fitness_values); new_population = []
        sorted_evaluated_population = sorted(evaluated_population_details, key=lambda x: x["fitness"], reverse=True)
        for i_elite in range(GA_ELITISM_COUNT):
            if i_elite < len(sorted_evaluated_population): new_population.append(list(sorted_evaluated_population[i_elite]["chromosome"]))
        num_to_select = GA_POPULATION_SIZE - GA_ELITISM_COUNT; population_for_selection = [{"chromosome": det["chromosome"], "adjusted_fitness": adj_fit} for det, adj_fit in zip(evaluated_population_details, adjusted_fitness_values)]
        for _ in range(num_to_select):
            if total_adjusted_fitness <= 1e-9: pick_details = random.choice(evaluated_population_details)
            else:
                r_pick = random.uniform(0, total_adjusted_fitness); current_sum_select = 0; selected_individual_details_fallback = None
                for individual_details_select in population_for_selection:
                    if selected_individual_details_fallback is None : selected_individual_details_fallback = individual_details_select
                    current_sum_select += individual_details_select["adjusted_fitness"]
                    if current_sum_select > r_pick: pick_details = individual_details_select; break
                else: pick_details = selected_individual_details_fallback if selected_individual_details_fallback else random.choice(population_for_selection)
            new_population.append(list(pick_details["chromosome"]))
        population = new_population; offspring_population = []; mating_pool = population[GA_ELITISM_COUNT:]; num_to_mate = len(mating_pool)
        for i_mate in range(0, num_to_mate, 2):
            if i_mate + 1 >= num_to_mate:
                if mating_pool: offspring_population.append(list(mating_pool[i_mate])); continue
            parent1 = mating_pool[i_mate]; parent2 = mating_pool[i_mate+1]; child1, child2 = list(parent1), list(parent2)
            if random.random() < GA_CX_PROB:
                alpha = random.random(); child1[0] = np.clip(alpha * parent1[0] + (1 - alpha) * parent2[0], R_BOUNDS[0], R_BOUNDS[1]); child2[0] = np.clip((1 - alpha) * parent1[0] + alpha * parent2[0], R_BOUNDS[0], R_BOUNDS[1])
                if random.random() < 0.5: child1[1], child2[1] = parent2[1], parent1[1]
            offspring_population.extend([child1, child2])
        while len(offspring_population) < (GA_POPULATION_SIZE - GA_ELITISM_COUNT):
            if mating_pool: offspring_population.append(list(random.choice(mating_pool)))
            elif population: offspring_population.append(list(random.choice(population)))
            else: offspring_population.append([random.uniform(R_BOUNDS[0], R_BOUNDS[1]), random.randint(0, num_valid_combinations - 1)])
        for i_mut in range(len(offspring_population)):
            if random.random() < GA_MUT_PROB: offspring_population[i_mut][0] = np.clip(offspring_population[i_mut][0] + random.gauss(0, (R_BOUNDS[1] - R_BOUNDS[0]) * 0.1), R_BOUNDS[0], R_BOUNDS[1])
            if random.random() < GA_MUT_PROB: offspring_population[i_mut][1] = random.randint(0, num_valid_combinations - 1)
        population = population[:GA_ELITISM_COUNT] + offspring_population[:(GA_POPULATION_SIZE - GA_ELITISM_COUNT)]
        current_best_gii_gen = max(valid_fitness_values) if valid_fitness_values else -float('inf')
        print(f"  代 {gen + 1} 最佳 GII: {-current_best_gii_gen:.4f} (成本: {current_best_gii_gen:.4f})"); print(f"  迄今为止最优 GII: {-best_overall_gii:.4f}")
    print("\n遗传算法优化结束。")

    # --- G. 处理和可视化优化结果 ---
    if best_overall_chromosome is not None:
        best_r_ga = best_overall_chromosome[0]; best_combo_idx_ga = int(best_overall_chromosome[1]); best_pos_indices_ga = all_valid_finger_combinations_global[best_combo_idx_ga]
        print("\n找到的最优参数 (GA):"); print(f"  r = {best_r_ga:.4f}, pos_indices = {best_pos_indices_ga}, (combo_idx = {best_combo_idx_ga})")
        print(f"最优 GII (GA): {-best_overall_gii:.4f} (成本: {best_overall_gii:.4f})"); print("\n使用最优参数生成最终可视化...")
        final_pressures_viz = best_overall_pressures; final_meshes_viz = best_overall_meshes
        if final_meshes_viz and len(final_meshes_viz) == 3 and all(m is not None for m in final_meshes_viz):
            final_chosen_indices_ga_viz = best_pos_indices_ga; pressures_display_str = "N/A"
            if final_pressures_viz is not None and hasattr(final_pressures_viz, 'size') and final_pressures_viz.size == 3: pressures_display_str = f"[{final_pressures_viz[0]:.0f}, {final_pressures_viz[1]:.0f}, {final_pressures_viz[2]:.0f}]"
            state_disp_ga=f"Cost={best_overall_gii:.3f}"; gii_disp_ga=f"GII={-best_overall_gii:.3f}" if best_overall_gii < -1e-9 else "GII:N/A"
            plotter_final_pv = setup_pv_plotter(title="PyVista - Optimal Grasp (GA - Imported Object)")
            if tray_pv_mesh_global: plotter_final_pv.add_mesh(tray_pv_mesh_global,color=tray_color_runtime,opacity=0.5,name='tray_final_pv_oriented')
            if object_mesh_global_static_pv:
                if 'colors' in object_mesh_global_static_pv.point_data and object_mesh_global_static_pv.point_data['colors'] is not None: plotter_final_pv.add_mesh(object_mesh_global_static_pv, scalars='colors', rgba=True, style='points', point_size=5, name='obj_final_pv_imported')
                else: plotter_final_pv.add_mesh(object_mesh_global_static_pv, color=object_point_color_runtime, style='points', point_size=5, name='obj_final_pv_imported')
            if pv_obb_object_mesh_global: plotter_final_pv.add_mesh(pv_obb_object_mesh_global, style='wireframe', color=object_obb_color_pv, line_width=2, name='object_obb_final_pv')
            if show_axes: plotter_final_pv.add_axes(interactive=True)
            for i_fv_ga,m_fv_ga in enumerate(final_meshes_viz):
                if m_fv_ga: plotter_final_pv.add_mesh(m_fv_ga,color=finger_color,style='surface',opacity=0.85,name=f'f_final_pv_{i_fv_ga}')
            params_txt_ga=f"r={best_r_ga:.2f}, P={pressures_display_str}, indices=({final_chosen_indices_ga_viz[0]},{final_chosen_indices_ga_viz[1]},{final_chosen_indices_ga_viz[2]})"
            plotter_final_pv.add_text(f"Optimal (GA):{params_txt_ga}\n{state_disp_ga}\n{gii_disp_ga}",position="upper_left",font_size=10,color=text_color_fig1_init)
            plotter_final_pv.camera_position='xy'; print("\n(PyVista) 显示最优抓取配置。请关闭窗口以继续。")
            plotter_final_pv.show(cpos='xy'); plotter_final_pv.close()
            
            visualize_poses_with_open3d(
                tray_geometry_transform_mm=T_actual_tray_geometry_world, # Visual tray
                tray_axes_transform_mm=T_tray_axes_vis_world,          # Axes for tray/finger ref
                object_axes_transform_mm=T_object_axes_vis_world,      # Axes for object
                object_points_world_mm=object_points_global_static,    # Object points
                object_colors_rgb_float=sampled_object_colors_float_o3d_global, 
                object_obb_world=world_obb_object_global,
                tray_radius_mm=tray_radius, tray_height_mm=tray_height, 
                window_title="Open3D - Optimal Poses & Axes (GA - Corrected Opt10BO Logic)")
        else: print("未能为最优参数生成手指网格或获取最终压力，无法进行PyVista最终可视化。");
        if best_overall_gii is not None : print(f"  (找到的最优成本为: {best_overall_gii:.3f})")
    else: print("\nGA未成功或无结果，无法显示最优。")
    print("\n程序结束。")