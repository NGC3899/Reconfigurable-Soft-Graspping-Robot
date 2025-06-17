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
import open3d as o3d # 确保 Open3D 已导入
import random # For selecting random distinct positions
import itertools # 导入 itertools 用于生成排列组合

# --- 调试可视化开关 ---
DEBUG_VISUALIZE_FAILED_GRASPS = False #PSO版本中可以按需开启
FAILED_GRASP_VIS_COUNT = 0
MAX_FAILED_GRASP_VIS = 5

# --- 打印版本信息 ---
try:
    print(f"Opt_10_PSO_Cuboid - 粒子群优化版本 (程序化长方体)") # 版本名更改
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

# --- 4. 配置参数 ---
tray_radius = 60.0; tray_height = 1.0; finger_width = 10.0
TARGET_POINT_COUNT_FOR_SIM = 2000

show_axes = True
finger_color = 'lightcoral'; tray_color = 'tan'; object_point_color = 'blue'
object_obb_color_pv = 'green'; object_obb_color_o3d = (0.1, 0.9, 0.1)

collision_threshold = 1.0; overlap_threshold = 1e-4
friction_coefficient = 0.5; eigenvalue_threshold = 1e-6
max_pressure = 40000.0
PRESSURE_STEP_EVAL_GRASP = 500.0; INITIAL_PRESSURE_EVAL_GRASP = 100.0
R_BOUNDS = (tray_radius * 0.3, tray_radius * 0.95)
N_FINGER_SLOTS = 10; DOT_PROD_TOLERANCE_LOCAL = 1e-6

# --- 长方体参数 (用户可定义) ---
CUBOID_DIMENSIONS = (150.0, 20.0, 15.0) # (长, 宽, 高) -> (x, y, z)
CUBOID_DISTANCE_FROM_BASE = 25.0 # 长方体底面中心到圆盘(z=0)的Z轴距离

CHARACTERISTIC_LENGTH_FOR_GII = 1.0 # 将在主脚本中根据物体尺寸计算

COST_MESH_FAILURE = 9.0; COST_OVERLAP = 5.0; COST_INTERSECTION = 4.0
COST_MAX_PRESSURE_NO_CONTACT = 6.0; COST_NO_CONTACT_OR_ITER_LIMIT = 3.0
COST_LOW_GII_OR_FEW_CONTACTS = 2.0

all_valid_finger_combinations_global = list(itertools.combinations(range(N_FINGER_SLOTS), 3))
num_valid_combinations = len(all_valid_finger_combinations_global)
print(f"总共有 {N_FINGER_SLOTS} 个手指槽位，生成了 {num_valid_combinations} 种唯一的（无序）三手指位置组合。")

# --- PSO 参数定义 ---
PSO_N_PARTICLES = 5  # 粒子数量
PSO_N_ITERATIONS = 4 # 迭代次数
PSO_W = 0.5           # 惯性权重
PSO_C1 = 1.5          # 个体学习因子
PSO_C2 = 1.5          # 群体学习因子
PSO_V_MAX_R_RATIO = 0.2 # r值的最大速度比例 (相对于R_BOUNDS范围)
PSO_V_MAX_COMBO_RATIO = 0.2 # combo_idx的最大速度比例 (相对于总组合数范围)


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
        n_contact = normal_finger / norm_mag
        try: t1, t2 = get_orthogonal_vectors(n_contact)
        except ValueError: continue
        r_contact = pt_on_mesh - object_centroid_for_gii
        d_list = [n_contact + mu * t1, n_contact - mu * t1, n_contact + mu * t2, n_contact - mu * t2]
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
def create_cuboid_point_cloud(dimensions, center, n_points=500):
    box_mesh = o3d.geometry.TriangleMesh.create_box(width=dimensions[0], depth=dimensions[1], height=dimensions[2])
    box_mesh.translate(-np.array(dimensions) / 2.0, relative=True)
    pcd = box_mesh.sample_points_uniformly(number_of_points=n_points)
    pcd.translate(center, relative=False)
    return np.asarray(pcd.points)
def visualize_poses_with_open3d(tray_geometry_transform_mm, tray_axes_transform_mm, object_axes_transform_mm, object_points_world_mm, object_colors_rgb_float, object_obb_world, tray_radius_mm, tray_height_mm, window_title="Open3D Relative Pose Visualization"):
    print(f"\n--- 打开 Open3D 窗口: {window_title} ---")
    geometries = []
    o3d_tray_canonical_centered = o3d.geometry.TriangleMesh.create_cylinder(radius=tray_radius_mm, height=tray_height_mm, resolution=20, split=4)
    o3d_tray_canonical_centered.translate(np.array([0,0,-tray_height_mm/2.0]))
    o3d_tray_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_tray_canonical_centered)
    o3d_tray_wireframe.transform(tray_geometry_transform_mm); o3d_tray_wireframe.paint_uniform_color(pv.Color(tray_color).float_rgb)
    geometries.append(o3d_tray_wireframe)
    o3d_object_pcd = o3d.geometry.PointCloud()
    if object_points_world_mm is not None and object_points_world_mm.shape[0] > 0:
        o3d_object_pcd.points = o3d.utility.Vector3dVector(object_points_world_mm)
        if object_colors_rgb_float is not None and len(object_colors_rgb_float) == len(object_points_world_mm): o3d_object_pcd.colors = o3d.utility.Vector3dVector(object_colors_rgb_float)
        else: o3d_object_pcd.paint_uniform_color(pv.Color(object_point_color).float_rgb)
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
T_pose_for_tray_display_and_finger_placement_global = None # Used in evaluate_grasp
tray_pv_mesh_global = None

def visualize_failed_pso_step(r_param, chosen_indices_param, pressures_param, finger_meshes_param, failure_reason_str, iteration, particle_idx):
    global DEBUG_VISUALIZE_FAILED_GRASPS, FAILED_GRASP_VIS_COUNT, MAX_FAILED_GRASP_VIS, object_mesh_global_static_pv, tray_pv_mesh_global, pv_obb_object_mesh_global, show_axes
    if not DEBUG_VISUALIZE_FAILED_GRASPS or FAILED_GRASP_VIS_COUNT >= MAX_FAILED_GRASP_VIS:
        if FAILED_GRASP_VIS_COUNT == MAX_FAILED_GRASP_VIS and DEBUG_VISUALIZE_FAILED_GRASPS: print("已达到最大失败可视化次数，后续失败将不再显示。"); FAILED_GRASP_VIS_COUNT +=1
        return
    FAILED_GRASP_VIS_COUNT += 1; plotter_title = f"PSO失败调试 Iter {iteration} Particle {particle_idx} #{FAILED_GRASP_VIS_COUNT}: {failure_reason_str}"
    plotter_debug = pv.Plotter(window_size=[900,700], title=plotter_title)
    if tray_pv_mesh_global: plotter_debug.add_mesh(tray_pv_mesh_global, color=tray_color, opacity=0.5, name='tray_debug')
    if object_mesh_global_static_pv: plotter_debug.add_mesh(object_mesh_global_static_pv, color=object_point_color, style='surface', opacity=0.6, name='obj_debug_cuboid')
    if pv_obb_object_mesh_global: plotter_debug.add_mesh(pv_obb_object_mesh_global, style='wireframe', color=object_obb_color_pv, line_width=2, name='obj_obb_debug')
    if show_axes: plotter_debug.add_axes(interactive=True)
    if finger_meshes_param:
        for i, mesh in enumerate(finger_meshes_param):
            if mesh: plotter_debug.add_mesh(mesh, color=finger_color, style='surface', opacity=0.75, name=f'finger_debug_{i}')
    params_text = f"r={r_param:.3f}, 位置=({chosen_indices_param[0]},{chosen_indices_param[1]},{chosen_indices_param[2]})\n"; pressures_text = f"压力=[{pressures_param[0]:.0f}, {pressures_param[1]:.0f}, {pressures_param[2]:.0f}]\n"; reason_text = f"原因: {failure_reason_str}"
    plotter_debug.add_text(params_text + pressures_text + reason_text, position="upper_left", font_size=10, color='black')
    plotter_debug.camera_position = 'xy'; print(f"\n(PyVista) 显示PSO失败步骤 Iter {iteration} Particle {particle_idx} #{FAILED_GRASP_VIS_COUNT}。原因: {failure_reason_str}。请关闭窗口以继续...")
    plotter_debug.show(cpos='xy'); plotter_debug.close()

def evaluate_grasp(r, pos_idx1, pos_idx2, pos_idx3, iteration_info="N/A"):
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
        print(f"错误: T_pose_for_tray_display_and_finger_placement_global 未在 evaluate_grasp 中初始化! (IterInfo: {iteration_info})")
        return 20.0, np.full(3, INITIAL_PRESSURE_EVAL_GRASP, dtype=float), [None]*3, None

    chosen_indices = [int(pos_idx1), int(pos_idx2), int(pos_idx3)]
    current_call_params_str = f"r={r:.3f}, 位置=({chosen_indices[0]},{chosen_indices[1]},{chosen_indices[2]})"

    return_pressures = np.full(3, INITIAL_PRESSURE_EVAL_GRASP, dtype=float)
    return_meshes = [None] * 3
    return_contacts = None

    cost_to_return = 20.0; gii_value_to_print = "N/A"; failure_reason_for_vis = ""
    deformed_finger_meshes_at_contact = [None] * 3

    try:
        current_pressures = np.full(3, INITIAL_PRESSURE_EVAL_GRASP, dtype=float)
        finger_contact_established = [False] * 3
        max_pressure_iterations = int((max_pressure - INITIAL_PRESSURE_EVAL_GRASP) / PRESSURE_STEP_EVAL_GRASP) * 3 + 25
        current_pressure_iter = 0
        while True:
            current_pressure_iter += 1
            if current_pressure_iter > max_pressure_iterations:
                cost_to_return = COST_NO_CONTACT_OR_ITER_LIMIT; failure_reason_for_vis = "迭代超限"
                print(f"  PSO评估 ({iteration_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                if DEBUG_VISUALIZE_FAILED_GRASPS:
                    iter_num_viz, particle_idx_viz = "N/A", "N/A"
                    if isinstance(iteration_info, str) and "-" in iteration_info:
                        parts = iteration_info.split("-");
                        if len(parts) == 2: iter_num_viz, particle_idx_viz = parts[0].strip(), parts[1].strip()
                    visualize_failed_pso_step(r, chosen_indices, current_pressures, deformed_finger_meshes_at_contact, failure_reason_for_vis, iter_num_viz, particle_idx_viz)
                return (cost_to_return, current_pressures, deformed_finger_meshes_at_contact, None)
            pressure_changed_this_step = False
            for i in range(3):
                if not finger_contact_established[i] and current_pressures[i] < max_pressure:
                    current_pressures[i] += PRESSURE_STEP_EVAL_GRASP
                    current_pressures[i] = min(current_pressures[i], max_pressure); pressure_changed_this_step = True
            current_step_finger_meshes = [None] * 3; mesh_generation_successful_this_iter = True
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
                angle_rad_pso = np.radians(current_angle_deg)
                rot_angle_z_placing = angle_rad_pso + np.pi / 2.0
                rot_z_placing = create_rotation_matrix_z(rot_angle_z_placing)
                target_pos_on_circle = np.array([ r * np.cos(angle_rad_pso), r * np.sin(angle_rad_pso), 0.0 ])
                T3_place = create_transformation_matrix_opt8(rot_z_placing, target_pos_on_circle)
                T_transform_finger_relative_to_tray_origin = T3_place @ T2_rotate_global @ T1_translate_global
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
                cost_to_return = COST_MESH_FAILURE; failure_reason_for_vis = "网格生成失败"
                print(f"  PSO评估 ({iteration_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                if DEBUG_VISUALIZE_FAILED_GRASPS:
                    iter_num_viz, particle_idx_viz = "N/A", "N/A"
                    if isinstance(iteration_info, str) and "-" in iteration_info:
                        parts = iteration_info.split("-");
                        if len(parts) == 2: iter_num_viz, particle_idx_viz = parts[0].strip(), parts[1].strip()
                    visualize_failed_pso_step(r, chosen_indices, current_pressures, deformed_finger_meshes_at_contact, failure_reason_for_vis, iter_num_viz, particle_idx_viz)
                return (cost_to_return, current_pressures, deformed_finger_meshes_at_contact, None)
            for i_mesh_update in range(3):
                if current_step_finger_meshes[i_mesh_update] is not None:
                     deformed_finger_meshes_at_contact[i_mesh_update] = current_step_finger_meshes[i_mesh_update].copy()
            has_overlap_this_iter = False; contact_made_by_finger_this_iter_flags = [False] * 3; finger_dot_products_this_iter = [[] for _ in range(3)]
            if object_points_global_static is not None and num_object_points_global_static > 0:
                for obj_pt_idx, obj_point in enumerate(object_points_global_static):
                    closest_dist_for_obj_pt = float('inf'); winning_finger_for_obj_pt = -1
                    winning_normal_for_obj_pt = None; winning_pt_on_mesh_for_obj_pt = None
                    for finger_idx, finger_mesh in enumerate(current_step_finger_meshes):
                        if finger_mesh is None or finger_mesh.n_cells == 0 : continue
                        has_normals = 'Normals' in finger_mesh.cell_data and finger_mesh.cell_data['Normals'] is not None and len(finger_mesh.cell_data['Normals']) == finger_mesh.n_cells
                        try:
                            closest_cell_id, pt_on_mesh_candidate = finger_mesh.find_closest_cell(obj_point, return_closest_point=True)
                            if closest_cell_id < 0 or closest_cell_id >= finger_mesh.n_cells: continue
                            dist = LA.norm(obj_point - pt_on_mesh_candidate)
                            if dist < closest_dist_for_obj_pt:
                                closest_dist_for_obj_pt = dist; winning_finger_for_obj_pt = finger_idx; winning_pt_on_mesh_for_obj_pt = pt_on_mesh_candidate
                                if has_normals: winning_normal_for_obj_pt = finger_mesh.cell_normals[closest_cell_id]
                                else: winning_normal_for_obj_pt = None
                            if dist < collision_threshold: contact_made_by_finger_this_iter_flags[finger_idx] = True
                        except RuntimeError: continue
                    if winning_finger_for_obj_pt != -1:
                        if closest_dist_for_obj_pt < overlap_threshold: has_overlap_this_iter = True; break
                        elif closest_dist_for_obj_pt < collision_threshold:
                            if winning_normal_for_obj_pt is not None and LA.norm(winning_normal_for_obj_pt) > 1e-9:
                                vector_to_point = obj_point - winning_pt_on_mesh_for_obj_pt
                                if LA.norm(vector_to_point) > 1e-9:
                                    dot_prod = np.dot(vector_to_point / LA.norm(vector_to_point), winning_normal_for_obj_pt / LA.norm(winning_normal_for_obj_pt))
                                    finger_dot_products_this_iter[winning_finger_for_obj_pt].append(dot_prod)
                if has_overlap_this_iter:
                    cost_to_return = COST_OVERLAP; failure_reason_for_vis = "重叠"
                    print(f"  PSO评估 ({iteration_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                    if DEBUG_VISUALIZE_FAILED_GRASPS:
                        iter_num_viz, particle_idx_viz = "N/A", "N/A";
                        if isinstance(iteration_info, str) and "-" in iteration_info: parts = iteration_info.split("-");
                        if len(parts) == 2: iter_num_viz, particle_idx_viz = parts[0].strip(), parts[1].strip()
                        visualize_failed_pso_step(r, chosen_indices, current_pressures, current_step_finger_meshes, failure_reason_for_vis, iter_num_viz, particle_idx_viz)
                    return (cost_to_return, current_pressures, current_step_finger_meshes, None)
            finger_intersects_this_iter = [False] * 3
            if not has_overlap_this_iter:
                for i in range(3):
                    if contact_made_by_finger_this_iter_flags[i] and finger_dot_products_this_iter[i]:
                        has_pos_dp = any(dp > DOT_PROD_TOLERANCE_LOCAL for dp in finger_dot_products_this_iter[i])
                        has_neg_dp = any(dp < -DOT_PROD_TOLERANCE_LOCAL for dp in finger_dot_products_this_iter[i])
                        if has_pos_dp and has_neg_dp: finger_intersects_this_iter[i] = True
            if any(finger_intersects_this_iter):
                cost_to_return = COST_INTERSECTION; failure_reason_for_vis = "穿透"
                print(f"  PSO评估 ({iteration_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                if DEBUG_VISUALIZE_FAILED_GRASPS:
                    iter_num_viz, particle_idx_viz = "N/A", "N/A";
                    if isinstance(iteration_info, str) and "-" in iteration_info: parts = iteration_info.split("-");
                    if len(parts) == 2: iter_num_viz, particle_idx_viz = parts[0].strip(), parts[1].strip()
                    visualize_failed_pso_step(r, chosen_indices, current_pressures, current_step_finger_meshes, failure_reason_for_vis, iter_num_viz, particle_idx_viz)
                return (cost_to_return, current_pressures, current_step_finger_meshes, None)
            all_fingers_established_this_iter_round = True
            for i in range(3):
                if contact_made_by_finger_this_iter_flags[i]:
                    if not finger_contact_established[i]: finger_contact_established[i] = True
                if not finger_contact_established[i]: all_fingers_established_this_iter_round = False
            if all_fingers_established_this_iter_round:
                return_pressures = current_pressures.copy(); return_meshes = [m.copy() if m else None for m in deformed_finger_meshes_at_contact]
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
                        if best_contact_for_this_finger: final_gii_contacts_collected.append(best_contact_for_this_finger)
                return_contacts = final_gii_contacts_collected
                if len(final_gii_contacts_collected) >= 2:
                    gii = calculate_gii_multi_contact(final_gii_contacts_collected, object_centroid_global_static, friction_coefficient, eigenvalue_threshold, characteristic_length=CHARACTERISTIC_LENGTH_FOR_GII)
                    if gii is not None and gii > 1e-9:
                        cost_to_return = -gii; gii_value_to_print = gii
                        print(f"  PSO评估 ({iteration_info}): {current_call_params_str} -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print:.4f}")
                        return (cost_to_return, return_pressures, return_meshes, return_contacts)
                    else:
                        cost_to_return = COST_LOW_GII_OR_FEW_CONTACTS; gii_value_to_print = gii if gii is not None else "计算失败"
                        failure_reason_for_vis = f"低GII ({gii_value_to_print if isinstance(gii_value_to_print, str) else gii_value_to_print:.4f})"
                        print(f"  PSO评估 ({iteration_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print if isinstance(gii_value_to_print, str) else gii_value_to_print:.4f}")
                        if DEBUG_VISUALIZE_FAILED_GRASPS:
                            iter_num_viz, particle_idx_viz = "N/A", "N/A";
                            if isinstance(iteration_info, str) and "-" in iteration_info: parts = iteration_info.split("-");
                            if len(parts) == 2: iter_num_viz, particle_idx_viz = parts[0].strip(), parts[1].strip()
                            visualize_failed_pso_step(r, chosen_indices, return_pressures, return_meshes, failure_reason_for_vis, iter_num_viz, particle_idx_viz)
                        return (cost_to_return, return_pressures, return_meshes, return_contacts)
                else:
                    cost_to_return = COST_LOW_GII_OR_FEW_CONTACTS; failure_reason_for_vis = "接触点不足 GII"
                    print(f"  PSO评估 ({iteration_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                    if DEBUG_VISUALIZE_FAILED_GRASPS:
                        iter_num_viz, particle_idx_viz = "N/A", "N/A";
                        if isinstance(iteration_info, str) and "-" in iteration_info: parts = iteration_info.split("-");
                        if len(parts) == 2: iter_num_viz, particle_idx_viz = parts[0].strip(), parts[1].strip()
                        visualize_failed_pso_step(r, chosen_indices, return_pressures, return_meshes, failure_reason_for_vis, iter_num_viz, particle_idx_viz)
                    return (cost_to_return, return_pressures, return_meshes, return_contacts)
            for i in range(3):
                if not finger_contact_established[i] and current_pressures[i] >= max_pressure:
                    cost_to_return = COST_MAX_PRESSURE_NO_CONTACT; failure_reason_for_vis = f"手指 {i} 最大压力无接触"
                    print(f"  PSO评估 ({iteration_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                    if DEBUG_VISUALIZE_FAILED_GRASPS:
                        iter_num_viz, particle_idx_viz = "N/A", "N/A";
                        if isinstance(iteration_info, str) and "-" in iteration_info: parts = iteration_info.split("-");
                        if len(parts) == 2: iter_num_viz, particle_idx_viz = parts[0].strip(), parts[1].strip()
                        visualize_failed_pso_step(r, chosen_indices, current_pressures, deformed_finger_meshes_at_contact, failure_reason_for_vis, iter_num_viz, particle_idx_viz)
                    return (cost_to_return, current_pressures, deformed_finger_meshes_at_contact, None)
            if not pressure_changed_this_step and not all(finger_contact_established):
                cost_to_return = COST_MAX_PRESSURE_NO_CONTACT; failure_reason_for_vis = "停滞无接触"
                print(f"  PSO评估 ({iteration_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                if DEBUG_VISUALIZE_FAILED_GRASPS:
                    iter_num_viz, particle_idx_viz = "N/A", "N/A";
                    if isinstance(iteration_info, str) and "-" in iteration_info: parts = iteration_info.split("-");
                    if len(parts) == 2: iter_num_viz, particle_idx_viz = parts[0].strip(), parts[1].strip()
                    visualize_failed_pso_step(r, chosen_indices, current_pressures, deformed_finger_meshes_at_contact, failure_reason_for_vis, iter_num_viz, particle_idx_viz)
                return (cost_to_return, current_pressures, deformed_finger_meshes_at_contact, None)
    except Exception as e_eval_grasp:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"DEBUG_PSO: EXCEPTION in evaluate_grasp for {current_call_params_str} (IterInfo: {iteration_info}): {e_eval_grasp}")
        traceback.print_exc(); print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        cost_to_return = 20.0; failure_reason_for_vis = "evaluate_grasp内部异常"
        current_pressures_fallback = current_pressures if 'current_pressures' in locals() else np.full(3, INITIAL_PRESSURE_EVAL_GRASP, dtype=float)
        deformed_meshes_fallback = deformed_finger_meshes_at_contact
        print(f"  PSO评估 ({iteration_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
        if DEBUG_VISUALIZE_FAILED_GRASPS:
            iter_num_viz, particle_idx_viz = "N/A", "N/A";
            if isinstance(iteration_info, str) and "-" in iteration_info: parts = iteration_info.split("-");
            if len(parts) == 2: iter_num_viz, particle_idx_viz = parts[0].strip(), parts[1].strip()
            visualize_failed_pso_step(r, chosen_indices, current_pressures_fallback, deformed_meshes_fallback, failure_reason_for_vis, iter_num_viz, particle_idx_viz)
        return (cost_to_return, None, None, None)

# --- 主脚本 ---
if __name__ == '__main__':
    # --- A. 定义场景和物体 (新: 程序化) ---
    # 定义圆盘/夹爪参考系在世界坐标系中的位姿 (即原点)
    T_pose_for_tray_display_and_finger_placement_global = np.identity(4)
    T_actual_tray_geometry_world = np.identity(4)
    T_tray_axes_vis_world = np.identity(4)

    # --- B. 创建长方体物体 ---
    cuboid_l, cuboid_w, cuboid_h = CUBOID_DIMENSIONS
    CUBOID_CENTER_WORLD = np.array([0.0, 0.0, CUBOID_DISTANCE_FROM_BASE + cuboid_h / 2.0])
    object_points_global_static = create_cuboid_point_cloud(CUBOID_DIMENSIONS, CUBOID_CENTER_WORLD, n_points=TARGET_POINT_COUNT_FOR_SIM)
    CHARACTERISTIC_LENGTH_FOR_GII = np.linalg.norm(np.array(CUBOID_DIMENSIONS))
    if object_points_global_static is None or object_points_global_static.shape[0] == 0:
        sys.exit("错误: 未能生成长方体物体点云")

    object_centroid_global_static = np.mean(object_points_global_static, axis=0)
    num_object_points_global_static = object_points_global_static.shape[0]
    print(f"\n已生成长方体物体，尺寸: {CUBOID_DIMENSIONS}, 中心: {object_centroid_global_static.round(2)}, 点数: {num_object_points_global_static}")

    # --- C. 创建可视化模型 ---
    object_mesh_global_static_pv = pv.Cube(center=CUBOID_CENTER_WORLD, x_length=cuboid_l, y_length=cuboid_w, z_length=cuboid_h)
    world_obb_object_global = o3d.geometry.OrientedBoundingBox()
    world_obb_object_global.center = CUBOID_CENTER_WORLD
    world_obb_object_global.R = np.identity(3)
    world_obb_object_global.extent = np.array(CUBOID_DIMENSIONS)
    world_obb_object_global.color = object_obb_color_o3d
    pv_obb_object_mesh_global = object_mesh_global_static_pv.copy()
    T_object_axes_vis_world = create_transformation_matrix_opt8(np.identity(3), CUBOID_CENTER_WORLD)
    tray_pv_mesh_global = pv.Cylinder(center=(0,0, -tray_height/2.0), direction=(0,0,1), radius=tray_radius, height=tray_height, resolution=30)
    tray_pv_mesh_global.transform(T_actual_tray_geometry_world, inplace=True)

    # --- D. 初始化模型和手指参考 ---
    initial_coords_ref_global = load_initial_coordinates(INITIAL_COORDS_PATH, NODE_COUNT)
    model_global,scaler_X_global,scaler_y_global,device_global = load_prediction_components(MODEL_PATH,X_SCALER_PATH,Y_SCALER_PATH,INPUT_DIM,OUTPUT_DIM,HIDDEN_LAYER_1,HIDDEN_LAYER_2,HIDDEN_LAYER_3)
    if initial_coords_ref_global is None or model_global is None: sys.exit("错误：未能初始化手指模型或坐标。")
    faces_np_global = create_faces_array(NODE_COUNT)
    if faces_np_global is None or faces_np_global.size == 0: sys.exit("错误：未能创建手指表面。")
    width_translation_vector_global = np.array([0, finger_width, 0])
    bottom_node_idx = np.argmin(initial_coords_ref_global[:,0]); ref_mid_pt = initial_coords_ref_global[bottom_node_idx] + width_translation_vector_global/2.0
    T1_translate_global = create_transformation_matrix_opt8(None,-ref_mid_pt); rot_ref_to_local = np.array([[0,1,0],[0,0,1],[1,0,0]]); T2_rotate_global = create_transformation_matrix_opt8(rot_ref_to_local,None)

    # --- E. 粒子群优化 (PSO) ---
    print(f"\n开始粒子群优化 (粒子数: {PSO_N_PARTICLES}, 迭代次数: {PSO_N_ITERATIONS})")
    print(f"目标物体: 程序化长方体, 尺寸={CUBOID_DIMENSIONS}")
    if DEBUG_VISUALIZE_FAILED_GRASPS: print(f"失败步骤可视化已启用。最多显示 {MAX_FAILED_GRASP_VIS} 次。")

    particles_position = np.zeros((PSO_N_PARTICLES, 2))
    particles_velocity = np.zeros((PSO_N_PARTICLES, 2))
    particles_pbest_position = np.zeros((PSO_N_PARTICLES, 2))
    particles_pbest_fitness = np.full(PSO_N_PARTICLES, -float('inf'))
    particles_pbest_pressures = [None] * PSO_N_PARTICLES
    particles_pbest_meshes = [None] * PSO_N_PARTICLES
    particles_pbest_contacts = [None] * PSO_N_PARTICLES
    gbest_position = np.zeros(2); gbest_fitness = -float('inf')
    gbest_pressures = None; gbest_meshes = None; gbest_contacts = None

    v_max_r = PSO_V_MAX_R_RATIO * (R_BOUNDS[1] - R_BOUNDS[0]); v_min_r = -v_max_r
    v_max_combo = PSO_V_MAX_COMBO_RATIO * (num_valid_combinations -1); v_min_combo = -v_max_combo

    for i in range(PSO_N_PARTICLES):
        particles_position[i, 0] = random.uniform(R_BOUNDS[0], R_BOUNDS[1])
        particles_position[i, 1] = float(random.randint(0, num_valid_combinations - 1))
        particles_velocity[i, 0] = random.uniform(v_min_r*0.1, v_max_r*0.1)
        particles_velocity[i, 1] = random.uniform(v_min_combo*0.1, v_max_combo*0.1)
        particles_pbest_position[i, :] = particles_position[i, :].copy()

    for iteration in range(PSO_N_ITERATIONS):
        print(f"\n--- PSO 迭代 {iteration + 1}/{PSO_N_ITERATIONS} ---")
        for i in range(PSO_N_PARTICLES):
            r_val = particles_position[i, 0]
            combo_idx_val = int(round(particles_position[i, 1]))
            combo_idx_val = max(0, min(combo_idx_val, num_valid_combinations - 1))
            particles_position[i, 1] = float(combo_idx_val)
            pos_indices = all_valid_finger_combinations_global[combo_idx_val]
            pos_idx1, pos_idx2, pos_idx3 = pos_indices[0], pos_indices[1], pos_indices[2]
            iteration_debug_info = f"{iteration+1}-{i+1}"
            cost, pressures, meshes, contacts = evaluate_grasp(r_val, pos_idx1, pos_idx2, pos_idx3, iteration_info=iteration_debug_info)
            current_fitness = -cost
            if current_fitness > particles_pbest_fitness[i]:
                particles_pbest_fitness[i] = current_fitness
                particles_pbest_position[i, :] = particles_position[i, :].copy()
                particles_pbest_pressures[i] = pressures; particles_pbest_meshes[i] = meshes; particles_pbest_contacts[i] = contacts
            if particles_pbest_fitness[i] > gbest_fitness:
                gbest_fitness = particles_pbest_fitness[i]
                gbest_position[:] = particles_pbest_position[i, :].copy()
                gbest_pressures = particles_pbest_pressures[i]; gbest_meshes = particles_pbest_meshes[i]; gbest_contacts = particles_pbest_contacts[i]
                print(f"  ** 新的全局最优! Iter {iteration+1}, Particle {i+1}, GII = {-gbest_fitness:.4f}, r={gbest_position[0]:.3f}, combo_idx={int(gbest_position[1])} **")
        for i in range(PSO_N_PARTICLES):
            r1 = random.random(); r2 = random.random()
            particles_velocity[i, 0] = (PSO_W * particles_velocity[i, 0] + PSO_C1 * r1 * (particles_pbest_position[i, 0] - particles_position[i, 0]) + PSO_C2 * r2 * (gbest_position[0] - particles_position[i, 0]))
            particles_velocity[i, 0] = np.clip(particles_velocity[i, 0], v_min_r, v_max_r)
            particles_velocity[i, 1] = (PSO_W * particles_velocity[i, 1] + PSO_C1 * r1 * (particles_pbest_position[i, 1] - particles_position[i, 1]) + PSO_C2 * r2 * (gbest_position[1] - particles_position[i, 1]))
            particles_velocity[i, 1] = np.clip(particles_velocity[i, 1], v_min_combo, v_max_combo)
            particles_position[i, 0] = particles_position[i, 0] + particles_velocity[i, 0]
            particles_position[i, 0] = np.clip(particles_position[i, 0], R_BOUNDS[0], R_BOUNDS[1])
            particles_position[i, 1] = particles_position[i, 1] + particles_velocity[i, 1]
            particles_position[i, 1] = np.clip(particles_position[i, 1], 0, num_valid_combinations - 1)
        current_best_gii_iter = gbest_fitness
        print(f"  迭代 {iteration + 1} 结束。当前全局最优 GII: {-current_best_gii_iter:.4f}")
    print("\n粒子群优化结束。")

    # --- F. 处理和可视化优化结果 ---
    if gbest_fitness > -float('inf'):
        best_r_pso = gbest_position[0]
        best_combo_idx_pso = int(round(gbest_position[1]))
        best_combo_idx_pso = max(0, min(best_combo_idx_pso, num_valid_combinations - 1))
        best_pos_indices_pso = all_valid_finger_combinations_global[best_combo_idx_pso]

        print("\n找到的最优参数 (PSO):")
        print(f"  r = {best_r_pso:.4f}, pos_indices = {best_pos_indices_pso}, (combo_idx = {best_combo_idx_pso})")
        print(f"最优 GII (PSO): {-gbest_fitness:.4f} (成本: {gbest_fitness:.4f})")
        print("\n使用最优参数生成最终可视化...")

        final_pressures_viz = gbest_pressures; final_meshes_viz = gbest_meshes
        if final_meshes_viz and len(final_meshes_viz) == 3 and all(m is not None for m in final_meshes_viz):
            pressures_display_str = "N/A"
            if final_pressures_viz is not None and hasattr(final_pressures_viz, 'size') and final_pressures_viz.size == 3:
                pressures_display_str = f"[{final_pressures_viz[0]:.0f}, {final_pressures_viz[1]:.0f}, {final_pressures_viz[2]:.0f}]"
            state_disp_pso=f"Cost={gbest_fitness:.3f}"; gii_disp_pso=f"GII={-gbest_fitness:.3f}" if gbest_fitness < -1e-9 else "GII:N/A"
            plotter_final_pv = pv.Plotter(window_size=[1000,800],title="PyVista - Optimal Grasp (PSO - Cuboid)")
            plotter_final_pv.add_mesh(tray_pv_mesh_global,color=tray_color,opacity=0.5,name='tray_final_pv_oriented')
            plotter_final_pv.add_mesh(object_mesh_global_static_pv, color=object_point_color, style='surface', opacity=0.6, name='obj_final_pv_cuboid')
            if pv_obb_object_mesh_global: plotter_final_pv.add_mesh(pv_obb_object_mesh_global, style='wireframe', color=object_obb_color_pv, line_width=2, name='object_obb_final_pv')
            if show_axes: plotter_final_pv.add_axes(interactive=True)
            for i_fv_pso,m_fv_pso in enumerate(final_meshes_viz):
                if m_fv_pso: plotter_final_pv.add_mesh(m_fv_pso,color=finger_color,style='surface',opacity=0.85,name=f'f_final_pv_{i_fv_pso}')
            params_txt_pso=f"r={best_r_pso:.2f}, P={pressures_display_str}, indices=({best_pos_indices_pso[0]},{best_pos_indices_pso[1]},{best_pos_indices_pso[2]})"
            plotter_final_pv.add_text(f"Optimal (PSO):{params_txt_pso}\n{state_disp_pso}\n{gii_disp_pso}",position="upper_left",font_size=10,color='black')
            plotter_final_pv.camera_position='xy'; print("\n(PyVista) 显示最优抓取配置。请关闭窗口以继续。")
            plotter_final_pv.show(cpos='xy'); plotter_final_pv.close()
            visualize_poses_with_open3d(
                tray_geometry_transform_mm=T_actual_tray_geometry_world, tray_axes_transform_mm=T_tray_axes_vis_world,
                object_axes_transform_mm=T_object_axes_vis_world, object_points_world_mm=object_points_global_static,
                object_colors_rgb_float=None, object_obb_world=world_obb_object_global,
                tray_radius_mm=tray_radius, tray_height_mm=tray_height,
                window_title="Open3D - Optimal Poses & Axes (PSO - Cuboid)")
        else:
            print("未能为最优参数生成手指网格或获取最终压力，无法进行PyVista最终可视化。")
            if gbest_fitness is not None : print(f"  (找到的最优成本为: {gbest_fitness:.3f}, 对应GII: {-gbest_fitness:.3f})")
    else:
        print("\nPSO未成功或无结果，无法显示最优。")
    print("\n程序结束。")

