# -*- coding: utf-8 -*-
# 本脚本使用遗传算法的方式进行构型推荐程序的目标函数求解，使用理想球形点云测试
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
DEBUG_VISUALIZE_FAILED_GRASPS = False # 设置为 True 以在 GA 失败步骤时进行可视化
FAILED_GRASP_VIS_COUNT = 0 # 用于给失败可视化窗口编号
MAX_FAILED_GRASP_VIS = 5 # 最多显示多少个失败的可视化，防止过多窗口

# --- 打印版本信息 ---
try:
    print(f"Opt_10_GA_Sphere - 使用遗传算法和程序化球体进行测试")
    print(f"Open3D version: {o3d.__version__}")
    print(f"PyVista version: {pv.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Torch version: {torch.__version__}")
except NameError:
    pass
except Exception as e:
    print(f"Error printing library versions: {e}")

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
NODE_COUNT = 63
OUTPUT_DIM = NODE_COUNT * 3
HIDDEN_LAYER_1 = 128
HIDDEN_LAYER_2 = 256
HIDDEN_LAYER_3 = 128

# --- 3. 文件路径定义 (模型和初始坐标仍然需要) ---
MODEL_PATH = 'best_mlp_model.pth'
X_SCALER_PATH = 'x_scaler.joblib'
Y_SCALER_PATH = 'y_scaler.joblib'
INITIAL_COORDS_PATH = 'initial_coordinates.txt'

# --- 4. 配置参数 ---
tray_radius = 60.0
tray_height = 1.0

finger_width = 10.0
TARGET_POINT_COUNT_FOR_SIM = 500 
show_axes = True
finger_color = 'lightcoral'
tray_color = 'tan'
object_point_color = 'blue' 
object_obb_color_pv = 'green' 
object_obb_color_o3d = (0.1, 0.9, 0.1)

collision_threshold = 1.0
overlap_threshold = 1e-4
friction_coefficient = 0.5
eigenvalue_threshold = 1e-6
max_pressure = 40000.0
PRESSURE_STEP_EVAL_GRASP = 500.0 # 注意：evaluate_grasp 内部的压力迭代步长
INITIAL_PRESSURE_EVAL_GRASP = 100.0 # evaluate_grasp 内部的初始压力

R_BOUNDS = (tray_radius * 0.3, tray_radius * 0.95) 
N_FINGER_SLOTS = 10
DOT_PROD_TOLERANCE_LOCAL = 1e-6

# 球体参数 (参照 Figure_1.txt)
SPHERE_OBJECT_RADIUS = 25.0 
SPHERE_OBJECT_CENTER_OFFSET_Z = SPHERE_OBJECT_RADIUS + 15.0 
SPHERE_OBJECT_CENTER_WORLD = np.array([0.0, 0.0, SPHERE_OBJECT_CENTER_OFFSET_Z]) 

CHARACTERISTIC_LENGTH_FOR_GII = SPHERE_OBJECT_RADIUS 

COST_MESH_FAILURE = 9.0
COST_OVERLAP = 5.0
COST_INTERSECTION = 4.0
COST_MAX_PRESSURE_NO_CONTACT = 6.0
COST_NO_CONTACT_OR_ITER_LIMIT = 3.0
COST_LOW_GII_OR_FEW_CONTACTS = 2.0

# 使用 itertools.combinations 生成规范化的组合索引
all_valid_finger_combinations_global = list(itertools.combinations(range(N_FINGER_SLOTS), 3))
num_valid_combinations = len(all_valid_finger_combinations_global)
print(f"总共有 {N_FINGER_SLOTS} 个手指槽位，生成了 {num_valid_combinations} 种唯一的（无序）三手指位置组合。")

# --- 遗传算法参数 ---
GA_POPULATION_SIZE = 8
GA_NUM_GENERATIONS = 10 # 总迭代代数
GA_CX_PROB = 0.7  # 交叉概率
GA_MUT_PROB = 0.2 # 变异概率
GA_ELITISM_COUNT = 2 # 精英保留数量

# --- 5. 辅助函数 ---
def o3d_create_transformation_matrix(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t.flatten(); return T
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
    if points is None: return None; points = np.asarray(points)
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
            torque = np.cross(r_contact, d_force)
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
def get_rotation_matrix_between_vectors(vec1, vec2):
    a = vec1 / np.linalg.norm(vec1); b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b); c = np.dot(a, b); s = np.linalg.norm(v)
    if s < 1e-9: return np.identity(3) if c > 0 else create_rotation_matrix(np.array([1.0,0,0]) if np.abs(a[0])<0.9 else np.array([0,1.0,0]), np.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r_mat = np.identity(3) + vx + vx @ vx * ((1 - c) / (s ** 2)); return r_mat

def create_sphere_point_cloud(radius, center, n_points=500):
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20) 
    pcd = sphere_mesh.sample_points_uniformly(number_of_points=n_points)
    pcd.translate(center, relative=False) 
    return np.asarray(pcd.points)

def visualize_poses_with_open3d(tray_geometry_transform_mm, tray_axes_transform_mm, bottle_axes_transform_mm, bottle_points_world_mm, bottle_colors_rgb_float, bottle_obb_world, tray_radius_mm, tray_height_mm, window_title="Open3D Relative Pose Visualization"):
    print(f"\n--- 打开 Open3D 窗口: {window_title} ---")
    geometries = []
    o3d_tray_canonical_centered = o3d.geometry.TriangleMesh.create_cylinder(radius=tray_radius_mm, height=tray_height_mm, resolution=20, split=4)
    o3d_tray_canonical_centered.translate(np.array([0,0,-tray_height_mm/2.0]))
    o3d_tray_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_tray_canonical_centered)
    o3d_tray_wireframe.transform(tray_geometry_transform_mm)
    o3d_tray_wireframe.paint_uniform_color(pv.Color(tray_color).float_rgb)
    geometries.append(o3d_tray_wireframe)
    o3d_bottle_pcd = o3d.geometry.PointCloud()
    o3d_bottle_pcd.points = o3d.utility.Vector3dVector(bottle_points_world_mm)
    if bottle_colors_rgb_float is not None and len(bottle_colors_rgb_float) == len(bottle_points_world_mm):
        o3d_bottle_pcd.colors = o3d.utility.Vector3dVector(bottle_colors_rgb_float)
    else: o3d_bottle_pcd.paint_uniform_color(pv.Color(object_point_color).float_rgb)
    geometries.append(o3d_bottle_pcd)
    if bottle_obb_world is not None: geometries.append(bottle_obb_world)
    tray_axes_size = tray_radius_mm * 0.7
    o3d_tray_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=tray_axes_size, origin=[0,0,0])
    o3d_tray_axes.transform(tray_axes_transform_mm)
    geometries.append(o3d_tray_axes)
    if bottle_points_world_mm.shape[0] > 0:
        temp_bottle_pcd_for_bbox = o3d.geometry.PointCloud(); temp_bottle_pcd_for_bbox.points = o3d.utility.Vector3dVector(bottle_points_world_mm)
        bbox_bottle = temp_bottle_pcd_for_bbox.get_axis_aligned_bounding_box()
        diag_len = LA.norm(bbox_bottle.get_max_bound() - bbox_bottle.get_min_bound())
        bottle_axes_size = diag_len * 0.25 if diag_len > 1e-2 else tray_axes_size * 0.8
    else: bottle_axes_size = tray_axes_size * 0.8
    bottle_axes_size = max(bottle_axes_size, 5.0)
    o3d_bottle_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=bottle_axes_size, origin=[0,0,0])
    o3d_bottle_axes.transform(bottle_axes_transform_mm)
    geometries.append(o3d_bottle_axes)
    world_axes_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=tray_radius_mm * 1.2, origin=[0,0,0])
    geometries.append(world_axes_o3d)
    o3d.visualization.draw_geometries(geometries, window_name=window_title, width=800, height=600)
    print(f"--- 关闭 Open3D 窗口: {window_title} ---")

# 全局变量定义
initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global = [None]*5
object_points_global_static, object_centroid_global_static, object_mesh_global_static = [None]*3
world_obb_object_global = None; pv_obb_object_mesh_global = None
num_object_points_global_static, faces_np_global, width_translation_vector_global = 0, None, None
T1_translate_global, T2_rotate_global = None, None
T_pose_for_tray_display_and_finger_placement_global = None 
tray_pv_global = None 

# 新增：调试可视化函数
def visualize_failed_ga_step(r_param, chosen_indices_param, pressures_param, finger_meshes_param, failure_reason_str, generation, individual_idx):
    global DEBUG_VISUALIZE_FAILED_GRASPS, FAILED_GRASP_VIS_COUNT, MAX_FAILED_GRASP_VIS
    global object_mesh_global_static, tray_pv_global, pv_obb_object_mesh_global, show_axes
    
    if not DEBUG_VISUALIZE_FAILED_GRASPS or FAILED_GRASP_VIS_COUNT >= MAX_FAILED_GRASP_VIS:
        if FAILED_GRASP_VIS_COUNT == MAX_FAILED_GRASP_VIS and DEBUG_VISUALIZE_FAILED_GRASPS:
            print("已达到最大失败可视化次数，后续失败将不再显示。")
            FAILED_GRASP_VIS_COUNT +=1 
        return

    FAILED_GRASP_VIS_COUNT += 1
    
    plotter_title = f"GA失败调试 Gen {generation} Ind {individual_idx} #{FAILED_GRASP_VIS_COUNT}: {failure_reason_str}"
    plotter_debug = pv.Plotter(window_size=[900,700], title=plotter_title)

    if tray_pv_global:
        plotter_debug.add_mesh(tray_pv_global, color=tray_color, opacity=0.5, name='tray_debug')
    if object_mesh_global_static: 
        plotter_debug.add_mesh(object_mesh_global_static, color=object_point_color, style='surface', opacity=0.6, name='obj_debug_sphere')
    if pv_obb_object_mesh_global: 
        plotter_debug.add_mesh(pv_obb_object_mesh_global, style='wireframe', color=object_obb_color_pv, line_width=2, name='obj_obb_debug')
    if show_axes:
        plotter_debug.add_axes_at_origin(labels_off=False, line_width=3)

    if finger_meshes_param:
        for i, mesh in enumerate(finger_meshes_param):
            if mesh:
                plotter_debug.add_mesh(mesh, color=finger_color, style='surface', opacity=0.75, name=f'finger_debug_{i}')
            else:
                print(f"  调试可视化警告: 手指 {i} 的网格为 None。")

    params_text = f"r={r_param:.3f}, 位置=({chosen_indices_param[0]},{chosen_indices_param[1]},{chosen_indices_param[2]})\n"
    pressures_text = f"压力=[{pressures_param[0]:.0f}, {pressures_param[1]:.0f}, {pressures_param[2]:.0f}]\n"
    reason_text = f"原因: {failure_reason_str}"
    plotter_debug.add_text(params_text + pressures_text + reason_text, position="upper_left", font_size=10, color='white')
    
    plotter_debug.camera_position = 'xy'
    print(f"\n(PyVista) 显示GA失败步骤 Gen {generation} Ind {individual_idx} #{FAILED_GRASP_VIS_COUNT}。原因: {failure_reason_str}。请关闭窗口以继续...")
    plotter_debug.show(cpos='xy') 
    plotter_debug.close()


def evaluate_grasp(r, pos_idx1, pos_idx2, pos_idx3, generation_info="N/A"): # 添加 generation_info 用于调试
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
                    gen_num, ind_idx = generation_info.split("-") if isinstance(generation_info, str) and "-" in generation_info else ("N/A", "N/A")
                    visualize_failed_ga_step(r, chosen_indices, current_pressures, deformed_finger_meshes_at_contact, failure_reason_for_vis, gen_num, ind_idx)
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
                    gen_num, ind_idx = generation_info.split("-") if isinstance(generation_info, str) and "-" in generation_info else ("N/A", "N/A")
                    visualize_failed_ga_step(r, chosen_indices, current_pressures, deformed_finger_meshes_at_contact, failure_reason_for_vis, gen_num, ind_idx)
                return (cost_to_return, current_pressures, deformed_finger_meshes_at_contact, None)

            for i_mesh_update in range(3):
                if current_step_finger_meshes[i_mesh_update] is not None:
                     deformed_finger_meshes_at_contact[i_mesh_update] = current_step_finger_meshes[i_mesh_update].copy()

            has_overlap_this_iter = False
            contact_made_by_finger_this_iter_flags = [False] * 3
            finger_dot_products_this_iter = [[] for _ in range(3)] 
            
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
                        gen_num, ind_idx = generation_info.split("-") if isinstance(generation_info, str) and "-" in generation_info else ("N/A", "N/A")
                        visualize_failed_ga_step(r, chosen_indices, current_pressures, current_step_finger_meshes, failure_reason_for_vis, gen_num, ind_idx)
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
                    gen_num, ind_idx = generation_info.split("-") if isinstance(generation_info, str) and "-" in generation_info else ("N/A", "N/A")
                    visualize_failed_ga_step(r, chosen_indices, current_pressures, current_step_finger_meshes, failure_reason_for_vis, gen_num, ind_idx)
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
                            gen_num, ind_idx = generation_info.split("-") if isinstance(generation_info, str) and "-" in generation_info else ("N/A", "N/A")
                            visualize_failed_ga_step(r, chosen_indices, return_pressures, return_meshes, failure_reason_for_vis, gen_num, ind_idx)
                        return (cost_to_return, return_pressures, return_meshes, return_contacts) 
                else: 
                    cost_to_return = COST_LOW_GII_OR_FEW_CONTACTS
                    failure_reason_for_vis = "接触点不足 GII"
                    print(f"  GA评估 ({generation_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                    if DEBUG_VISUALIZE_FAILED_GRASPS:
                        gen_num, ind_idx = generation_info.split("-") if isinstance(generation_info, str) and "-" in generation_info else ("N/A", "N/A")
                        visualize_failed_ga_step(r, chosen_indices, return_pressures, return_meshes, failure_reason_for_vis, gen_num, ind_idx)
                    return (cost_to_return, return_pressures, return_meshes, return_contacts) 

            for i in range(3):
                if not finger_contact_established[i] and current_pressures[i] >= max_pressure:
                    cost_to_return = COST_MAX_PRESSURE_NO_CONTACT
                    failure_reason_for_vis = f"手指 {i} 最大压力无接触"
                    print(f"  GA评估 ({generation_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                    if DEBUG_VISUALIZE_FAILED_GRASPS:
                        gen_num, ind_idx = generation_info.split("-") if isinstance(generation_info, str) and "-" in generation_info else ("N/A", "N/A")
                        visualize_failed_ga_step(r, chosen_indices, current_pressures, deformed_finger_meshes_at_contact, failure_reason_for_vis, gen_num, ind_idx)
                    return (cost_to_return, current_pressures, deformed_finger_meshes_at_contact, None) 
            
            if not pressure_changed_this_step and not all(finger_contact_established):
                cost_to_return = COST_MAX_PRESSURE_NO_CONTACT 
                failure_reason_for_vis = "停滞无接触"
                print(f"  GA评估 ({generation_info}): {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                if DEBUG_VISUALIZE_FAILED_GRASPS:
                    gen_num, ind_idx = generation_info.split("-") if isinstance(generation_info, str) and "-" in generation_info else ("N/A", "N/A")
                    visualize_failed_ga_step(r, chosen_indices, current_pressures, deformed_finger_meshes_at_contact, failure_reason_for_vis, gen_num, ind_idx)
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
            gen_num, ind_idx = generation_info.split("-") if isinstance(generation_info, str) and "-" in generation_info else ("N/A", "N/A")
            visualize_failed_ga_step(r, chosen_indices, current_pressures_fallback, 
                                     deformed_meshes_fallback, 
                                     failure_reason_for_vis, gen_num, ind_idx)
        return (cost_to_return, None, None, None)


# --- 主脚本 ---
if __name__ == '__main__':
    # --- A. 定义场景和物体 ---
    T_pose_for_tray_display_and_finger_placement_global = np.identity(4) 
    T_actual_tray_geometry_world = np.identity(4) 
    T_tray_axes_vis_world = np.identity(4) 

    object_points_global_static = create_sphere_point_cloud(SPHERE_OBJECT_RADIUS, SPHERE_OBJECT_CENTER_WORLD, n_points=TARGET_POINT_COUNT_FOR_SIM)
    if object_points_global_static is None or object_points_global_static.shape[0] == 0: 
        sys.exit("错误: 未能生成球形物体点云")
    
    object_centroid_global_static = np.mean(object_points_global_static, axis=0)
    num_object_points_global_static = object_points_global_static.shape[0]
    print(f"已生成球形物体，中心: {object_centroid_global_static.round(2)}, 点数: {num_object_points_global_static}")

    object_mesh_global_static = pv.Sphere(radius=SPHERE_OBJECT_RADIUS, center=SPHERE_OBJECT_CENTER_WORLD, theta_resolution=30, phi_resolution=30)
    
    world_obb_object_global = o3d.geometry.OrientedBoundingBox()
    world_obb_object_global.center = SPHERE_OBJECT_CENTER_WORLD
    world_obb_object_global.R = np.identity(3)
    world_obb_object_global.extent = np.array([2*SPHERE_OBJECT_RADIUS, 2*SPHERE_OBJECT_RADIUS, 2*SPHERE_OBJECT_RADIUS])
    world_obb_object_global.color = object_obb_color_o3d

    pv_obb_object_mesh_global = pv.Cube(center=SPHERE_OBJECT_CENTER_WORLD, 
                                        x_length=2*SPHERE_OBJECT_RADIUS, 
                                        y_length=2*SPHERE_OBJECT_RADIUS, 
                                        z_length=2*SPHERE_OBJECT_RADIUS)
    
    T_object_axes_vis_world = create_transformation_matrix_opt8(np.identity(3), SPHERE_OBJECT_CENTER_WORLD)

    # --- B. 初始化模型和手指参考 ---
    initial_coords_ref_global = load_initial_coordinates(INITIAL_COORDS_PATH, NODE_COUNT)
    model_global,scaler_X_global,scaler_y_global,device_global = load_prediction_components(MODEL_PATH,X_SCALER_PATH,Y_SCALER_PATH,INPUT_DIM,OUTPUT_DIM,HIDDEN_LAYER_1,HIDDEN_LAYER_2,HIDDEN_LAYER_3)
    if initial_coords_ref_global is None or model_global is None: sys.exit("错误：未能初始化手指模型或坐标。")
    faces_np_global = create_faces_array(NODE_COUNT) 
    if faces_np_global is None or faces_np_global.size == 0: sys.exit("错误：未能创建手指表面。")
    width_translation_vector_global = np.array([0, finger_width, 0]) 
    bottom_node_idx = np.argmin(initial_coords_ref_global[:,0]) 
    ref_mid_pt = initial_coords_ref_global[bottom_node_idx] + width_translation_vector_global/2.0 
    T1_translate_global = create_transformation_matrix_opt8(None,-ref_mid_pt) 
    rot_ref_to_local = np.array([[0,1,0],[0,0,1],[1,0,0]]) 
    T2_rotate_global = create_transformation_matrix_opt8(rot_ref_to_local,None) 

    tray_pv_center_for_vis = np.array([0,0, -tray_height/2.0]) 
    tray_pv_global = pv.Cylinder(center=tray_pv_center_for_vis, direction=(0,0,1), 
                                 radius=tray_radius, height=tray_height, resolution=30)


    # --- C. 遗传算法 ---
    print(f"\n开始遗传算法优化 (群体大小: {GA_POPULATION_SIZE}, 代数: {GA_NUM_GENERATIONS})")
    print(f"目标物体: 球体, 半径={SPHERE_OBJECT_RADIUS:.2f}, 中心={SPHERE_OBJECT_CENTER_WORLD}")
    if DEBUG_VISUALIZE_FAILED_GRASPS:
        print(f"失败步骤可视化已启用。最多显示 {MAX_FAILED_GRASP_VIS} 次。")

    # 1. 初始化群体
    population = []
    for _ in range(GA_POPULATION_SIZE):
        r_gene = random.uniform(R_BOUNDS[0], R_BOUNDS[1])
        combo_idx_gene = random.randint(0, num_valid_combinations - 1)
        population.append([r_gene, combo_idx_gene]) # 染色体: [r, combo_idx]

    best_overall_gii = -float('inf') 
    best_overall_chromosome = None
    best_overall_pressures = None
    best_overall_meshes = None
    best_overall_contacts = None

    for gen in range(GA_NUM_GENERATIONS):
        print(f"\n--- 第 {gen + 1}/{GA_NUM_GENERATIONS} 代 ---")
        
        # 2. 适应度评估
        fitness_values = []
        evaluated_population_details = [] # 存储评估结果以供选择

        for i, chromosome in enumerate(population):
            r_val = chromosome[0]
            combo_idx_val = int(chromosome[1])
            
            # 解码 combo_idx
            pos_indices = all_valid_finger_combinations_global[combo_idx_val]
            pos_idx1, pos_idx2, pos_idx3 = pos_indices[0], pos_indices[1], pos_indices[2]
            
            generation_debug_info = f"{gen+1}-{i+1}" # 代数-个体编号
            cost, pressures, meshes, contacts = evaluate_grasp(r_val, pos_idx1, pos_idx2, pos_idx3, generation_info=generation_debug_info)
            
            fitness = -cost # 因为 evaluate_grasp 返回成本（越小越好，负的GII更好）
            fitness_values.append(fitness)
            evaluated_population_details.append({
                "chromosome": chromosome,
                "fitness": fitness,
                "cost": cost,
                "pressures": pressures,
                "meshes": meshes,
                "contacts": contacts,
                "r": r_val,
                "pos_indices": pos_indices
            })

            if fitness > best_overall_gii:
                best_overall_gii = fitness
                best_overall_chromosome = chromosome
                best_overall_pressures = pressures
                best_overall_meshes = meshes
                best_overall_contacts = contacts
                print(f"  ** 新的最优解! Gen {gen+1}, Ind {i+1}, GII = {-best_overall_gii:.4f}, r={r_val:.3f}, pos={pos_indices} **")

        # 3. 选择 (轮盘赌选择)
        total_fitness = sum(f if f > -float('inf') else 0 for f in fitness_values) # 处理可能的-inf
        # 如果所有适应度都非常差（例如，都是高成本导致负适应度很大），调整以进行选择
        min_fitness = min(fitness_values)
        adjusted_fitness_values = [f - min_fitness + 1e-6 for f in fitness_values] #确保非负且有区分
        total_adjusted_fitness = sum(adjusted_fitness_values)

        new_population = []
        
        # 精英保留
        sorted_evaluated_population = sorted(evaluated_population_details, key=lambda x: x["fitness"], reverse=True)
        for i in range(GA_ELITISM_COUNT):
            if i < len(sorted_evaluated_population):
                new_population.append(list(sorted_evaluated_population[i]["chromosome"])) #确保是列表副本

        # 3. 选择 (轮盘赌选择)
        total_fitness = sum(f if f > -float('inf') else 0 for f in fitness_values) # 处理可能的-inf
        # 如果所有适应度都非常差（例如，都是高成本导致负适应度很大），调整以进行选择
        min_fitness = min(fitness_values) if fitness_values else 0 # 处理 fitness_values 为空的情况
        adjusted_fitness_values = [f - min_fitness + 1e-6 for f in fitness_values] #确保非负且有区分
        total_adjusted_fitness = sum(adjusted_fitness_values)

        new_population = []
        
        # 精英保留
        # sorted_evaluated_population 已经包含了适应度和染色体等信息
        sorted_evaluated_population = sorted(evaluated_population_details, key=lambda x: x["fitness"], reverse=True)
        for i in range(GA_ELITISM_COUNT):
            if i < len(sorted_evaluated_population):
                new_population.append(list(sorted_evaluated_population[i]["chromosome"])) 

        # 轮盘赌选择剩余个体
        num_to_select = GA_POPULATION_SIZE - GA_ELITISM_COUNT
        
        # 创建一个包含染色体和其对应调整后适应度的列表，用于轮盘赌选择
        # 确保这里的顺序与 evaluated_population_details 一致，以便正确匹配
        # 或者直接使用 evaluated_population_details 中的 "fitness" 来计算 adjusted_fitness
        
        # 为了安全，我们直接从 evaluated_population_details 中按顺序取 adjusted_fitness
        # 假设 fitness_values 和 evaluated_population_details 的顺序是一致的
        
        population_for_selection = []
        for i, details_item in enumerate(evaluated_population_details):
            population_for_selection.append({
                "chromosome": details_item["chromosome"],
                "adjusted_fitness": adjusted_fitness_values[i] # 使用对应索引的调整后适应度
            })

        for _ in range(num_to_select):
            if total_adjusted_fitness <= 0: 
                # 如果总调整适应度为0或负（不太可能，因为加了1e-6），随机选择
                pick_details = random.choice(evaluated_population_details)
            else:
                r_pick = random.uniform(0, total_adjusted_fitness)
                current_sum = 0
                # 在 population_for_selection 上进行轮盘赌
                for individual_details in population_for_selection:
                    current_sum += individual_details["adjusted_fitness"]
                    if current_sum > r_pick:
                        pick_details = individual_details # pick_details 现在是一个包含 "chromosome" 和 "adjusted_fitness" 的字典
                        break
                else: 
                    pick_details = random.choice(population_for_selection) # Fallback
            new_population.append(list(pick_details["chromosome"]))

        population = new_population

        # 4. 交叉
        offspring_population = []
        for i in range(0, GA_POPULATION_SIZE - GA_ELITISM_COUNT, 2): # 从非精英部分选择配对
            if i + 1 >= len(population): # 如果是奇数个非精英，最后一个直接加入
                offspring_population.append(population[i])
                continue

            parent1 = population[i]
            parent2 = population[i+1]
            
            child1, child2 = list(parent1), list(parent2) # 创建副本

            if random.random() < GA_CX_PROB:
                # r 基因交叉 (算术交叉)
                alpha = random.random()
                child1[0] = alpha * parent1[0] + (1 - alpha) * parent2[0]
                child2[0] = (1 - alpha) * parent1[0] + alpha * parent2[0]
                child1[0] = np.clip(child1[0], R_BOUNDS[0], R_BOUNDS[1])
                child2[0] = np.clip(child2[0], R_BOUNDS[0], R_BOUNDS[1])

                # combo_idx 基因交叉 (单点交叉)
                if random.random() < 0.5: # 交换 combo_idx
                    child1[1], child2[1] = parent2[1], parent1[1]
            
            offspring_population.extend([child1, child2])
        
        # 如果群体大小为奇数，且精英数量也为奇数，确保offspring_population填满
        if len(offspring_population) < GA_POPULATION_SIZE - GA_ELITISM_COUNT:
            if population: # 确保population非空
                 offspring_population.append(random.choice(population))


        # 5. 变异
        for i in range(len(offspring_population)):
            if random.random() < GA_MUT_PROB:
                # r 基因变异 (高斯扰动)
                offspring_population[i][0] += random.gauss(0, (R_BOUNDS[1] - R_BOUNDS[0]) * 0.1) # 标准差为范围的10%
                offspring_population[i][0] = np.clip(offspring_population[i][0], R_BOUNDS[0], R_BOUNDS[1])
            
            if random.random() < GA_MUT_PROB:
                # combo_idx 基因变异 (随机重置)
                offspring_population[i][1] = random.randint(0, num_valid_combinations - 1)
        
        # 形成新一代群体 (精英 + 子代)
        population = population[:GA_ELITISM_COUNT] + offspring_population
        # 确保种群大小正确
        if len(population) > GA_POPULATION_SIZE:
            population = population[:GA_POPULATION_SIZE]
        elif len(population) < GA_POPULATION_SIZE: # 如果因为奇偶数问题导致数量不足
            while len(population) < GA_POPULATION_SIZE:
                population.append(list(random.choice(sorted_evaluated_population[:max(1, GA_ELITISM_COUNT)])["chromosome"]))


        current_best_gii_gen = max(fitness_values)
        print(f"  代 {gen + 1} 最佳 GII: {-current_best_gii_gen:.4f} (成本: {current_best_gii_gen:.4f})")
        print(f"  迄今为止最优 GII: {-best_overall_gii:.4f}")


    print("\n遗传算法优化结束。")

    # --- D. 处理和可视化优化结果 ---
    if best_overall_chromosome is not None:
        best_r_ga = best_overall_chromosome[0]
        best_combo_idx_ga = int(best_overall_chromosome[1])
        best_pos_indices_ga = all_valid_finger_combinations_global[best_combo_idx_ga]
        
        print("\n找到的最优参数 (GA):")
        print(f"  r          = {best_r_ga:.4f}")
        print(f"  pos_idx1   = {best_pos_indices_ga[0]}")
        print(f"  pos_idx2   = {best_pos_indices_ga[1]}")
        print(f"  pos_idx3   = {best_pos_indices_ga[2]}")
        print(f"  (combo_idx = {best_combo_idx_ga})") 
        print(f"最优 GII (GA): {-best_overall_gii:.4f} (成本: {best_overall_gii:.4f})")

        print("\n使用最优参数生成最终可视化...")
        # final_pressures_viz, final_meshes_viz, final_contacts_viz 来自存储的 best_overall_...
        final_pressures_viz = best_overall_pressures
        final_meshes_viz = best_overall_meshes
        # final_contacts_viz = best_overall_contacts # 如果需要，可以用于更详细的接触点可视化

        if final_meshes_viz and len(final_meshes_viz) == 3 and all(m is not None for m in final_meshes_viz):
            final_chosen_indices_ga_viz = best_pos_indices_ga
            
            pressures_display_str = "N/A"
            if final_pressures_viz is not None and hasattr(final_pressures_viz, 'size') and final_pressures_viz.size == 3:
                pressures_display_str = f"[{final_pressures_viz[0]:.0f}, {final_pressures_viz[1]:.0f}, {final_pressures_viz[2]:.0f}]"

            state_disp_ga=f"Cost={best_overall_gii:.3f}"; gii_disp_ga=f"GII={-best_overall_gii:.3f}" if best_overall_gii < -1e-9 else "GII:N/A"
            plotter_final_pv = pv.Plotter(window_size=[1000,800],title="PyVista - Optimal Grasp (GA - Sphere)", off_screen=False)
            plotter_final_pv.add_mesh(tray_pv_global,color=tray_color,opacity=0.5,name='tray_final_pv_oriented')
            
            plotter_final_pv.add_mesh(object_mesh_global_static, color=object_point_color, style='surface', opacity=0.6, name='obj_final_pv_sphere')
            
            if pv_obb_object_mesh_global is not None: 
                plotter_final_pv.add_mesh(pv_obb_object_mesh_global, style='wireframe', color=object_obb_color_pv, line_width=2, name='object_obb_final_pv')
            if show_axes: plotter_final_pv.add_axes_at_origin(labels_off=False,line_width=3) 
            
            for i_fv_ga,m_fv_ga in enumerate(final_meshes_viz):
                if m_fv_ga: plotter_final_pv.add_mesh(m_fv_ga,color=finger_color,style='surface',opacity=0.85,name=f'f_final_pv_{i_fv_ga}')
            
            params_txt_ga=f"r={best_r_ga:.2f}, P={pressures_display_str}, indices=({final_chosen_indices_ga_viz[0]},{final_chosen_indices_ga_viz[1]},{final_chosen_indices_ga_viz[2]})"
            plotter_final_pv.add_text(f"Optimal (GA):{params_txt_ga}\n{state_disp_ga}\n{gii_disp_ga}",position="upper_left",font_size=10,color='white')
            plotter_final_pv.camera_position='xy'
            print("\n(PyVista) 显示最优抓取配置。请关闭窗口以继续。")
            plotter_final_pv.show(cpos='xy') 
            plotter_final_pv.close()

            visualize_poses_with_open3d(
                tray_geometry_transform_mm=T_actual_tray_geometry_world, 
                tray_axes_transform_mm=T_tray_axes_vis_world,          
                bottle_axes_transform_mm=T_object_axes_vis_world,       
                bottle_points_world_mm=object_points_global_static,     
                bottle_colors_rgb_float=None, 
                bottle_obb_world=world_obb_object_global,
                tray_radius_mm=tray_radius,
                tray_height_mm=tray_height,
                window_title="Open3D - Optimal Poses & Axes (GA - Sphere)"
            )
        else:
            print("未能为最优参数生成手指网格或获取最终压力，无法进行PyVista最终可视化。")
            if best_overall_gii is not None : print(f"  (找到的最优成本为: {best_overall_gii:.3f})")

    else:
        print("\nGA未成功或无结果，无法显示最优。")
    print("\n程序结束。")
