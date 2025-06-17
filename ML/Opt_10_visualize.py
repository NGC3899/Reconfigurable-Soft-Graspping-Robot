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
import vtk # For VTK font constants

# --- 打印版本信息 ---
try:
    print(f"Opt_10_enhanced_visualization.txt (PyVista Interaction & O3D Tray Wireframe) - OBJECT LONGEST EDGE ALIGNED TO LOCAL X, N_SLOTS FINGER POSITIONS")
    print(f"Open3D version: {o3d.__version__}")
    print(f"PyVista version: {pv.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Torch version: {torch.__version__}")
except NameError:
    pass
except Exception as e:
    print(f"Error printing library versions: {e}")

# --- 导入贝叶斯优化库 ---
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer # Added Integer
    from skopt.utils import use_named_args
except ImportError:
    print("错误：需要安装 scikit-optimize 库。请运行 'pip install scikit-optimize'")
    sys.exit()

# --- 1. ML 模型定义 (来自 Opt_9.txt 源文件) ---
class MLPRegression(nn.Module):
    def __init__(self, input_dim, output_dim, h1, h2, h3):
        super(MLPRegression, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, h3), nn.Tanh(),
            nn.Linear(h3, output_dim) )
    def forward(self, x): return self.network(x)

# --- 2. ML 相关参数定义 (来自 Opt_9.txt 源文件) ---
INPUT_DIM = 1
NODE_COUNT = 63
OUTPUT_DIM = NODE_COUNT * 3
HIDDEN_LAYER_1 = 128
HIDDEN_LAYER_2 = 256
HIDDEN_LAYER_3 = 128

# --- 3. 文件路径定义 (来自 Opt_9.txt 源文件) ---
MODEL_PATH = 'best_mlp_model.pth'
X_SCALER_PATH = 'x_scaler.joblib'
Y_SCALER_PATH = 'y_scaler.joblib'
INITIAL_COORDS_PATH = 'initial_coordinates.txt'

GRASP_OUTPUTS_BASE_PATH = r"C:\Users\admin\Desktop\grasp_outputs" # <--- 修改为您的路径
RELATIVE_POSE_FILENAME = "relative_gripper_to_object_pose_CONDITIONAL_REORIENT.txt"
# RELATIVE_POSE_FILENAME = "relative_gripper_to_object_pose.txt"

HIGH_RES_OBJECT_DESKTOP_PATH = r"C:\Users\admin\Desktop" # <--- 修改为您的路径
# HIGH_RES_OBJECT_FILENAME = "Graphene_Bottle.ply"
HIGH_RES_OBJECT_FILENAME = "Bird_Model.ply" # Make sure this file exists

RELATIVE_POSE_FILE_PATH = os.path.join(GRASP_OUTPUTS_BASE_PATH, RELATIVE_POSE_FILENAME)
HIGH_RES_OBJECT_PLY_PATH = os.path.join(HIGH_RES_OBJECT_DESKTOP_PATH, HIGH_RES_OBJECT_FILENAME)

# --- 4. 配置参数 (来自 Opt_9.txt 源文件) ---
tray_radius = 60.0
tray_height = 1.0
tray_center = np.array([0.0, 0.0, -tray_height / 2.0]) # Original tray center definition
finger_width = 10.0
TARGET_POINT_COUNT_FOR_SIM = 1000
show_axes = True # This controls plotter.add_axes_at_origin()
OBJECT_SCALE_FACTOR = 0.8

FINGER_VIZ_THICKNESS = 1.5 

# --- 美化参数 ---
finger_color_viz = '#ff7f0e'  # Matplotlib Orange (更鲜明)
tray_color_viz = '#BDB7A4'    # Desaturated Tan/Khaki
object_point_color_viz = '#1f77b4' # Matplotlib Blue
object_obb_color_viz = '#2ca02c'   # Matplotlib Green
background_color_viz = '#EAEAEA' # Light Gray
text_color_viz = 'black'
font_family_viz = 'times'
# --- end 美化参数 ---

colliding_point_color = 'magenta'
intersection_point_color = 'yellow'
overlap_point_color = 'orange'
collision_threshold = 1.0
overlap_threshold = 1e-4
friction_coefficient = 0.5
eigenvalue_threshold = 1e-6
max_pressure = 40000.0
NUM_INITIAL_SEARCHES = 1
PRESSURE_STEP_INIT_SEARCH = 300.0
N_CALLS_BO = 200
N_INITIAL_POINTS_BO = 2
R_BOUNDS = (tray_radius * 0.3, tray_radius * 0.95)
P_BOUNDS = (0.0, max_pressure)
N_FINGER_SLOTS = 9
contact_marker_radius = 1.0
contact_normal_length = 5.0
contact_plane_size = 4.0
show_finger_normals = True # Not currently visualized in the main plotter
finger_normal_vis_scale = 1.0
finger_normal_color = 'gray'
contact_normal_color = 'black'
OBJECT_SCALE_FACTOR = 0.8

param_names = ['r', 'p1', 'p2', 'p3', 'pos_idx1', 'pos_idx2', 'pos_idx3']
dimensions = [
    Real(name='r', low=R_BOUNDS[0], high=R_BOUNDS[1]),
    Real(name='p1', low=P_BOUNDS[0], high=P_BOUNDS[1]),
    Real(name='p2', low=P_BOUNDS[0], high=P_BOUNDS[1]),
    Real(name='p3', low=P_BOUNDS[0], high=P_BOUNDS[1]),
    Integer(name='pos_idx1', low=0, high=N_FINGER_SLOTS - 1),
    Integer(name='pos_idx2', low=0, high=N_FINGER_SLOTS - 1),
    Integer(name='pos_idx3', low=0, high=N_FINGER_SLOTS - 1)
]

# --- 5. 辅助函数 (保持不变) ---
def o3d_create_transformation_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def create_rotation_matrix(axis, angle_rad):
    axis = np.asarray(axis).astype(float)
    axis_norm = LA.norm(axis)
    if axis_norm < 1e-9 : return np.identity(3)
    axis /= axis_norm
    a = np.cos(angle_rad / 2.0)
    b, c, d = -axis * np.sin(angle_rad / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

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
    num_pts = points.shape[0]
    h_points = np.hstack((points, np.ones((num_pts, 1))))
    t_h = h_points @ matrix.T
    w = t_h[:, 3, np.newaxis]
    return np.divide(t_h[:, :3], w, out=np.zeros_like(t_h[:, :3]), where=w!=0)

def load_transformation_matrix_from_txt(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 变换矩阵文件 '{file_path}' 未找到.")
        return None
    try:
        matrix = np.loadtxt(file_path)
        if matrix.shape == (4, 4):
            print(f"成功从 '{file_path}' 加载变换矩阵.")
            return matrix
        else:
            print(f"错误: 从 '{file_path}' 加载的矩阵形状不是 (4, 4)，而是 {matrix.shape}.")
            return None
    except Exception as e:
        print(f"加载变换矩阵 '{file_path}' 时出错: {e}")
        return None

def load_initial_coordinates(file_path, expected_nodes):
    try:
        coords = np.loadtxt(file_path, dtype=np.float32, usecols=(0, 1, 2))
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"加载初始坐标时出错: {e}")
        return None
    if coords.shape == (expected_nodes, 3):
        print(f"成功加载 {coords.shape[0]} 个初始节点坐标。")
        return coords
    else:
        print(f"错误：坐标形状 {coords.shape} 与预期 ({expected_nodes},3) 不符。")
        return None

def create_faces_array(num_nodes_per_curve):
    faces = []
    num_quads = num_nodes_per_curve - 1
    if num_quads <= 0: return np.array([], dtype=int)
    for i in range(num_quads):
        p1,p2=i,i+1
        p3,p4=(i+1)+num_nodes_per_curve,i+num_nodes_per_curve
        faces.append([4,p1,p2,p3,p4])
    return np.hstack(faces)

def sort_points_spatially(points):
    if points is None: return None
    points = np.asarray(points)
    if points.shape[0] < 2: return points
    num_points = points.shape[0]
    sorted_indices = []
    remaining_indices = list(range(num_points))
    start_node_index = np.argmin(points[:,0]+points[:,1]+points[:,2])
    current_index = start_node_index
    sorted_indices.append(current_index)
    if current_index in remaining_indices: remaining_indices.pop(remaining_indices.index(current_index))
    while remaining_indices:
        last_point = points[current_index,np.newaxis]
        remaining_points_array = points[remaining_indices]
        if remaining_points_array.ndim == 1: remaining_points_array = remaining_points_array[np.newaxis,:]
        if remaining_points_array.shape[0] == 0: break
        try:
            distances = cdist(last_point,remaining_points_array)[0]
        except Exception as e_cdist:
            print(f"Error during cdist: {e_cdist}")
            break
        if distances.size == 0: break
        nearest_neighbor_relative_index = np.argmin(distances)
        nearest_neighbor_absolute_index = remaining_indices[nearest_neighbor_relative_index]
        sorted_indices.append(nearest_neighbor_absolute_index)
        current_index = nearest_neighbor_absolute_index
        if nearest_neighbor_absolute_index in remaining_indices:
            remaining_indices.pop(remaining_indices.index(nearest_neighbor_absolute_index))
    if len(sorted_indices) != num_points:
        print(f"Warning: Spatial sort only processed {len(sorted_indices)} of {num_points} points.")
    return points[sorted_indices]

def get_orthogonal_vectors(normal_vector):
    n = np.asarray(normal_vector).astype(float)
    norm_n = LA.norm(n)
    if norm_n < 1e-9: raise ValueError("Normal vector zero.")
    n /= norm_n
    if np.abs(n[0]) > 0.9: v_arbitrary = np.array([0., 1., 0.])
    else: v_arbitrary = np.array([1., 0., 0.])
    t1 = np.cross(n, v_arbitrary)
    norm_t1 = LA.norm(t1)
    if norm_t1 < 1e-9:
        v_arbitrary = np.array([0., 0., 1.])
        t1 = np.cross(n, v_arbitrary)
        norm_t1 = LA.norm(t1)
        if norm_t1 < 1e-9:
            if np.abs(n[0]) > 0.9: t1 = np.array([0.,1.,0.])
            elif np.abs(n[1]) > 0.9: t1 = np.array([1.,0.,0.])
            else: t1 = np.array([1.,0.,0.])
            norm_t1 = LA.norm(t1)
            if norm_t1 < 1e-9: raise ValueError("Fallback t1 is zero.")
    t1 /= norm_t1
    t2_temp = np.cross(n, t1)
    norm_t2 = LA.norm(t2_temp)
    if norm_t2 < 1e-9: raise ValueError("Cannot compute tangent 2.")
    t2 = t2_temp / norm_t2
    return t1, t2

def load_prediction_components(model_path,x_scaler_path,y_scaler_path,input_dim,output_dim,h1,h2,h3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ML使用设备: {device}")
    model = MLPRegression(input_dim,output_dim,h1,h2,h3)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"加载模型 {model_path} 时出错: {e}")
        return None,None,None,None
    scaler_X,scaler_y=None,None
    try:
        scaler_X=joblib.load(x_scaler_path)
        print(f"X Scaler 加载成功。")
    except FileNotFoundError:
        print(f"警告: 未找到 X scaler '{x_scaler_path}'。")
    except Exception as e:
        print(f"加载 X Scaler '{x_scaler_path}' 时出错: {e}")
        return None,None,None,None
    try:
        scaler_y=joblib.load(y_scaler_path)
        print(f"Y Scaler 加载成功。")
    except FileNotFoundError:
        print(f"警告: 未找到 Y scaler '{y_scaler_path}'。")
    except Exception as e:
        print(f"加载 Y Scaler '{y_scaler_path}' 时出错: {e}")
        return None,None,None,None
    return model,scaler_X,scaler_y,device

def predict_displacements_for_pressure(model,scaler_X,scaler_y,device,pressure_value):
    if model is None:
        print("模型未加载。")
        return None
    pressure_value=np.clip(pressure_value,P_BOUNDS[0],P_BOUNDS[1])
    input_p=np.array([[pressure_value]],dtype=np.float32)
    if scaler_X:
        try:
            input_p_scaled = scaler_X.transform(input_p)
        except Exception as e:
            print(f"X scaler 标准化出错: {e}")
            return None
    else:
        input_p_scaled = input_p
    input_tensor=torch.tensor(input_p_scaled,dtype=torch.float32).to(device)
    predicted_original_scale=None
    with torch.no_grad():
        try:
            predicted_scaled_tensor=model(input_tensor)
            predicted_scaled=predicted_scaled_tensor.cpu().numpy()
            if scaler_y:
                try:
                    predicted_original_scale = scaler_y.inverse_transform(predicted_scaled)
                except Exception as e:
                    print(f"Y scaler 反标准化出错: {e}")
                    return None
            else:
                predicted_original_scale = predicted_scaled
        except Exception as e:
            print(f"模型预测出错: {e}")
            return None
    if predicted_original_scale is not None:
        if predicted_original_scale.shape[1] != OUTPUT_DIM:
            print(f"错误：模型输出维度错误")
            return None
        return predicted_original_scale.reshape(NODE_COUNT, 3)
    else:
        return None

def calculate_gii_multi_contact(contacts_info,object_centroid_for_gii,mu,eigenvalue_thresh):
    if not contacts_info or len(contacts_info) < 2: return None
    all_wrenches = []
    valid_contacts_for_gii = 0
    for contact in contacts_info:
        if isinstance(contact, (tuple, list)) and len(contact) >= 6:
            pt_on_mesh = contact[4]
            normal_finger = contact[5]
        else:
            continue
        if pt_on_mesh is None or normal_finger is None: continue
        pt_on_mesh = np.asarray(pt_on_mesh)
        normal_finger = np.asarray(normal_finger)
        if pt_on_mesh.size != 3 or normal_finger.size != 3: continue
        norm_mag = LA.norm(normal_finger)
        if norm_mag < 1e-6: continue
        n_contact = - (normal_finger / norm_mag)
        try:
            t1, t2 = get_orthogonal_vectors(n_contact)
        except ValueError:
            continue
        r_contact = pt_on_mesh - object_centroid_for_gii
        d_list = [n_contact + mu * t1, n_contact - mu * t1, n_contact + mu * t2, n_contact - mu * t2]
        all_wrenches.extend([np.concatenate((d, np.cross(r_contact, d))) for d in d_list])
        valid_contacts_for_gii += 1
    if valid_contacts_for_gii < 2 : return None
    grasp_matrix_G = np.column_stack(all_wrenches)
    J = grasp_matrix_G @ grasp_matrix_G.T
    try:
        eigenvalues = LA.eigvalsh(J)
        non_zero_eigenvalues = eigenvalues[eigenvalues > eigenvalue_thresh]
        if len(non_zero_eigenvalues) > 0:
            lambda_min = np.min(non_zero_eigenvalues)
            if lambda_min < 0 and np.abs(lambda_min) < eigenvalue_thresh: lambda_min = 0.0
            elif lambda_min < 0: return None
            lambda_max = np.max(non_zero_eigenvalues)
            if lambda_max < eigenvalue_thresh: return 0.0
            return np.sqrt(lambda_min / lambda_max) if lambda_max > 1e-9 else 0.0
        else:
            return 0.0
    except LA.LinAlgError:
        return None

def get_rotation_matrix_between_vectors(vec1, vec2):
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        if c > 0: return np.identity(3)
        else:
            if np.abs(a[0]) > 0.9 or np.abs(a[1]) > 0.9 : axis_ortho = np.array([0.0, 0.0, 1.0])
            else: axis_ortho = np.array([1.0, 0.0, 0.0])
            if np.linalg.norm(np.cross(a, axis_ortho)) < 1e-6: axis_ortho = np.array([0.0, 1.0, 0.0])
            return create_rotation_matrix(axis_ortho, np.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.identity(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return r

# 全局变量定义
initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, \
device_global = [None]*5
object_points_global_static, object_centroid_global_static, object_mesh_global_static = [None]*3
world_obb_object_global = None
pv_obb_object_mesh_global = None
num_object_points_global_static, faces_np_global, width_translation_vector_global = 0, None, None
T1_translate_global, T2_rotate_global = None, None
T_pose_for_tray_display_and_finger_placement_global = None

# 新增：抓取评估指标的权重 (全局)
W_GII = 1.0  # GII的权重
W_FINGER_SPREAD = 0.7 # 手指分布均匀性指标的权重 (初始建议值，可调整)

@use_named_args(dimensions=dimensions)
def evaluate_grasp(r, p1, p2, p3, pos_idx1, pos_idx2, pos_idx3):
    global initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global, \
           object_points_global_static, object_centroid_global_static, num_object_points_global_static, \
           faces_np_global, width_translation_vector_global, T1_translate_global, T2_rotate_global, \
           T_pose_for_tray_display_and_finger_placement_global, N_FINGER_SLOTS, \
           W_GII, W_FINGER_SPREAD # 使用全局权重

    chosen_indices = [int(pos_idx1), int(pos_idx2), int(pos_idx3)]
    if len(set(chosen_indices)) < 3:
        return 10.0 # 惩罚：手指位置索引不唯一

    # --- 计算手指分布均匀性指标 (norm_metric_finger_spread) ---
    angles_deg = sorted([(idx / N_FINGER_SLOTS) * 360.0 for idx in chosen_indices])
    s_a1, s_a2, s_a3 = angles_deg[0], angles_deg[1], angles_deg[2]

    sep1 = s_a2 - s_a1
    sep2 = s_a3 - s_a2
    sep3 = 360.0 - (s_a3 - s_a1) 

    min_sep = min(sep1, sep2, sep3)
    
    # 归一化 (0到1，1表示最佳120度间隔)
    # 最小可能的min_sep是相邻槽位间的角度 (360/N_FINGER_SLOTS)
    # 最大可能的min_sep是120度 (理想情况)
    min_possible_min_sep = 360.0 / N_FINGER_SLOTS if N_FINGER_SLOTS > 0 else 0
    max_possible_min_sep = 120.0
    
    norm_metric_finger_spread = 0.0
    denominator = max_possible_min_sep - min_possible_min_sep
    if N_FINGER_SLOTS >=3 and denominator > 1e-6 : # 避免除以零, 且对于N_SLOTS<3意义不大
        norm_metric_finger_spread = (min_sep - min_possible_min_sep) / denominator
        norm_metric_finger_spread = np.clip(norm_metric_finger_spread, 0.0, 1.0)
    elif N_FINGER_SLOTS >=3 and abs(min_sep - max_possible_min_sep) < 1e-6 : # 例如 N_SLOTS = 3, 任何有效放置都是120度
        norm_metric_finger_spread = 1.0
    # --- End 手指分布指标计算 ---

    print_eval_string = f"  评估: r={r:.3f}, P=({p1:.0f},{p2:.0f},{p3:.0f}), pos=({chosen_indices[0]},{chosen_indices[1]},{chosen_indices[2]}), spread={norm_metric_finger_spread:.3f}"
    print(print_eval_string, end="")

    current_pressures = [p1, p2, p3]
    current_placement_radius = r
    deformed_finger_meshes_world_this_eval = []
    mesh_generation_successful = True
    for i in range(3):
        displacements_matrix = predict_displacements_for_pressure(model_global, scaler_X_global, scaler_y_global, device_global, current_pressures[i])
        if displacements_matrix is None: mesh_generation_successful = False; break
        deformed_curve1_ref_unordered = initial_coords_ref_global + displacements_matrix
        curve2_ref = initial_coords_ref_global + width_translation_vector_global
        deformed_curve2_ref_unordered = curve2_ref + displacements_matrix
        sorted_deformed_curve1_ref = sort_points_spatially(deformed_curve1_ref_unordered)
        sorted_deformed_curve2_ref = sort_points_spatially(deformed_curve2_ref_unordered)
        if sorted_deformed_curve1_ref is None or sorted_deformed_curve2_ref is None: mesh_generation_successful = False; break
        sorted_deformed_vertices_ref = np.vstack((sorted_deformed_curve1_ref, sorted_deformed_curve2_ref))
        if faces_np_global is None or faces_np_global.size == 0: mesh_generation_successful = False; break
        try: deformed_mesh_ref = pv.PolyData(sorted_deformed_vertices_ref, faces=faces_np_global)
        except Exception: mesh_generation_successful = False; break
        
        current_pos_index = chosen_indices[i]
        current_angle_deg = current_pos_index * (360.0 / N_FINGER_SLOTS)
        angle_rad = np.radians(current_angle_deg)
        rot_angle_z_placing = angle_rad + np.pi / 2.0
        rot_z_placing = create_rotation_matrix_z(rot_angle_z_placing)
        target_pos_on_circle = np.array([ current_placement_radius * np.cos(angle_rad), current_placement_radius * np.sin(angle_rad), 0.0 ])
        T3_place = create_transformation_matrix_opt8(rot_z_placing, target_pos_on_circle)
        T_transform_finger_relative_to_tray_origin = T3_place @ T2_rotate_global @ T1_translate_global
        if T_pose_for_tray_display_and_finger_placement_global is None: mesh_generation_successful = False; break
        T_final_finger_world = T_pose_for_tray_display_and_finger_placement_global @ T_transform_finger_relative_to_tray_origin
        try:
            final_transformed_finger_mesh = deformed_mesh_ref.transform(T_final_finger_world, inplace=False)
            if final_transformed_finger_mesh is None or final_transformed_finger_mesh.n_points == 0: mesh_generation_successful = False; break
            final_transformed_finger_mesh.clean(inplace=True)
            if final_transformed_finger_mesh.n_cells > 0:
                final_transformed_finger_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True, auto_orient_normals=True, non_manifold_traversal=False)
        except Exception: mesh_generation_successful = False; break
        deformed_finger_meshes_world_this_eval.append(final_transformed_finger_mesh)

    if not mesh_generation_successful:
        print(f" -> 网格生成失败, Cost=10.0")
        return 10.0

    object_point_status = ['Non-Contact'] * num_object_points_global_static
    closest_contact_per_finger_eval = [None] * 3
    min_dist_per_finger_eval = [float('inf')] * 3
    finger_dot_products = [[] for _ in range(3)]
    has_overlap_eval = False
    num_contact_points = 0
    dot_prod_tolerance = 1e-6

    if object_points_global_static is None or num_object_points_global_static == 0:
        print(f" -> 物体点云为空, Cost=10.0")
        return 10.0

    for obj_pt_idx, obj_point in enumerate(object_points_global_static):
        closest_dist_for_this_pt = float('inf')
        finger_idx_for_this_pt = -1
        normal_for_this_pt = None
        pt_on_mesh_for_this_pt = None
        for finger_idx, finger_mesh in enumerate(deformed_finger_meshes_world_this_eval):
            if finger_mesh is None or finger_mesh.n_cells == 0 : continue
            has_normals = 'Normals' in finger_mesh.cell_data and finger_mesh.cell_data['Normals'] is not None
            try:
                closest_cell_id, pt_on_mesh_candidate = finger_mesh.find_closest_cell(obj_point, return_closest_point=True)
                if closest_cell_id < 0 or closest_cell_id >= finger_mesh.n_cells: continue
                dist = LA.norm(obj_point - pt_on_mesh_candidate)
                if dist < closest_dist_for_this_pt:
                    closest_dist_for_this_pt = dist
                    finger_idx_for_this_pt = finger_idx
                    pt_on_mesh_for_this_pt = pt_on_mesh_candidate
                    current_normal = None
                    if has_normals and closest_cell_id < len(finger_mesh.cell_data['Normals']):
                        current_normal = finger_mesh.cell_normals[closest_cell_id]
                    normal_for_this_pt = current_normal
                if dist < collision_threshold and dist < min_dist_per_finger_eval[finger_idx]:
                    min_dist_per_finger_eval[finger_idx] = dist
                    current_normal_for_gii = None
                    if has_normals and closest_cell_id < len(finger_mesh.cell_data['Normals']):
                        current_normal_for_gii = finger_mesh.cell_normals[closest_cell_id]
                    if current_normal_for_gii is not None:
                        closest_contact_per_finger_eval[finger_idx] = (dist, obj_pt_idx, finger_idx, closest_cell_id, pt_on_mesh_candidate, current_normal_for_gii)
            except Exception: continue

        if finger_idx_for_this_pt != -1:
            dist = closest_dist_for_this_pt
            normal_closest = normal_for_this_pt
            pt_on_mesh_closest = pt_on_mesh_for_this_pt
            if dist < overlap_threshold:
                object_point_status[obj_pt_idx] = 'Overlap'; has_overlap_eval = True
            elif dist < collision_threshold:
                object_point_status[obj_pt_idx] = 'Contact'; num_contact_points += 1
                if normal_closest is not None and LA.norm(normal_closest) > 1e-9:
                    vector_to_point = obj_point - pt_on_mesh_closest
                    if LA.norm(vector_to_point) > 1e-9:
                        dot_prod = np.dot(vector_to_point / LA.norm(vector_to_point), normal_closest / LA.norm(normal_closest))
                        finger_dot_products[finger_idx_for_this_pt].append(dot_prod)

    final_contacts_for_gii_eval = [info for info in closest_contact_per_finger_eval if info is not None]
    grasp_state_eval = "No Contact"
    cost = 3.0 
    gii_value_for_print = 0.0 # 用于打印的GII值

    if has_overlap_eval:
        grasp_state_eval = "Overlap"; cost = 5.0
    else:
        finger_intersects = [False] * 3
        for i in range(3):
            if finger_dot_products[i]:
                has_positive_dp = any(dp > dot_prod_tolerance for dp in finger_dot_products[i])
                has_negative_dp = any(dp < -dot_prod_tolerance for dp in finger_dot_products[i])
                if has_positive_dp and has_negative_dp: finger_intersects[i] = True
        if any(finger_intersects):
            grasp_state_eval = "Intersection"; cost = 4.0
        elif num_contact_points > 0 :
            grasp_state_eval = "Contact"
            num_gii_contacts = len(final_contacts_for_gii_eval)
            gii = None
            if num_gii_contacts >= 2:
                gii = calculate_gii_multi_contact(final_contacts_for_gii_eval, object_centroid_global_static, friction_coefficient, eigenvalue_threshold)
            
            if gii is not None and gii > 1e-9:
                gii_value_for_print = gii
                cost = - (W_GII * gii + W_FINGER_SPREAD * norm_metric_finger_spread)
            else: 
                gii_value_for_print = 0.0 # GII无效或太小
                if num_gii_contacts < 2: cost = 1.0 + ((2.0 - num_gii_contacts) * 0.3)
                else: cost = 2.0 
    
    param_str_short = f"r={r:.2f},P=({p1:.0f},{p2:.0f},{p3:.0f}),pos=({chosen_indices[0]},{chosen_indices[1]},{chosen_indices[2]})"
    if grasp_state_eval == "Contact" and cost < 0:
        print(f" -> {param_str_short}, St={grasp_state_eval}, GII={gii_value_for_print:.4f}, Spr={norm_metric_finger_spread:.3f}, Score={-cost:.4f}, Cost={cost:.4f}")
    else:
        print(f" -> {param_str_short}, St={grasp_state_eval}, NCont={num_contact_points}, NGIIC={len(final_contacts_for_gii_eval)}, Spr={norm_metric_finger_spread:.3f}, Cost={cost:.4f}")
    return cost

# --- find_initial_grasp (基本保持不变，仅修改打印信息中的参数名) ---
def find_initial_grasp(initial_r, initial_finger_indices, pressure_step, max_pressure_init):
    print(f"\n--- 开始寻找初始接触点 (r = {initial_r:.2f}, indices = {initial_finger_indices}) ---")
    global initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global, \
           object_points_global_static, num_object_points_global_static, faces_np_global, \
           width_translation_vector_global, T1_translate_global, T2_rotate_global, \
           T_pose_for_tray_display_and_finger_placement_global, N_FINGER_SLOTS

    current_pressures_init = np.array([0.0, 0.0, 0.0])
    finger_contact_achieved_init = [False, False, False]
    init_iter_count = 0
    last_meshes = [None] * 3
    last_point_status = ['Non-Contact'] * num_object_points_global_static if num_object_points_global_static > 0 else []
    last_closest_contacts = [None] * 3
    dot_prod_tolerance = 1e-6
    max_iterations = int( (max_pressure_init / pressure_step) * 3 * 1.5) + 10

    while True:
        init_iter_count += 1
        if init_iter_count > max_iterations:
            print("--- 初始点搜索: 达到最大迭代次数 ---")
            return { 'status': 'IterationLimit', 'r': initial_r, 'finger_indices': initial_finger_indices, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }

        pressure_changed_init = False
        print(f"  初始点搜索 Iter {init_iter_count}: P=[{current_pressures_init[0]:.0f}, {current_pressures_init[1]:.0f}, {current_pressures_init[2]:.0f}] (r={initial_r:.2f}, indices={initial_finger_indices})")
        for i in range(3):
            if not finger_contact_achieved_init[i] and current_pressures_init[i] < max_pressure_init:
                current_pressures_init[i] += pressure_step
                current_pressures_init[i] = min(current_pressures_init[i], max_pressure_init)
                pressure_changed_init = True
        if not pressure_changed_init and all(finger_contact_achieved_init): pass
        elif not pressure_changed_init and not all(finger_contact_achieved_init):
            print("--- 初始点搜索: 压力达到上限或无变化，但未实现三指接触 ---")
            return { 'status': 'MaxPressureReachedOrNoChange', 'r': initial_r, 'finger_indices': initial_finger_indices, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }

        deformed_finger_meshes_init = [None] * 3; valid_preds = True
        for i in range(3):
            displacements_matrix = predict_displacements_for_pressure(model_global, scaler_X_global, scaler_y_global, device_global, current_pressures_init[i])
            if displacements_matrix is None: valid_preds = False; break
            deformed_curve1_ref_unordered = initial_coords_ref_global + displacements_matrix
            curve2_ref = initial_coords_ref_global + width_translation_vector_global
            deformed_curve2_ref_unordered = curve2_ref + displacements_matrix
            sorted_deformed_curve1_ref = sort_points_spatially(deformed_curve1_ref_unordered)
            sorted_deformed_curve2_ref = sort_points_spatially(deformed_curve2_ref_unordered)
            if sorted_deformed_curve1_ref is None or sorted_deformed_curve2_ref is None: valid_preds = False; break
            sorted_deformed_vertices_ref = np.vstack((sorted_deformed_curve1_ref, sorted_deformed_curve2_ref))
            if faces_np_global is None or faces_np_global.size == 0: valid_preds = False; break
            try: deformed_mesh_ref = pv.PolyData(sorted_deformed_vertices_ref, faces=faces_np_global)
            except Exception: valid_preds = False; break
            current_pos_index = initial_finger_indices[i]
            current_angle_deg = current_pos_index * (360.0 / N_FINGER_SLOTS)
            angle_rad = np.radians(current_angle_deg)
            rot_angle_z_placing = angle_rad + np.pi / 2.0
            rot_z_placing = create_rotation_matrix_z(rot_angle_z_placing)
            target_pos_on_circle = np.array([ initial_r * np.cos(angle_rad), initial_r * np.sin(angle_rad), 0.0 ])
            T3_place = create_transformation_matrix_opt8(rot_z_placing, target_pos_on_circle)
            T_transform_finger_relative_to_tray_origin = T3_place @ T2_rotate_global @ T1_translate_global
            if T_pose_for_tray_display_and_finger_placement_global is None: valid_preds = False; break
            T_final_finger_world = T_pose_for_tray_display_and_finger_placement_global @ T_transform_finger_relative_to_tray_origin
            try:
                final_transformed_finger_mesh = deformed_mesh_ref.transform(T_final_finger_world, inplace=False)
                if final_transformed_finger_mesh is None or final_transformed_finger_mesh.n_points == 0: valid_preds = False; break
                final_transformed_finger_mesh.clean(inplace=True)
                if final_transformed_finger_mesh.n_cells > 0:
                    final_transformed_finger_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True, auto_orient_normals=True, non_manifold_traversal=False)
                # 为手指网格增加厚度
                try:
                    # 确保法线已计算且方向合理 (compute_normals 中 auto_orient_normals=True 有帮助)
                    thickened_mesh = final_transformed_finger_mesh.thicken(thickness=FINGER_VIZ_THICKNESS, inplace=False)
                    deformed_finger_meshes_init[i] = thickened_mesh
                except Exception as e_thicken:
                    # print(f"警告: 在find_initial_grasp中为手指 {i} 增加厚度失败: {e_thicken}")
                    deformed_finger_meshes_init[i] = final_transformed_finger_mesh # 出错则回退到原始网格
            except Exception: valid_preds = False; break
        if not valid_preds:
            print("  初始点搜索: 预测或网格处理失败。")
            return { 'status': 'PredictionFailed', 'r': initial_r, 'finger_indices': initial_finger_indices, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }
        last_meshes = [mesh.copy() if mesh is not None else None for mesh in deformed_finger_meshes_init]
        has_overlap_init = False; num_contact_points_init = 0
        current_closest_contacts = [None] * 3; current_min_dists = [float('inf')] * 3
        current_point_status = ['Non-Contact'] * num_object_points_global_static if num_object_points_global_static > 0 else []
        finger_dot_products_init = [[] for _ in range(3)]; contacting_fingers_indices_this_iter = set()
        if object_points_global_static is not None and num_object_points_global_static > 0:
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
                            if current_normal_contact is not None: current_closest_contacts[finger_idx] = (dist, obj_pt_idx, finger_idx, closest_cell_id, pt_on_mesh_candidate, current_normal_contact)
                    except Exception: continue
                if finger_idx_for_this_pt != -1:
                    dist_final_for_pt = closest_dist_for_this_pt; normal_final_for_pt = normal_for_this_pt; pt_on_mesh_final_for_pt = pt_on_mesh_for_this_pt
                    if dist_final_for_pt < overlap_threshold: current_point_status[obj_pt_idx] = 'Overlap'; has_overlap_init = True
                    elif dist_final_for_pt < collision_threshold:
                        current_point_status[obj_pt_idx] = 'Contact'; num_contact_points_init += 1; contacting_fingers_indices_this_iter.add(finger_idx_for_this_pt)
                        if normal_final_for_pt is not None and LA.norm(normal_final_for_pt) > 1e-9:
                            vector_to_point = obj_point - pt_on_mesh_final_for_pt
                            if LA.norm(vector_to_point) > 1e-9:
                                dot_prod = np.dot(vector_to_point/LA.norm(vector_to_point), normal_final_for_pt/LA.norm(normal_final_for_pt))
                                finger_dot_products_init[finger_idx_for_this_pt].append(dot_prod)
                if has_overlap_init: break
        if has_overlap_init:
            print("  初始点搜索: 检测到重叠。")
            return { 'status': 'Overlap', 'r': initial_r, 'finger_indices': initial_finger_indices, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': current_point_status, 'contact_info': current_closest_contacts }
        last_point_status = current_point_status; last_closest_contacts = current_closest_contacts
        finger_intersects_init = [False] * 3
        if num_contact_points_init > 0:
            for i in range(3):
                if finger_dot_products_init[i]:
                    has_pos_dp = any(dp > dot_prod_tolerance for dp in finger_dot_products_init[i])
                    has_neg_dp = any(dp < -dot_prod_tolerance for dp in finger_dot_products_init[i])
                    if has_pos_dp and has_neg_dp: finger_intersects_init[i] = True
        if any(finger_intersects_init):
            print("  初始点搜索: 检测到交叉穿透。")
            return { 'status': 'Intersection', 'r': initial_r, 'finger_indices': initial_finger_indices, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }
        for i in range(3):
            if i in contacting_fingers_indices_this_iter: finger_contact_achieved_init[i] = True
        print(f"  初始点搜索: 接触状态={list(finger_contact_achieved_init)}")
        if all(finger_contact_achieved_init):
            print(f"--- 成功找到初始接触点组合 (r={initial_r:.2f}, indices={initial_finger_indices}, P={current_pressures_init.round(0)}) ---")
            return { 'status': 'FoundInitialPoint', 'r': initial_r, 'finger_indices': initial_finger_indices, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }
        if not pressure_changed_init and not all(finger_contact_achieved_init):
            print("--- 初始点搜索: 压力达到上限或无变化，但未实现三指接触 (循环终止检查) ---")
            return { 'status': 'MaxPressureReachedOrNoChangeLoopEnd', 'r': initial_r, 'finger_indices': initial_finger_indices, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }
    return { 'status': 'LoopEndUnknown', 'r': initial_r, 'finger_indices': initial_finger_indices, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }

# --- visualize_poses_with_open3d (保持不变) ---
def visualize_poses_with_open3d(
                                tray_geometry_transform_mm, 
                                tray_axes_transform_mm,     
                                bottle_axes_transform_mm,   
                                bottle_points_world_mm,     
                                bottle_colors_rgb_float,
                                bottle_obb_world,           
                                tray_radius_mm, tray_height_mm,
                                window_title="Open3D Relative Pose Visualization"):
    print(f"\n--- 打开 Open3D 窗口: {window_title} ---")
    geometries = []
    o3d_tray_canonical_centered = o3d.geometry.TriangleMesh.create_cylinder(radius=tray_radius_mm, height=tray_height_mm, resolution=20, split=4)
    o3d_tray_canonical_centered.translate(np.array([0,0,-tray_height_mm/2.0]))
    o3d_tray_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_tray_canonical_centered)
    o3d_tray_wireframe.transform(tray_geometry_transform_mm)
    o3d_tray_wireframe.paint_uniform_color(pv.Color(tray_color_viz).float_rgb) # 使用美化后的颜色
    geometries.append(o3d_tray_wireframe)

    o3d_bottle_pcd = o3d.geometry.PointCloud()
    o3d_bottle_pcd.points = o3d.utility.Vector3dVector(bottle_points_world_mm)
    if bottle_colors_rgb_float is not None and len(bottle_colors_rgb_float) == len(bottle_points_world_mm):
        o3d_bottle_pcd.colors = o3d.utility.Vector3dVector(bottle_colors_rgb_float)
    else:
        o3d_bottle_pcd.paint_uniform_color(pv.Color(object_point_color_viz).float_rgb) # 使用美化后的颜色
    geometries.append(o3d_bottle_pcd)

    if bottle_obb_world is not None:
        # OBB颜色在创建时已设置，这里确保它与PyVista一致
        bottle_obb_world.color = pv.Color(object_obb_color_viz).float_rgb 
        geometries.append(bottle_obb_world)

    tray_axes_size = tray_radius_mm * 0.7
    o3d_tray_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=tray_axes_size, origin=[0,0,0])
    o3d_tray_axes.transform(tray_axes_transform_mm)
    geometries.append(o3d_tray_axes)

    if bottle_points_world_mm.shape[0] > 0:
        temp_bottle_pcd_for_bbox = o3d.geometry.PointCloud()
        temp_bottle_pcd_for_bbox.points = o3d.utility.Vector3dVector(bottle_points_world_mm)
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
    o3d.visualization.draw_geometries(geometries, window_name=window_title, width=1000, height=800) # 增大窗口
    print(f"--- 关闭 Open3D 窗口: {window_title} ---")

# --- 主脚本 ---
if __name__ == '__main__':
    # --- A 至 I 部分基本保持不变，除了打印和文件名 ---
    T_gn_gripper_TO_gn_object_meters = load_transformation_matrix_from_txt(RELATIVE_POSE_FILE_PATH)
    if T_gn_gripper_TO_gn_object_meters is None: sys.exit("错误：未能加载物体相对于GraspNet夹爪的位姿。")
    T_original_tray_pose_world = create_transformation_matrix_opt8(np.identity(3), tray_center)
    T_gn_object_TO_gn_gripper_meters = np.linalg.inv(T_gn_gripper_TO_gn_object_meters)
    T_imported_obj_relative_to_tray_mm = copy.deepcopy(T_gn_object_TO_gn_gripper_meters)
    T_imported_obj_relative_to_tray_mm[:3, 3] *= 1000.0
    T_original_object_pose_world = T_original_tray_pose_world @ T_imported_obj_relative_to_tray_mm
    T_tray_coord_system_in_world = T_original_object_pose_world
    T_object_coord_system_in_world = T_original_tray_pose_world
    angle_rad_fix = np.pi / 2; cos_fix = np.cos(angle_rad_fix); sin_fix = np.sin(angle_rad_fix)
    T_local_orientation_fix_for_tray_geometry = np.array([[cos_fix,0,sin_fix,0],[0,1,0,0],[-sin_fix,0,cos_fix,0],[0,0,0,1]])
    T_actual_tray_geometry_world = T_tray_coord_system_in_world @ T_local_orientation_fix_for_tray_geometry
    T_pose_for_tray_display_and_finger_placement_global = T_actual_tray_geometry_world
    print(f"\n加载高精度物体点云: {HIGH_RES_OBJECT_PLY_PATH}")
    if not os.path.exists(HIGH_RES_OBJECT_PLY_PATH): sys.exit(f"错误: {HIGH_RES_OBJECT_PLY_PATH} 未找到")
    try:
        high_res_object_o3d = o3d.io.read_point_cloud(HIGH_RES_OBJECT_PLY_PATH)
        if not high_res_object_o3d.has_points(): sys.exit(f"错误: {HIGH_RES_OBJECT_PLY_PATH} 为空")
    except Exception as e: sys.exit(f"加载 {HIGH_RES_OBJECT_PLY_PATH} 错: {e}")
    current_high_res_points_mm_orig_frame = np.asarray(high_res_object_o3d.points)
    if OBJECT_SCALE_FACTOR != 1.0: current_high_res_points_mm_orig_frame *= OBJECT_SCALE_FACTOR
    centroid_high_res_mm = np.mean(current_high_res_points_mm_orig_frame, axis=0)
    points_high_res_centered_mm = current_high_res_points_mm_orig_frame - centroid_high_res_mm
    points_high_res_centered_aligned_mm = points_high_res_centered_mm
    if points_high_res_centered_mm.shape[0] > 0:
        local_pcd_for_alignment = o3d.geometry.PointCloud(); local_pcd_for_alignment.points = o3d.utility.Vector3dVector(points_high_res_centered_mm)
        try:
            local_obb = local_pcd_for_alignment.get_oriented_bounding_box(); longest_extent_idx = np.argmax(local_obb.extent)
            local_longest_axis_vec = local_obb.R[:, longest_extent_idx]; target_local_x_axis = np.array([1.0, 0.0, 0.0])
            R_local_align = get_rotation_matrix_between_vectors(local_longest_axis_vec, target_local_x_axis)
            if R_local_align is not None:
                T_local_align_homogeneous = create_transformation_matrix_opt8(R_local_align, None)
                points_high_res_centered_aligned_mm = transform_points_opt8(points_high_res_centered_mm, T_local_align_homogeneous)
        except Exception as e_obb_align: print(f"  局部OBB对齐过程中出错: {e_obb_align}。")
    object_points_transformed_full_mm = transform_points_opt8(points_high_res_centered_aligned_mm, T_object_coord_system_in_world)
    final_sampled_points_mm = None; sampled_colors_uint8_pv = None; sampled_colors_float_o3d = None
    original_hr_colors_float_o3d = np.asarray(high_res_object_o3d.colors) if high_res_object_o3d.has_colors() else None
    if len(object_points_transformed_full_mm) > TARGET_POINT_COUNT_FOR_SIM:
        indices = np.random.choice(len(object_points_transformed_full_mm), TARGET_POINT_COUNT_FOR_SIM, replace=False)
        final_sampled_points_mm = object_points_transformed_full_mm[indices]
        if original_hr_colors_float_o3d is not None and indices.size > 0:
            sampled_colors_float_o3d = original_hr_colors_float_o3d[indices]
            if sampled_colors_float_o3d.ndim == 2 and sampled_colors_float_o3d.shape[1] == 3: sampled_colors_uint8_pv = (sampled_colors_float_o3d * 255).astype(np.uint8)
            else: sampled_colors_uint8_pv = None; sampled_colors_float_o3d = None
    elif len(object_points_transformed_full_mm) > 0 :
        final_sampled_points_mm = object_points_transformed_full_mm
        if original_hr_colors_float_o3d is not None:
            if original_hr_colors_float_o3d.ndim == 2 and original_hr_colors_float_o3d.shape[1] == 3:
                sampled_colors_float_o3d = original_hr_colors_float_o3d; sampled_colors_uint8_pv = (original_hr_colors_float_o3d * 255).astype(np.uint8)
            else: sampled_colors_uint8_pv = None; sampled_colors_float_o3d = None
    else: sys.exit(f"错误: 变换后的高精度点云不含点.")
    object_points_global_static = final_sampled_points_mm
    object_mesh_global_static = pv.PolyData(object_points_global_static)
    if sampled_colors_uint8_pv is not None: object_mesh_global_static.point_data['colors'] = sampled_colors_uint8_pv
    object_centroid_global_static = np.mean(object_points_global_static, axis=0)
    num_object_points_global_static = object_points_global_static.shape[0]
    if num_object_points_global_static > 0:
        o3d_temp_pcd_for_obb = o3d.geometry.PointCloud(); o3d_temp_pcd_for_obb.points = o3d.utility.Vector3dVector(object_points_global_static)
        world_obb_object_global = o3d_temp_pcd_for_obb.get_oriented_bounding_box()
        world_obb_object_global.color = pv.Color(object_obb_color_viz).float_rgb # 使用美化后的颜色
        pv_obb_object_mesh_global = pv.Cube(center=(0,0,0), x_length=world_obb_object_global.extent[0], y_length=world_obb_object_global.extent[1], z_length=world_obb_object_global.extent[2])
        T_pv_obb_transform = np.eye(4); T_pv_obb_transform[:3,:3] = world_obb_object_global.R; T_pv_obb_transform[:3,3] = world_obb_object_global.center
        pv_obb_object_mesh_global.transform(T_pv_obb_transform, inplace=True)
    initial_coords_ref_global = load_initial_coordinates(INITIAL_COORDS_PATH, NODE_COUNT)
    model_global,scaler_X_global,scaler_y_global,device_global = load_prediction_components(MODEL_PATH,X_SCALER_PATH,Y_SCALER_PATH,INPUT_DIM,OUTPUT_DIM,HIDDEN_LAYER_1,HIDDEN_LAYER_2,HIDDEN_LAYER_3)
    if initial_coords_ref_global is None or model_global is None: sys.exit("错误：未能初始化手指模型或坐标。")
    faces_np_global = create_faces_array(NODE_COUNT)
    width_translation_vector_global = np.array([0, finger_width, 0])
    bottom_node_idx = np.argmin(initial_coords_ref_global[:,0]); ref_mid_pt = initial_coords_ref_global[bottom_node_idx] + width_translation_vector_global/2.0
    T1_translate_global = create_transformation_matrix_opt8(None,-ref_mid_pt)
    rot_ref_to_local = np.array([[0,1,0],[0,0,1],[1,0,0]]); T2_rotate_global = create_transformation_matrix_opt8(rot_ref_to_local,None)
    if object_points_global_static is None: sys.exit("错误：object_points_global_static 未初始化。")
    tray_pv_canonical = pv.Cylinder(center=(0,0,0), radius=tray_radius, height=tray_height, direction=(0,0,1), resolution=30) # 减少一点托盘分辨率
    tray_pv = tray_pv_canonical.transform(T_actual_tray_geometry_world, inplace=False)
    print(f"\n--- 开始在 {NUM_INITIAL_SEARCHES} 个随机 (r, finger_indices) 组合上寻找初始接触点 ---")
    collected_initial_points = []; last_search_attempt_result = None
    x_coords_rel_new_obj_centroid = object_points_global_static[:, 0] - object_centroid_global_static[0]
    y_coords_rel_new_obj_centroid = object_points_global_static[:, 1] - object_centroid_global_static[1]
    distances_from_new_obj_centroid_xy = np.sqrt(x_coords_rel_new_obj_centroid**2 + y_coords_rel_new_obj_centroid**2)
    if distances_from_new_obj_centroid_xy.size > 0:
        effective_radius_of_object_at_old_tray_pos = np.max(distances_from_new_obj_centroid_xy)
        r_search_min_heuristic = effective_radius_of_object_at_old_tray_pos * 0.7 # 调整启发式搜索
        r_search_max_heuristic = tray_radius * 1.1 
    else: r_search_min_heuristic = R_BOUNDS[0]; r_search_max_heuristic = R_BOUNDS[1]
    r_search_min = np.clip(r_search_min_heuristic, R_BOUNDS[0], R_BOUNDS[1] * 0.95)
    r_search_max = np.clip(r_search_max_heuristic, r_search_min + (R_BOUNDS[1] - R_BOUNDS[0]) * 0.05, R_BOUNDS[1])
    if r_search_min >= r_search_max: r_search_min=R_BOUNDS[0]; r_search_max=R_BOUNDS[1]
    print(f"更新后的初始搜索范围: r in [{r_search_min:.2f}, {r_search_max:.2f}]")
    for k_init in range(NUM_INITIAL_SEARCHES):
        print(f"\n--- 初始搜索尝试 {k_init+1}/{NUM_INITIAL_SEARCHES} ---")
        r_val = np.random.uniform(r_search_min,r_search_max)
        possible_indices = list(range(N_FINGER_SLOTS))
        chosen_k_init_indices = random.sample(possible_indices, 3); chosen_k_init_indices.sort()
        search_res = find_initial_grasp(initial_r=r_val,initial_finger_indices=chosen_k_init_indices,pressure_step=PRESSURE_STEP_INIT_SEARCH,max_pressure_init=max_pressure)
        last_search_attempt_result = search_res
        if search_res and search_res['status']=='FoundInitialPoint':
            params_entry=[search_res['r']] + search_res['pressures'] + search_res['finger_indices']
            collected_initial_points.append(params_entry)
            print(f"*** 找到有效初始点: r={params_entry[0]:.2f},P={np.round(params_entry[1:4],0)},indices=({params_entry[4]},{params_entry[5]},{params_entry[6]}) ***")
        else: print(f"--- 未找到有效初始接触点 (状态: {search_res.get('status', 'Unknown')}) ---")
    print(f"\n--- 初始接触点搜索完成，共找到 {len(collected_initial_points)} 个有效初始点 ---")

    # --- PyVista 可视化初始搜索尝试 ---
    if last_search_attempt_result and 'finger_meshes' in last_search_attempt_result:
        print("\n(PyVista) 可视化最后一次初始搜索尝试的状态...")
        title_init_pv_viz = f"PyVista - Initial Search Attempt (Status: {last_search_attempt_result.get('status','N/A')})"
        
        plotter_theme = pv.themes.DocumentTheme()
        plotter_theme.font.family = font_family_viz
        plotter_theme.font.color = text_color_viz 
        plotter_theme.font.size = 12 # For text overlay
        plotter_theme.font.label_size = 10 # For axes labels from add_axes_at_origin
        plotter_theme.background = pv.Color(background_color_viz)

        plotter_init_pv = pv.Plotter(window_size=[1000,800], theme=plotter_theme, title=title_init_pv_viz)
        plotter_init_pv.enable_anti_aliasing('msaa', multi_samples=8)
        plotter_init_pv.enable_parallel_projection()
        # plotter_init_pv.enable_specular_lighting() # REMOVED - enable_lightkit should handle this
        plotter_init_pv.specular_power = 10.0 # Control shininess
        plotter_init_pv.remove_all_lights() # Start fresh
        plotter_init_pv.enable_lightkit()   # Add a good set of default lights, includes specular

        # 自定义角点方向指示器 (CubeAxesActor)
        if plotter_init_pv.renderer.cube_axes_actor is not None and hasattr(plotter_init_pv.renderer.cube_axes_actor, 'GetXAxisCaptionActor2D'):
            actor = plotter_init_pv.renderer.cube_axes_actor

            # 新增：不显示轴标题 (X, Y, Z 文字)
            actor.SetXTitle("") # 设置X轴标题为空字符串
            actor.SetYTitle("") # 设置Y轴标题为空字符串
            actor.SetZTitle("") # 设置Z轴标题为空字符串

            for i in range(3): # X, Y, Z
                cap_prop = actor.GetCaptionTextProperty(i)
                if cap_prop: # vtk 9.3+
                    cap_prop.SetFontFamily(vtk.VTK_TIMES)
                    cap_prop.SetFontSize(10) # 较小的字体
                    cap_prop.SetColor(0,0,0) # 黑色
                    cap_prop.SetBold(0)
                else: # Older vtk or different actor setup
                    # Try general title properties if caption specific not found
                    title_prop = actor.GetTitleTextProperty(i)
                    if title_prop:
                        title_prop.SetFontFamily(vtk.VTK_TIMES)
                        title_prop.SetFontSize(10)
                        title_prop.SetColor(0,0,0)
                        title_prop.SetBold(0)

            actor.SetXTitle("X"); actor.SetYTitle("Y"); actor.SetZTitle("Z")
            actor.GetProperty().SetLineWidth(1.0) # 使方向指示器的轴线更细

        plotter_init_pv.add_mesh(tray_pv, color=pv.Color(tray_color_viz, opacity=200), smooth_shading=True, name='tray_init_pv')
        if object_mesh_global_static:
            if 'colors' in object_mesh_global_static.point_data and object_mesh_global_static.point_data['colors'] is not None:
                plotter_init_pv.add_mesh(object_mesh_global_static, scalars='colors', rgba=True, style='points', point_size=3.5, render_points_as_spheres=True, name='obj_init_pv_color')
            else:
                plotter_init_pv.add_mesh(object_mesh_global_static, color=object_point_color_viz, style='points', point_size=3.5, render_points_as_spheres=True, name='obj_init_pv_def')
        # if pv_obb_object_mesh_global is not None:
            # plotter_init_pv.add_mesh(pv_obb_object_mesh_global, style='wireframe', color=object_obb_color_viz, line_width=1.5, name='object_obb_init_pv')
        if show_axes: # 轴在原点
            plotter_init_pv.add_axes_at_origin(line_width=1.5, labels_off=True) # 标签由主题字体大小控制

        meshes_viz_i_pv = last_search_attempt_result.get('finger_meshes',[])
        for i,m_viz_i_pv in enumerate(meshes_viz_i_pv):
            if m_viz_i_pv is not None:
                plotter_init_pv.add_mesh(
                    m_viz_i_pv,
                    color=finger_color_viz,
                    style='surface',  # 保持表面渲染
                    opacity=0.95,
                    smooth_shading=True,
                    show_edges=True,       # 新增：显示边框
                    edge_color='dimgray',  # 新增：边框颜色
                    line_width=0.5,        # 新增：边框线宽 (可以调整)
                    name=f'f_init_pv_{i}'
                )
        
        r_disp = last_search_attempt_result.get('r','?'); indices_disp = last_search_attempt_result.get('finger_indices', ['?']*3)
        pressures_disp = np.array(last_search_attempt_result.get('pressures',['?']*3)).round(0); status_disp = last_search_attempt_result.get('status','N/A')
        status_txt_i_pv = f"Last Search: r={r_disp:.2f}, indices=({indices_disp[0]},{indices_disp[1]},{indices_disp[2]}), P={pressures_disp}\nStatus: {status_disp}"
        plotter_init_pv.add_text(status_txt_i_pv, position="upper_left", font=font_family_viz, font_size=10, color=text_color_viz, name="status_text_init")
        
        plotter_init_pv.camera.azimuth = 45
        plotter_init_pv.camera.elevation = 25
        plotter_init_pv.camera.zoom(1.3)
        print("(PyVista) 显示最后初始搜索结果。按Q关闭此窗口。")
        plotter_init_pv.show(cpos=None, auto_close=False) # cpos=None to use adjusted camera

        visualize_poses_with_open3d(
            tray_geometry_transform_mm=T_actual_tray_geometry_world, tray_axes_transform_mm=T_tray_coord_system_in_world,
            bottle_axes_transform_mm=T_object_coord_system_in_world, bottle_points_world_mm=object_points_global_static,
            bottle_colors_rgb_float=sampled_colors_float_o3d, bottle_obb_world=world_obb_object_global,
            tray_radius_mm=tray_radius, tray_height_mm=tray_height,
            window_title="Open3D - Initial Poses & Axes (Enhanced)"
        )
    else: print("警告：初始搜索结果不完整或物体未加载，无法进行PyVista和Open3D可视化。")

    # --- 贝叶斯优化 ---
    proceed_bo = bool(collected_initial_points) or N_INITIAL_POINTS_BO > 0
    if not collected_initial_points and N_INITIAL_POINTS_BO > 0: print("未通过初始搜索找到有效三点接触，但BO将尝试自行生成初始点。")
    elif not proceed_bo: print("没有初始点（通过搜索或BO设定），跳过优化。")
    res_bo = None
    if proceed_bo:
        n_rand_bo_call = N_INITIAL_POINTS_BO; x0_for_bo = None
        if collected_initial_points:
            x0_for_bo = collected_initial_points; n_rand_bo_call = max(0, N_INITIAL_POINTS_BO - len(x0_for_bo))
        print(f"\n开始贝叶斯优化 (提供{len(x0_for_bo) if x0_for_bo else 0}点,BO随机探索{n_rand_bo_call}点,总调用目标{N_CALLS_BO}次)")
        try:
            res_bo = gp_minimize(func=evaluate_grasp,dimensions=dimensions,acq_func="EI",n_calls=N_CALLS_BO,n_initial_points=n_rand_bo_call,x0=x0_for_bo,random_state=123,noise=0.01)
        except Exception as e: print(f"\nBO过程错: {e}"); traceback.print_exc()
        print("\n贝叶斯优化结束。")

    # --- PyVista 可视化最优结果 ---
    if res_bo:
        best_p_list_bo = res_bo.x; best_c_bo = res_bo.fun
        best_p_dict_bo = dict(zip(param_names,best_p_list_bo))
        print("\n使用最优参数重新评估最终状态...")
        # Re-evaluate to get gii and spread for printing, if not already stored from last call
        # This call also prints the detailed line
        final_c_check_bo = evaluate_grasp(best_p_list_bo) 
        
        # Extract GII and Spread for text display (assuming evaluate_grasp prints them or they can be retrieved)
        # For simplicity, we'll rely on the print from evaluate_grasp.
        # If evaluate_grasp returned a dict with these values, we'd use them here.
        # For now, the text will show the combined score.
        
        print("\n找到的最优参数:")
        for n,v in best_p_dict_bo.items(): print(f"  {n:<12} = {v:.4f}" if not n.startswith("pos_idx") else f"  {n:<12} = {int(v)}")
        print(f"优化找到成本: {best_c_bo:.4f}") # This is -(W_GII*gii + W_SPREAD*spread)
        print(f"重评成本 (与优化成本应一致): {final_c_check_bo:.4f}")
        # To print GII and Spread separately, they need to be returned by evaluate_grasp or recalculated
        # For now, we assume the print within evaluate_grasp is sufficient.

        print("\n使用最优参数生成最终可视化...")
        final_pressures_bo_viz = [best_p_dict_bo['p1'],best_p_dict_bo['p2'],best_p_dict_bo['p3']]
        final_chosen_indices_bo_viz = [int(best_p_dict_bo['pos_idx1']), int(best_p_dict_bo['pos_idx2']), int(best_p_dict_bo['pos_idx3'])]
        final_meshes_bo_viz = []; all_preds_ok_bo_viz = True
        if len(set(final_chosen_indices_bo_viz)) < 3:
            print("错误：BO最优参数的手指位置索引不唯一，无法生成有效可视化。"); all_preds_ok_bo_viz = False
        else:
            for i_f_bo_viz in range(3):
                disps_f_bo = predict_displacements_for_pressure(model_global,scaler_X_global,scaler_y_global,device_global,final_pressures_bo_viz[i_f_bo_viz])
                if disps_f_bo is None: all_preds_ok_bo_viz=False; break
                c1u_f_bo=initial_coords_ref_global+disps_f_bo; c2r_f_bo=initial_coords_ref_global+width_translation_vector_global; c2u_f_bo=c2r_f_bo+disps_f_bo
                sc1_f_bo=sort_points_spatially(c1u_f_bo); sc2_f_bo=sort_points_spatially(c2u_f_bo)
                if sc1_f_bo is None or sc2_f_bo is None: all_preds_ok_bo_viz=False; break
                verts_f_bo=np.vstack((sc1_f_bo,sc2_f_bo))
                if faces_np_global is None or faces_np_global.size==0: all_preds_ok_bo_viz=False; break
                try: mesh_r_f_bo=pv.PolyData(verts_f_bo,faces=faces_np_global)
                except Exception: all_preds_ok_bo_viz=False; break
                current_pos_index_bo = final_chosen_indices_bo_viz[i_f_bo_viz]
                ang_d_f_bo = current_pos_index_bo * (360.0 / N_FINGER_SLOTS)
                ang_r_f_bo=np.radians(ang_d_f_bo); rot_z_f_bo=create_rotation_matrix_z(ang_r_f_bo+np.pi/2.)
                pos_c_f_bo=np.array([best_p_dict_bo['r']*np.cos(ang_r_f_bo),best_p_dict_bo['r']*np.sin(ang_r_f_bo),0.])
                T3_f_bo=create_transformation_matrix_opt8(rot_z_f_bo,pos_c_f_bo)
                T_transform_finger_relative_to_tray_origin_bo = T3_f_bo @ T2_rotate_global @ T1_translate_global
                if T_pose_for_tray_display_and_finger_placement_global is None: all_preds_ok_bo_viz=False; break
                T_f_finger_bo_world = T_pose_for_tray_display_and_finger_placement_global @ T_transform_finger_relative_to_tray_origin_bo
                try:
                    mesh_tf_f_bo=mesh_r_f_bo.transform(T_f_finger_bo_world,inplace=False)
                    if mesh_tf_f_bo is None or mesh_tf_f_bo.n_points==0: all_preds_ok_bo_viz=False; break
                    mesh_tf_f_bo.clean(inplace=True)
                    if mesh_tf_f_bo.n_cells>0: mesh_tf_f_bo.compute_normals(cell_normals=True,point_normals=False,inplace=True,auto_orient_normals=True,non_manifold_traversal=False)
                    # 为手指网格增加厚度
                    try:
                        thickened_mesh_bo = mesh_tf_f_bo.thicken(thickness=FINGER_VIZ_THICKNESS, inplace=False)
                        final_meshes_bo_viz.append(thickened_mesh_bo)
                    except Exception as e_thicken:
                        print(f"警告: 在最终可视化中为手指 {i_f_bo_viz} 增加厚度失败: {e_thicken}")
                        final_meshes_bo_viz.append(mesh_tf_f_bo) # 出错则回退到原始网格
                except Exception: all_preds_ok_bo_viz=False; break
        if all_preds_ok_bo_viz and len(final_meshes_bo_viz)==3:
            # For text display, we need GII and Spread from the best parameters
            # Re-calculate spread for the best parameters
            angles_best_deg = sorted([(idx / N_FINGER_SLOTS) * 360.0 for idx in final_chosen_indices_bo_viz])
            s_a1_b, s_a2_b, s_a3_b = angles_best_deg[0], angles_best_deg[1], angles_best_deg[2]
            sep1_b = s_a2_b - s_a1_b; sep2_b = s_a3_b - s_a2_b; sep3_b = 360.0 - (s_a3_b - s_a1_b)
            min_sep_b = min(sep1_b, sep2_b, sep3_b)
            min_possible_min_sep_b = 360.0 / N_FINGER_SLOTS if N_FINGER_SLOTS > 0 else 0
            max_possible_min_sep_b = 120.0
            norm_metric_finger_spread_best = 0.0
            denominator_b = max_possible_min_sep_b - min_possible_min_sep_b
            if N_FINGER_SLOTS >=3 and denominator_b > 1e-6 :
                norm_metric_finger_spread_best = (min_sep_b - min_possible_min_sep_b) / denominator_b
                norm_metric_finger_spread_best = np.clip(norm_metric_finger_spread_best, 0.0, 1.0)
            elif N_FINGER_SLOTS >=3 and abs(min_sep_b - max_possible_min_sep_b) < 1e-6 :
                norm_metric_finger_spread_best = 1.0
            
            # We need GII for the best parameters. It's not directly returned by gp_minimize.
            # The 'final_c_check_bo' is -(W_GII*gii + W_SPREAD*spread).
            # If W_GII is not 0, we can estimate gii if spread is known.
            # Or, we need to modify evaluate_grasp to return these values or store them globally.
            # For now, we'll display the combined score.
            gii_display_val = "N/A" # Placeholder
            if final_c_check_bo < 0 and W_GII > 1e-6:
                 # Estimate GII, this is an approximation if W_SPREAD is also non-zero
                 gii_display_val = f"{(-final_c_check_bo - W_FINGER_SPREAD * norm_metric_finger_spread_best) / W_GII:.3f}"
            
            state_disp_bo=f"Score={-final_c_check_bo:.3f} (GII={gii_display_val}, Spread={norm_metric_finger_spread_best:.3f})"

            title_final_pv_viz = "PyVista - Optimized Grasp Configuration"
            plotter_theme_final = pv.themes.DocumentTheme() # Re-init theme for safety
            plotter_theme_final.font.family = font_family_viz
            plotter_theme_final.font.color = text_color_viz
            plotter_theme_final.font.size = 12
            plotter_theme_final.font.label_size = 10
            plotter_theme_final.background = pv.Color(background_color_viz)

            plotter_final_pv = pv.Plotter(window_size=[1000,800], theme=plotter_theme_final, title=title_final_pv_viz)
            plotter_final_pv.enable_anti_aliasing('msaa', multi_samples=8)
            plotter_final_pv.enable_parallel_projection()
            # plotter_final_pv.enable_specular_lighting() # REMOVED - enable_lightkit should handle this
            plotter_final_pv.specular_power = 10.0 # Control shininess
            plotter_final_pv.remove_all_lights(); plotter_final_pv.enable_lightkit()

            if plotter_final_pv.renderer.cube_axes_actor is not None and hasattr(plotter_final_pv.renderer.cube_axes_actor, 'GetXAxisCaptionActor2D'):
                actor = plotter_final_pv.renderer.cube_axes_actor
                # 新增：不显示轴标题 (X, Y, Z 文字)
                actor.SetXTitle("") # 设置X轴标题为空字符串
                actor.SetYTitle("") # 设置Y轴标题为空字符串
                actor.SetZTitle("") # 设置Z轴标题为空字符串
                for i in range(3):
                    cap_prop = actor.GetCaptionTextProperty(i)
                    if cap_prop:
                        cap_prop.SetFontFamily(vtk.VTK_TIMES); cap_prop.SetFontSize(10); cap_prop.SetColor(0,0,0); cap_prop.SetBold(0)
                    else:
                        title_prop = actor.GetTitleTextProperty(i)
                        if title_prop:
                           title_prop.SetFontFamily(vtk.VTK_TIMES); title_prop.SetFontSize(10); title_prop.SetColor(0,0,0); title_prop.SetBold(0)
                actor.SetXTitle("X"); actor.SetYTitle("Y"); actor.SetZTitle("Z")
                actor.GetProperty().SetLineWidth(1.0)

            plotter_final_pv.add_mesh(tray_pv, color=pv.Color(tray_color_viz, opacity=200), smooth_shading=True, name='tray_final_pv')
            if object_mesh_global_static:
                if 'colors' in object_mesh_global_static.point_data and object_mesh_global_static.point_data['colors'] is not None:
                    plotter_final_pv.add_mesh(object_mesh_global_static,scalars='colors',rgba=True,style='points',point_size=3.5,render_points_as_spheres=True,name='obj_final_pv_c')
                else:
                    plotter_final_pv.add_mesh(object_mesh_global_static,color=object_point_color_viz,style='points',point_size=3.5,render_points_as_spheres=True,name='obj_final_pv_d')
            # if pv_obb_object_mesh_global is not None:
                # plotter_final_pv.add_mesh(pv_obb_object_mesh_global, style='wireframe', color=object_obb_color_viz, line_width=1.5, name='object_obb_final_pv')
            if show_axes: plotter_final_pv.add_axes_at_origin(line_width=1.5, labels_off=True)
            for i_fv_bo,m_fv_bo in enumerate(final_meshes_bo_viz):
                if m_fv_bo:
                    plotter_final_pv.add_mesh(
                        m_fv_bo,
                        color=finger_color_viz,
                        style='surface',
                        opacity=0.95,
                        smooth_shading=True,
                        show_edges=True,       # 新增
                        edge_color='dimgray',  # 新增
                        line_width=0.5,        # 新增
                        name=f'f_final_pv_{i_fv_bo}'
                    )
            params_txt_bo=f"r={best_p_dict_bo['r']:.2f}, P1={best_p_dict_bo['p1']:.0f}, P2={best_p_dict_bo['p2']:.0f}, P3={best_p_dict_bo['p3']:.0f}\nIndices=({final_chosen_indices_bo_viz[0]},{final_chosen_indices_bo_viz[1]},{final_chosen_indices_bo_viz[2]})"
            plotter_final_pv.add_text(f"Optimal Grasp:\n{params_txt_bo}\n{state_disp_bo}",position="upper_left", font=font_family_viz, font_size=10, color=text_color_viz, name="status_text_final")
            
            plotter_final_pv.camera.azimuth = 45
            plotter_final_pv.camera.elevation = 25
            plotter_final_pv.camera.zoom(1.3)
            print("\n(PyVista) 显示最优抓取配置。按Q退出此窗口.")
            plotter_final_pv.show(cpos=None, auto_close=False)

            visualize_poses_with_open3d(
                tray_geometry_transform_mm=T_actual_tray_geometry_world, tray_axes_transform_mm=T_tray_coord_system_in_world,
                bottle_axes_transform_mm=T_object_coord_system_in_world, bottle_points_world_mm=object_points_global_static,
                bottle_colors_rgb_float=sampled_colors_float_o3d, bottle_obb_world=world_obb_object_global,
                tray_radius_mm=tray_radius, tray_height_mm=tray_height,
                window_title="Open3D - Optimal Poses & Axes (Enhanced)"
            )
        else: print("未能为最优参数生成手指网格，无法进行PyVista最终可视化。")
    else: print("\nBO未成功或无结果，无法显示最优。")
    print("\n程序结束。")
