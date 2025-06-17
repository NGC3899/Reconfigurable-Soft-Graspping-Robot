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

# --- 打印版本信息 ---
try:
    print(f"Opt_9.txt (PyVista Interaction & O3D Tray Wireframe) - OBJECT LONGEST EDGE ALIGNED TO LOCAL X")
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
    from skopt.space import Real
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
HIGH_RES_OBJECT_FILENAME = "Graphene_Bottle.ply"
# HIGH_RES_OBJECT_FILENAME = "Bird_Model.ply"

RELATIVE_POSE_FILE_PATH = os.path.join(GRASP_OUTPUTS_BASE_PATH, RELATIVE_POSE_FILENAME)
HIGH_RES_OBJECT_PLY_PATH = os.path.join(HIGH_RES_OBJECT_DESKTOP_PATH, HIGH_RES_OBJECT_FILENAME)

# --- 4. 配置参数 (来自 Opt_9.txt 源文件) ---
tray_radius = 60.0
tray_height = 1.0
tray_center = np.array([0.0, 0.0, -tray_height / 2.0]) # Original tray center definition
finger_width = 10.0
TARGET_POINT_COUNT_FOR_SIM = 2000
show_axes = True # This controls plotter.add_axes_at_origin()
finger_color = 'lightcoral'
tray_color = 'tan'
object_point_color = 'blue'
object_obb_color_pv = 'green' # PyVista OBB 颜色
object_obb_color_o3d = (0.1, 0.9, 0.1) # Open3D OBB 颜色 (R,G,B)
colliding_point_color = 'magenta'
intersection_point_color = 'yellow'
overlap_point_color = 'orange'
collision_threshold = 1.0
overlap_threshold = 1e-4
friction_coefficient = 0.5
eigenvalue_threshold = 1e-6
max_pressure = 40000.0
NUM_INITIAL_SEARCHES = 1
PRESSURE_STEP_INIT_SEARCH = 100.0
N_CALLS_BO = 20
N_INITIAL_POINTS_BO = 15
R_BOUNDS = (tray_radius * 0.3, tray_radius * 0.95)
P_BOUNDS = (0.0, max_pressure)
ANGLE_OFFSET_BOUNDS = (0.0, 120.0)
contact_marker_radius = 1.0
contact_normal_length = 5.0
contact_plane_size = 4.0
show_finger_normals = True
finger_normal_vis_scale = 1.0
finger_normal_color = 'gray'
contact_normal_color = 'black'
OBJECT_SCALE_FACTOR = 0.8  # 新增: 物体点云的线性缩放因子 (1.0 表示不缩放)

param_names = ['r', 'p1', 'p2', 'p3', 'angle_offset']
dimensions = [
    Real(name='r', low=R_BOUNDS[0], high=R_BOUNDS[1]),
    Real(name='p1', low=P_BOUNDS[0], high=P_BOUNDS[1]),
    Real(name='p2', low=P_BOUNDS[0], high=P_BOUNDS[1]),
    Real(name='p3', low=P_BOUNDS[0], high=P_BOUNDS[1]),
    Real(name='angle_offset', low=ANGLE_OFFSET_BOUNDS[0], high=ANGLE_OFFSET_BOUNDS[1])
]

# --- 5. 辅助函数 (来自 Opt_9.txt 源文件，包含之前的语法修正) ---
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

# 新增辅助函数: 计算两个向量之间的旋转矩阵
def get_rotation_matrix_between_vectors(vec1, vec2):
    """ Get the rotation matrix that aligns vec1 to vec2. """
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b) # Cosine of the angle
    s = np.linalg.norm(v) # Sine of the angle

    if s < 1e-9: # Vectors are collinear
        if c > 0: # Same direction (c is close to 1)
            return np.identity(3)
        else: # Opposite direction (c is close to -1), 180-degree rotation
            # Find an arbitrary axis orthogonal to 'a'
            if np.abs(a[0]) > 0.9 or np.abs(a[1]) > 0.9 : # if a is mostly along X or Y
                axis_ortho = np.array([0.0, 0.0, 1.0]) # use Z
            else: # if a is mostly along Z or other
                axis_ortho = np.array([1.0, 0.0, 0.0]) # use X
            
            # Ensure axis_ortho is indeed orthogonal
            if np.linalg.norm(np.cross(a, axis_ortho)) < 1e-6: # If a was parallel to chosen axis_ortho
                axis_ortho = np.array([0.0, 1.0, 0.0]) # Try Y axis
            
            # If still parallel (e.g. a was [0,1,0]), then cross product will be zero.
            # This case should be rare if a is normalized.
            # A robust way for 180 deg: return 2 * np.outer(a,a) - np.identity(3) if a is unit
            # Or use the existing create_rotation_matrix
            return create_rotation_matrix(axis_ortho, np.pi) # Rotate 180 deg around an orthogonal axis

    # Rodrigues' rotation formula
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    r = np.identity(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return r

# 全局变量定义
initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, \
device_global = [None]*5
object_points_global_static, object_centroid_global_static, object_mesh_global_static = [None]*3
world_obb_object_global = None # 新增: 全局存储物体的OBB (Open3D对象)
pv_obb_object_mesh_global = None # 新增: 全局存储物体的OBB (PyVista网格)
num_object_points_global_static, faces_np_global, width_translation_vector_global = 0, None, None
T1_translate_global, T2_rotate_global = None, None
T_pose_for_tray_display_and_finger_placement_global = None # 用于在手指放置函数中访问对调后的托盘几何位姿

@use_named_args(dimensions=dimensions)
def evaluate_grasp(r, p1, p2, p3, angle_offset):
    global initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global, \
           object_points_global_static, object_centroid_global_static, num_object_points_global_static, \
           faces_np_global, width_translation_vector_global, T1_translate_global, T2_rotate_global, \
           T_pose_for_tray_display_and_finger_placement_global # 访问全局的、对调后的托盘几何位姿

    print_eval_string = f"  评估: r={r:.4f}, p1={p1:.1f}, p2={p2:.1f}, p3={p3:.1f}, offset={angle_offset:.2f}°"
    print(print_eval_string, end="")

    current_pressures = [p1, p2, p3]
    current_placement_radius = r
    deformed_finger_meshes_world_this_eval = []
    mesh_generation_successful = True
    for i in range(3):
        displacements_matrix = predict_displacements_for_pressure(model_global, scaler_X_global, scaler_y_global, device_global, current_pressures[i])
        if displacements_matrix is None:
            mesh_generation_successful = False
            break
        deformed_curve1_ref_unordered = initial_coords_ref_global + displacements_matrix
        curve2_ref = initial_coords_ref_global + width_translation_vector_global
        deformed_curve2_ref_unordered = curve2_ref + displacements_matrix
        sorted_deformed_curve1_ref = sort_points_spatially(deformed_curve1_ref_unordered)
        sorted_deformed_curve2_ref = sort_points_spatially(deformed_curve2_ref_unordered)
        if sorted_deformed_curve1_ref is None or sorted_deformed_curve2_ref is None:
            mesh_generation_successful = False
            break
        sorted_deformed_vertices_ref = np.vstack((sorted_deformed_curve1_ref, sorted_deformed_curve2_ref))
        if faces_np_global is None or faces_np_global.size == 0:
            mesh_generation_successful = False
            break
        try:
            deformed_mesh_ref = pv.PolyData(sorted_deformed_vertices_ref, faces=faces_np_global)
        except Exception:
            mesh_generation_successful = False
            break
        base_angle_deg = [0, 120, 240][i]
        current_angle_deg = base_angle_deg + angle_offset
        angle_rad = np.radians(current_angle_deg)
        rot_angle_z_placing = angle_rad + np.pi / 2.0
        rot_z_placing = create_rotation_matrix_z(rot_angle_z_placing)
        target_pos_on_circle = np.array([ current_placement_radius * np.cos(angle_rad), current_placement_radius * np.sin(angle_rad), 0.0 ])
        T3_place = create_transformation_matrix_opt8(rot_z_placing, target_pos_on_circle)

        # 手指相对于托盘自身原点的变换
        T_transform_finger_relative_to_tray_origin = T3_place @ T2_rotate_global @ T1_translate_global
        if T_pose_for_tray_display_and_finger_placement_global is None:
            print("错误: evaluate_grasp - 全局托盘几何位姿未设置!")
            mesh_generation_successful = False; break
        # 将手指变换到世界坐标系（基于新的托盘几何位姿）
        T_final_finger_world = T_pose_for_tray_display_and_finger_placement_global @ T_transform_finger_relative_to_tray_origin

        try:
            final_transformed_finger_mesh = deformed_mesh_ref.transform(T_final_finger_world, inplace=False)
            if final_transformed_finger_mesh is None or final_transformed_finger_mesh.n_points == 0:
                mesh_generation_successful = False
                break
            final_transformed_finger_mesh.clean(inplace=True)
            if final_transformed_finger_mesh.n_cells > 0:
                final_transformed_finger_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True, auto_orient_normals=True, non_manifold_traversal=False)
        except Exception:
            mesh_generation_successful = False
            break
        deformed_finger_meshes_world_this_eval.append(final_transformed_finger_mesh)

    if not mesh_generation_successful:
        print(f" -> r={r:.2f},P={np.round(current_pressures).astype(int)},off={angle_offset:.1f} -> 网格生成失败, Cost=10.0")
        return 10.0

    object_point_status = ['Non-Contact'] * num_object_points_global_static
    closest_contact_per_finger_eval = [None] * 3
    min_dist_per_finger_eval = [float('inf')] * 3
    finger_dot_products = [[] for _ in range(3)]
    has_overlap_eval = False
    num_contact_points = 0
    dot_prod_tolerance = 1e-6

    if object_points_global_static is None or num_object_points_global_static == 0:
        print(f" -> r={r:.2f},P={np.round(current_pressures).astype(int)},off={angle_offset:.1f} -> 物体点云为空, Cost=10.0")
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
            except Exception:
                continue
        if finger_idx_for_this_pt != -1:
            dist = closest_dist_for_this_pt
            normal_closest = normal_for_this_pt
            pt_on_mesh_closest = pt_on_mesh_for_this_pt
            if dist < overlap_threshold:
                object_point_status[obj_pt_idx] = 'Overlap'
                has_overlap_eval = True
            elif dist < collision_threshold:
                object_point_status[obj_pt_idx] = 'Contact'
                num_contact_points += 1
                if normal_closest is not None and LA.norm(normal_closest) > 1e-9:
                    vector_to_point = obj_point - pt_on_mesh_closest
                    if LA.norm(vector_to_point) > 1e-9:
                        dot_prod = np.dot(vector_to_point / LA.norm(vector_to_point), normal_closest / LA.norm(normal_closest))
                        finger_dot_products[finger_idx_for_this_pt].append(dot_prod)

    final_contacts_for_gii_eval = [info for info in closest_contact_per_finger_eval if info is not None]
    grasp_state_eval = "No Contact"
    cost = 3.0
    if has_overlap_eval:
        grasp_state_eval = "Overlap"
        cost = 5.0
    else:
        finger_intersects = [False] * 3
        for i in range(3):
            if finger_dot_products[i]:
                has_positive_dp = any(dp > dot_prod_tolerance for dp in finger_dot_products[i])
                has_negative_dp = any(dp < -dot_prod_tolerance for dp in finger_dot_products[i])
                if has_positive_dp and has_negative_dp: finger_intersects[i] = True

        if any(finger_intersects):
            grasp_state_eval = "Intersection"
            cost = 4.0
        elif num_contact_points > 0:
            grasp_state_eval = "Contact"
            num_gii_contacts = len(final_contacts_for_gii_eval)
            if num_gii_contacts >= 2:
                gii = calculate_gii_multi_contact(final_contacts_for_gii_eval, object_centroid_global_static, friction_coefficient, eigenvalue_threshold)
                if gii is not None and gii > 1e-9:
                    cost = -gii
                else:
                    cost = 2.0 # GII calculation failed or result too small
            else: # Not enough contacts for GII
                cost = 1.0 + ( (2.0 - num_gii_contacts) * 0.3 ) # Penalty for fewer than 2 GII contacts
        # else: No contact, cost remains 3.0

    eval_params_str = f"r={r:.2f},P={np.array([p1,p2,p3]).round(0).astype(int)},off={angle_offset:.1f}"
    if grasp_state_eval == "Contact" and cost < 0: # Successful grasp with GII
        print(f" -> {eval_params_str}, State={grasp_state_eval}, GII={-cost:.4f}, Cost={cost:.4f}")
    else:
        print(f" -> {eval_params_str}, State={grasp_state_eval}, NCont={num_contact_points}, NGIIC={len(final_contacts_for_gii_eval)}, Cost={cost:.4f}")
    return cost


def find_initial_grasp(initial_r, initial_angle_offset, pressure_step, max_pressure_init):
    print(f"\n--- 开始寻找初始接触点 (r = {initial_r:.2f}, offset = {initial_angle_offset:.1f}°) ---")
    global initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global, \
           object_points_global_static, num_object_points_global_static, faces_np_global, \
           width_translation_vector_global, T1_translate_global, T2_rotate_global, \
           T_pose_for_tray_display_and_finger_placement_global # 访问全局的、对调后的托盘几何位姿

    current_pressures_init = np.array([0.0, 0.0, 0.0])
    finger_contact_achieved_init = [False, False, False]
    init_iter_count = 0
    last_meshes = [None] * 3
    last_point_status = ['Non-Contact'] * num_object_points_global_static if num_object_points_global_static > 0 else []
    last_closest_contacts = [None] * 3
    dot_prod_tolerance = 1e-6

    # Max iterations to prevent infinite loop if pressures max out without contact
    max_iterations = int( (max_pressure_init / pressure_step) * 3 * 1.5) + 10 # Heuristic limit

    while True:
        init_iter_count += 1
        if init_iter_count > max_iterations:
            print("--- 初始点搜索: 达到最大迭代次数 ---")
            return { 'status': 'IterationLimit', 'r': initial_r, 'angle_offset': initial_angle_offset, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }

        pressure_changed_init = False
        print(f"  初始点搜索 Iter {init_iter_count}: P=[{current_pressures_init[0]:.0f}, {current_pressures_init[1]:.0f}, {current_pressures_init[2]:.0f}]")
        for i in range(3):
            if not finger_contact_achieved_init[i] and current_pressures_init[i] < max_pressure_init:
                current_pressures_init[i] += pressure_step
                current_pressures_init[i] = min(current_pressures_init[i], max_pressure_init)
                pressure_changed_init = True
        if not pressure_changed_init and all(finger_contact_achieved_init): # All contacts made, no pressure change needed
             pass # Proceed to check this state
        elif not pressure_changed_init and not all(finger_contact_achieved_init): # No pressure change, but not all contacts made (likely max pressure on some)
            print("--- 初始点搜索: 压力达到上限或无变化，但未实现三指接触 ---")
            return { 'status': 'MaxPressureReachedOrNoChange', 'r': initial_r, 'angle_offset': initial_angle_offset, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }


        deformed_finger_meshes_init = [None] * 3
        valid_preds = True
        for i in range(3):
            displacements_matrix = predict_displacements_for_pressure(model_global, scaler_X_global, scaler_y_global, device_global, current_pressures_init[i])
            if displacements_matrix is None:
                valid_preds = False
                break
            deformed_curve1_ref_unordered = initial_coords_ref_global + displacements_matrix
            curve2_ref = initial_coords_ref_global + width_translation_vector_global
            deformed_curve2_ref_unordered = curve2_ref + displacements_matrix
            sorted_deformed_curve1_ref = sort_points_spatially(deformed_curve1_ref_unordered)
            sorted_deformed_curve2_ref = sort_points_spatially(deformed_curve2_ref_unordered)
            if sorted_deformed_curve1_ref is None or sorted_deformed_curve2_ref is None:
                valid_preds = False
                break
            sorted_deformed_vertices_ref = np.vstack((sorted_deformed_curve1_ref, sorted_deformed_curve2_ref))
            if faces_np_global is None or faces_np_global.size == 0:
                valid_preds = False
                break
            try:
                deformed_mesh_ref = pv.PolyData(sorted_deformed_vertices_ref, faces=faces_np_global)
            except Exception:
                valid_preds = False
                break
            base_angle_deg = [0, 120, 240][i]
            current_angle_deg = base_angle_deg + initial_angle_offset
            angle_rad = np.radians(current_angle_deg)
            rot_angle_z_placing = angle_rad + np.pi / 2.0
            rot_z_placing = create_rotation_matrix_z(rot_angle_z_placing)
            target_pos_on_circle = np.array([ initial_r * np.cos(angle_rad), initial_r * np.sin(angle_rad), 0.0 ])
            T3_place = create_transformation_matrix_opt8(rot_z_placing, target_pos_on_circle)

            T_transform_finger_relative_to_tray_origin = T3_place @ T2_rotate_global @ T1_translate_global
            if T_pose_for_tray_display_and_finger_placement_global is None:
                print("错误: find_initial_grasp - 全局托盘几何位姿未设置!")
                valid_preds = False; break
            T_final_finger_world = T_pose_for_tray_display_and_finger_placement_global @ T_transform_finger_relative_to_tray_origin

            try:
                final_transformed_finger_mesh = deformed_mesh_ref.transform(T_final_finger_world, inplace=False)
                if final_transformed_finger_mesh is None or final_transformed_finger_mesh.n_points == 0:
                    valid_preds = False
                    break
                final_transformed_finger_mesh.clean(inplace=True)
                if final_transformed_finger_mesh.n_cells > 0:
                    final_transformed_finger_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True, auto_orient_normals=True, non_manifold_traversal=False)
                deformed_finger_meshes_init[i] = final_transformed_finger_mesh
            except Exception:
                valid_preds = False
                break
        if not valid_preds:
            print("  初始点搜索: 预测或网格处理失败。")
            return { 'status': 'PredictionFailed', 'r': initial_r, 'angle_offset': initial_angle_offset, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }

        last_meshes = [mesh.copy() if mesh is not None else None for mesh in deformed_finger_meshes_init] # Store copies

        has_overlap_init = False
        num_contact_points_init = 0
        current_closest_contacts = [None] * 3
        current_min_dists = [float('inf')] * 3
        current_point_status = ['Non-Contact'] * num_object_points_global_static if num_object_points_global_static > 0 else []
        finger_dot_products_init = [[] for _ in range(3)]
        contacting_fingers_indices_this_iter = set()

        if object_points_global_static is not None and num_object_points_global_static > 0:
            for obj_pt_idx, obj_point in enumerate(object_points_global_static):
                closest_dist_for_this_pt = float('inf')
                finger_idx_for_this_pt = -1
                normal_for_this_pt = None
                pt_on_mesh_for_this_pt = None
                for finger_idx, finger_mesh in enumerate(deformed_finger_meshes_init):
                    if finger_mesh is None or finger_mesh.n_cells == 0: continue
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
                        if dist < collision_threshold and dist < current_min_dists[finger_idx]: # For GII contact info
                            current_min_dists[finger_idx] = dist
                            current_normal_contact = None
                            if has_normals and closest_cell_id < len(finger_mesh.cell_data['Normals']):
                                current_normal_contact = finger_mesh.cell_normals[closest_cell_id]
                            if current_normal_contact is not None:
                                current_closest_contacts[finger_idx] = (dist, obj_pt_idx, finger_idx, closest_cell_id, pt_on_mesh_candidate, current_normal_contact)
                    except Exception:
                        continue
                if finger_idx_for_this_pt != -1:
                    dist_final_for_pt = closest_dist_for_this_pt
                    normal_final_for_pt = normal_for_this_pt
                    pt_on_mesh_final_for_pt = pt_on_mesh_for_this_pt

                    if dist_final_for_pt < overlap_threshold:
                        current_point_status[obj_pt_idx] = 'Overlap'
                        has_overlap_init = True
                        # break # from obj_pt_idx loop if overlap is found
                    elif dist_final_for_pt < collision_threshold:
                        current_point_status[obj_pt_idx] = 'Contact'
                        num_contact_points_init += 1
                        contacting_fingers_indices_this_iter.add(finger_idx_for_this_pt)
                        if normal_final_for_pt is not None and LA.norm(normal_final_for_pt) > 1e-9:
                            vector_to_point = obj_point - pt_on_mesh_final_for_pt
                            if LA.norm(vector_to_point) > 1e-9:
                                dot_prod = np.dot(vector_to_point/LA.norm(vector_to_point), normal_final_for_pt/LA.norm(normal_final_for_pt))
                                finger_dot_products_init[finger_idx_for_this_pt].append(dot_prod)
                if has_overlap_init: break # Break from object point loop

        if has_overlap_init:
            print("  初始点搜索: 检测到重叠。")
            return { 'status': 'Overlap', 'r': initial_r, 'angle_offset': initial_angle_offset, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': current_point_status, 'contact_info': current_closest_contacts }

        last_point_status = current_point_status # Update last known good status if no overlap
        last_closest_contacts = current_closest_contacts

        finger_intersects_init = [False] * 3
        if num_contact_points_init > 0:
            for i in range(3):
                if finger_dot_products_init[i]:
                    has_pos_dp = any(dp > dot_prod_tolerance for dp in finger_dot_products_init[i])
                    has_neg_dp = any(dp < -dot_prod_tolerance for dp in finger_dot_products_init[i])
                    if has_pos_dp and has_neg_dp: finger_intersects_init[i] = True
        if any(finger_intersects_init):
            print("  初始点搜索: 检测到交叉穿透。")
            return { 'status': 'Intersection', 'r': initial_r, 'angle_offset': initial_angle_offset, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }

        # Update overall contact achieved status
        for i in range(3):
            if i in contacting_fingers_indices_this_iter:
                finger_contact_achieved_init[i] = True

        print(f"  初始点搜索: 接触状态={list(finger_contact_achieved_init)}")

        if all(finger_contact_achieved_init):
            print(f"--- 成功找到初始接触点组合 (r={initial_r:.2f}, offset={initial_angle_offset:.1f}°, P={current_pressures_init.round(0)}) ---")
            return { 'status': 'FoundInitialPoint', 'r': initial_r, 'angle_offset': initial_angle_offset, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }

        # If no pressure was changed in this iteration (e.g., all non-contacting fingers are at max_pressure)
        # and not all contacts are achieved, then we can't improve further.
        if not pressure_changed_init and not all(finger_contact_achieved_init):
            print("--- 初始点搜索: 压力达到上限或无变化，但未实现三指接触 (循环终止检查) ---")
            return { 'status': 'MaxPressureReachedOrNoChangeLoopEnd', 'r': initial_r, 'angle_offset': initial_angle_offset, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }

    # Fallback, should be caught by iteration limit or other conditions
    return { 'status': 'LoopEndUnknown', 'r': initial_r, 'angle_offset': initial_angle_offset, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }


# --- 辅助函数: 使用Open3D可视化位姿关系 (MODIFIED SIGNATURE) ---
def visualize_poses_with_open3d(
                                tray_geometry_transform_mm, # 用于变换托盘几何
                                tray_axes_transform_mm,     # 用于变换托盘坐标轴
                                bottle_axes_transform_mm,   # 用于变换瓶子坐标轴 (与瓶子点云的世界位姿一致)
                                bottle_points_world_mm,     # 瓶子点云（已在世界坐标系）
                                bottle_colors_rgb_float,
                                bottle_obb_world,           # 新增: 物体的OBB (Open3D对象)
                                tray_radius_mm, tray_height_mm,
                                window_title="Open3D Relative Pose Visualization"):
    print(f"\n--- 打开 Open3D 窗口: {window_title} ---")
    geometries = []
    # 1. 托盘 (线框 Cylinder) - 使用其几何变换
    # 创建标准的、Z轴为高度轴的圆柱体
    o3d_tray_canonical_centered = o3d.geometry.TriangleMesh.create_cylinder(radius=tray_radius_mm, height=tray_height_mm, resolution=20, split=4)
    o3d_tray_canonical_centered.translate(np.array([0,0,-tray_height_mm/2.0])) # 将其几何中心移至原点
    o3d_tray_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_tray_canonical_centered)
    o3d_tray_wireframe.transform(tray_geometry_transform_mm) # 应用最终的几何变换
    o3d_tray_wireframe.paint_uniform_color(pv.Color(tray_color).float_rgb)
    geometries.append(o3d_tray_wireframe)

    # 2. 瓶子 (PointCloud) - 点云已在世界坐标系
    o3d_bottle_pcd = o3d.geometry.PointCloud()
    o3d_bottle_pcd.points = o3d.utility.Vector3dVector(bottle_points_world_mm)
    if bottle_colors_rgb_float is not None and len(bottle_colors_rgb_float) == len(bottle_points_world_mm):
        o3d_bottle_pcd.colors = o3d.utility.Vector3dVector(bottle_colors_rgb_float)
    else:
        o3d_bottle_pcd.paint_uniform_color(pv.Color(object_point_color).float_rgb)
    geometries.append(o3d_bottle_pcd)

    # 新增: 添加物体的OBB
    if bottle_obb_world is not None:
        geometries.append(bottle_obb_world) # OBB对象已包含颜色

    # 3. 托盘坐标系 - 使用其坐标系变换
    tray_axes_size = tray_radius_mm * 0.7
    o3d_tray_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=tray_axes_size, origin=[0,0,0])
    o3d_tray_axes.transform(tray_axes_transform_mm)
    geometries.append(o3d_tray_axes)

    # 4. 瓶子坐标系 - 使用其坐标系变换
    if bottle_points_world_mm.shape[0] > 0:
        temp_bottle_pcd_for_bbox = o3d.geometry.PointCloud()
        temp_bottle_pcd_for_bbox.points = o3d.utility.Vector3dVector(bottle_points_world_mm)
        bbox_bottle = temp_bottle_pcd_for_bbox.get_axis_aligned_bounding_box()
        diag_len = LA.norm(bbox_bottle.get_max_bound() - bbox_bottle.get_min_bound())
        bottle_axes_size = diag_len * 0.25 if diag_len > 1e-2 else tray_axes_size * 0.8
    else:
        bottle_axes_size = tray_axes_size * 0.8
    bottle_axes_size = max(bottle_axes_size, 5.0)

    o3d_bottle_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=bottle_axes_size, origin=[0,0,0])
    o3d_bottle_axes.transform(bottle_axes_transform_mm)
    geometries.append(o3d_bottle_axes)

    world_axes_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=tray_radius_mm * 1.2, origin=[0,0,0]) # World frame
    geometries.append(world_axes_o3d)

    o3d.visualization.draw_geometries(geometries, window_name=window_title, width=800, height=600)
    print(f"--- 关闭 Open3D 窗口: {window_title} ---")


# --- 主脚本 ---
if __name__ == '__main__':
    # --- A. 加载GraspNet位姿 ---
    T_gn_gripper_TO_gn_object_meters = load_transformation_matrix_from_txt(RELATIVE_POSE_FILE_PATH)
    if T_gn_gripper_TO_gn_object_meters is None: sys.exit("错误：未能加载物体相对于GraspNet夹爪的位姿。")
    print("\n已加载“物体在GraspNet夹爪中的位姿” (T_gn_gripper_TO_gn_object_meters):\n", T_gn_gripper_TO_gn_object_meters)

    # --- B. 定义托盘原始位姿 (基于 tray_center) ---
    T_original_tray_pose_world = create_transformation_matrix_opt8(np.identity(3), tray_center)
    print("\n计算得到的“原始托盘”（圆形铁盘）的位姿 (T_original_tray_pose_world):\n", T_original_tray_pose_world)

    # --- C. 计算逆相对位姿 ---
    T_gn_object_TO_gn_gripper_meters = np.linalg.inv(T_gn_gripper_TO_gn_object_meters)
    print("\n“GraspNet夹爪”在“GraspNet物体”中的位姿 (T_gn_object_TO_gn_gripper_meters):\n", T_gn_object_TO_gn_gripper_meters)

    # --- D. 单位转换 (物体相对于托盘的位姿) ---
    T_imported_obj_relative_to_tray_mm = copy.deepcopy(T_gn_object_TO_gn_gripper_meters)
    T_imported_obj_relative_to_tray_mm[:3, 3] *= 1000.0
    print("\n转换单位后：“导入物体”在“托盘”中的目标相对位姿 (T_imported_obj_relative_to_tray_mm):\n", T_imported_obj_relative_to_tray_mm)

    # --- E. 计算导入物体在场景中的“原始”最终目标位姿 (相对于原始托盘位姿) ---
    T_original_object_pose_world = T_original_tray_pose_world @ T_imported_obj_relative_to_tray_mm
    print("\n计算得到的“导入物体”在Opt_9场景中的“原始”最终目标位姿 (T_original_object_pose_world):\n", T_original_object_pose_world)

    # --- X. 定义对调后的位姿 及 托盘自身朝向修正 ---
    # 托盘的 *坐标系* 将使用物体原始的位姿
    T_tray_coord_system_in_world = T_original_object_pose_world
    # 物体点云的 *坐标系* 将使用托盘原始的位姿
    T_object_coord_system_in_world = T_original_tray_pose_world # 这是物体坐标系在世界中的目标位姿

    # 创建一个绕Y轴旋转+90度的变换矩阵，用于改变圆柱体自身的朝向
    angle_rad_fix = np.pi / 2  # 绕Y轴逆时针旋转90度
    cos_fix = np.cos(angle_rad_fix)
    sin_fix = np.sin(angle_rad_fix)
    T_local_orientation_fix_for_tray_geometry = np.array([
        [cos_fix, 0, sin_fix, 0],
        [0,       1,       0, 0],
        [-sin_fix,0, cos_fix, 0],
        [0,       0,       0, 1]
    ])
    T_actual_tray_geometry_world = T_tray_coord_system_in_world @ T_local_orientation_fix_for_tray_geometry
    T_pose_for_tray_display_and_finger_placement_global = T_actual_tray_geometry_world

    print("\n--- 位姿对调及托盘朝向修正后 ---")
    print("  托盘坐标系在世界中的位姿 (T_tray_coord_system_in_world):\n", T_tray_coord_system_in_world)
    print("  托盘几何体在世界中的位姿 (T_actual_tray_geometry_world):\n", T_actual_tray_geometry_world)
    print("  物体点云坐标系在世界中的位姿 (T_object_coord_system_in_world):\n", T_object_coord_system_in_world)


    # --- F. 加载高精度物体点云并进行局部朝向修正 ---
    print(f"\n加载高精度物体点云: {HIGH_RES_OBJECT_PLY_PATH}")
    if not os.path.exists(HIGH_RES_OBJECT_PLY_PATH): sys.exit(f"错误: {HIGH_RES_OBJECT_PLY_PATH} 未找到")
    try:
        high_res_object_o3d = o3d.io.read_point_cloud(HIGH_RES_OBJECT_PLY_PATH)
        if not high_res_object_o3d.has_points(): sys.exit(f"错误: {HIGH_RES_OBJECT_PLY_PATH} 为空")
    except Exception as e: sys.exit(f"加载 {HIGH_RES_OBJECT_PLY_PATH} 错: {e}")
    current_high_res_points_mm_orig_frame = np.asarray(high_res_object_o3d.points)
    print(f"加载的 '{HIGH_RES_OBJECT_FILENAME}' 含 {len(current_high_res_points_mm_orig_frame)} 点 (假定单位毫米).")

    # 应用物体点云缩放
    if OBJECT_SCALE_FACTOR != 1.0:
        print(f"  应用缩放因子 {OBJECT_SCALE_FACTOR} 到物体点云的原始坐标。")
        current_high_res_points_mm_orig_frame = current_high_res_points_mm_orig_frame * OBJECT_SCALE_FACTOR

    centroid_high_res_mm = np.mean(current_high_res_points_mm_orig_frame, axis=0)
    points_high_res_centered_mm = current_high_res_points_mm_orig_frame - centroid_high_res_mm

    # --- X.A. 新增: 计算局部OBB并对齐最长边到局部X轴 ---
    print("\n--- 对齐物体局部点云的最长OBB边到其局部X轴 ---")
    points_high_res_centered_aligned_mm = points_high_res_centered_mm # 默认不旋转
    if points_high_res_centered_mm.shape[0] > 0:
        local_pcd_for_alignment = o3d.geometry.PointCloud()
        local_pcd_for_alignment.points = o3d.utility.Vector3dVector(points_high_res_centered_mm)
        try:
            local_obb = local_pcd_for_alignment.get_oriented_bounding_box()

            longest_extent_idx = np.argmax(local_obb.extent)
            # local_obb.R 的列是OBB的局部轴在点云的（当前）局部坐标系中的表示
            local_longest_axis_vec = local_obb.R[:, longest_extent_idx]
            
            target_local_x_axis = np.array([1.0, 0.0, 0.0])

            print(f"  局部OBB最长轴 (在点云当前局部坐标系中): {local_longest_axis_vec.round(3)}")
            print(f"  目标对齐轴 (点云局部X轴): {target_local_x_axis}")

            R_local_align = get_rotation_matrix_between_vectors(local_longest_axis_vec, target_local_x_axis)
            
            if R_local_align is not None:
                print(f"  计算得到的局部对齐旋转矩阵:\n{R_local_align.round(3)}")
                T_local_align_homogeneous = create_transformation_matrix_opt8(R_local_align, None)
                points_high_res_centered_aligned_mm = transform_points_opt8(points_high_res_centered_mm, T_local_align_homogeneous)
                print("  已将点云的局部姿态调整为使其OBB最长边对齐局部X轴。")
            else:
                print("  局部对齐旋转计算失败或无需旋转，使用原始居中点云。")
        except Exception as e_obb_align:
            print(f"  局部OBB对齐过程中出错: {e_obb_align}。将使用原始居中点云。")
            points_high_res_centered_aligned_mm = points_high_res_centered_mm # Fallback
    else:
        print("  点云为空，跳过局部对齐步骤。")
    
    # 将 *已在局部对齐过的* 物体点云变换到其新的、对调后的世界位置 (由 T_object_coord_system_in_world 定义)
    object_points_transformed_full_mm = transform_points_opt8(points_high_res_centered_aligned_mm, T_object_coord_system_in_world)


    # --- G. 点云抽稀 ---
    print(f"\n对定位后的 '{HIGH_RES_OBJECT_FILENAME}' 进行抽稀至约 {TARGET_POINT_COUNT_FOR_SIM} 点...")
    final_sampled_points_mm = None
    sampled_colors_uint8_pv = None
    sampled_colors_float_o3d = None
    original_hr_colors_float_o3d = np.asarray(high_res_object_o3d.colors) if high_res_object_o3d.has_colors() else None

    if len(object_points_transformed_full_mm) > TARGET_POINT_COUNT_FOR_SIM:
        indices = np.random.choice(len(object_points_transformed_full_mm), TARGET_POINT_COUNT_FOR_SIM, replace=False)
        final_sampled_points_mm = object_points_transformed_full_mm[indices]
        if original_hr_colors_float_o3d is not None and indices.size > 0:
            sampled_colors_float_o3d = original_hr_colors_float_o3d[indices]
            if sampled_colors_float_o3d.ndim == 2 and sampled_colors_float_o3d.shape[1] == 3: # Ensure it's RGB
                 sampled_colors_uint8_pv = (sampled_colors_float_o3d * 255).astype(np.uint8)
            else:
                 print(f"警告: 采样后的点云颜色维度不是3 (RGB)，PyVista颜色可能不正确。Shape: {sampled_colors_float_o3d.shape}")
                 sampled_colors_uint8_pv = None
                 sampled_colors_float_o3d = None
        print(f"  随机采样后点云数量: {len(final_sampled_points_mm)}")
    elif len(object_points_transformed_full_mm) > 0 :
        final_sampled_points_mm = object_points_transformed_full_mm
        if original_hr_colors_float_o3d is not None:
            if original_hr_colors_float_o3d.ndim == 2 and original_hr_colors_float_o3d.shape[1] == 3:
                sampled_colors_float_o3d = original_hr_colors_float_o3d
                sampled_colors_uint8_pv = (original_hr_colors_float_o3d * 255).astype(np.uint8)
            else:
                print(f"警告: 原始点云颜色维度不是3 (RGB)，PyVista/Open3D颜色可能不正确。Shape: {original_hr_colors_float_o3d.shape}")
                sampled_colors_uint8_pv = None
                sampled_colors_float_o3d = None
        print(f"  点云点数: {len(final_sampled_points_mm)} (无需抽稀).")
    else: sys.exit(f"错误: 变换后的高精度点云不含点.")

    # --- H. 更新全局变量 ---
    object_points_global_static = final_sampled_points_mm
    object_mesh_global_static = pv.PolyData(object_points_global_static)
    if sampled_colors_uint8_pv is not None:
        object_mesh_global_static.point_data['colors'] = sampled_colors_uint8_pv
        print("高精度物体颜色已处理并赋给PyVista对象。")
    else:
        print(f"高精度点云无颜色信息或颜色格式不兼容，PyVista对象将使用默认颜色。")
    object_centroid_global_static = np.mean(object_points_global_static, axis=0)
    num_object_points_global_static = object_points_global_static.shape[0]
    print(f"\n已加载并处理 '{HIGH_RES_OBJECT_FILENAME}'。用于仿真点数: {num_object_points_global_static}，质心: {object_centroid_global_static.round(3)}")

    # 新增: 计算并存储物体的OBB (基于最终的世界坐标点云)
    if num_object_points_global_static > 0:
        o3d_temp_pcd_for_obb = o3d.geometry.PointCloud()
        o3d_temp_pcd_for_obb.points = o3d.utility.Vector3dVector(object_points_global_static)
        world_obb_object_global = o3d_temp_pcd_for_obb.get_oriented_bounding_box()
        world_obb_object_global.color = object_obb_color_o3d # 设置Open3D OBB颜色
        print(f"  计算得到的物体OBB (世界坐标系): 中心={world_obb_object_global.center.round(3)}, 范围={world_obb_object_global.extent.round(3)}")


        print(f"  物体OBB的三维尺寸 (沿OBB局部轴的长度): X={world_obb_object_global.extent[0]:.3f} mm, Y={world_obb_object_global.extent[1]:.3f} mm, Z={world_obb_object_global.extent[2]:.3f} mm")

        pv_obb_object_mesh_global = pv.Cube(center=(0,0,0), # 创建在原点
                                     x_length=world_obb_object_global.extent[0],
                                     y_length=world_obb_object_global.extent[1],
                                     z_length=world_obb_object_global.extent[2])
        # 构建OBB的变换矩阵
        T_pv_obb_transform = np.eye(4)
        T_pv_obb_transform[:3,:3] = world_obb_object_global.R
        T_pv_obb_transform[:3,3] = world_obb_object_global.center
        pv_obb_object_mesh_global.transform(T_pv_obb_transform, inplace=True)
    else:
        world_obb_object_global = None
        pv_obb_object_mesh_global = None
        print("  物体点云为空，无法计算OBB。")


    # --- I. 初始化手指模型 ---
    initial_coords_ref_global = load_initial_coordinates(INITIAL_COORDS_PATH, NODE_COUNT)
    model_global,scaler_X_global,scaler_y_global,device_global = load_prediction_components(MODEL_PATH,X_SCALER_PATH,Y_SCALER_PATH,INPUT_DIM,OUTPUT_DIM,HIDDEN_LAYER_1,HIDDEN_LAYER_2,HIDDEN_LAYER_3)
    if initial_coords_ref_global is None or model_global is None: sys.exit("错误：未能初始化手指模型或坐标。")
    faces_np_global = create_faces_array(NODE_COUNT)
    width_translation_vector_global = np.array([0, finger_width, 0])
    bottom_node_idx = np.argmin(initial_coords_ref_global[:,0])
    ref_mid_pt = initial_coords_ref_global[bottom_node_idx] + width_translation_vector_global/2.
    T1_translate_global = create_transformation_matrix_opt8(None,-ref_mid_pt)
    rot_ref_to_local = np.array([[0,1,0],[0,0,1],[1,0,0]])
    T2_rotate_global = create_transformation_matrix_opt8(rot_ref_to_local,None)

    # --- J. 优化流程 ---
    print("\n--- 检查所有必要的全局变量是否已准备就绪 (继续Opt_9流程) ---")
    if object_points_global_static is None: sys.exit("错误：object_points_global_static 未初始化。")

    # PyVista托盘 (圆柱体) 使用其最终的几何世界位姿进行变换
    tray_pv_canonical = pv.Cylinder(center=(0,0,0), radius=tray_radius, height=tray_height, direction=(0,0,1), resolution=100)
    tray_pv = tray_pv_canonical.transform(T_actual_tray_geometry_world, inplace=False)
    print(f"PyVista tray_pv (圆柱体) 已创建并变换到其新的、修正了朝向的几何位姿。")


    print(f"\n--- 开始在 {NUM_INITIAL_SEARCHES} 个随机 (r, angle_offset) 组合上寻找初始接触点 ---")
    collected_initial_points = []
    last_search_attempt_result = None

    x_coords_rel_new_obj_centroid = object_points_global_static[:, 0] - object_centroid_global_static[0]
    y_coords_rel_new_obj_centroid = object_points_global_static[:, 1] - object_centroid_global_static[1]
    distances_from_new_obj_centroid_xy = np.sqrt(x_coords_rel_new_obj_centroid**2 + y_coords_rel_new_obj_centroid**2)

    if distances_from_new_obj_centroid_xy.size > 0:
        effective_radius_of_object_at_old_tray_pos = np.max(distances_from_new_obj_centroid_xy)
        r_search_min_heuristic = effective_radius_of_object_at_old_tray_pos * 0.8
        r_search_max_heuristic = tray_radius * 1.2
    else:
        print("警告: 无法从点云计算有效半径，使用基于托盘半径的默认值。")
        r_search_min_heuristic = R_BOUNDS[0]
        r_search_max_heuristic = R_BOUNDS[1]

    r_search_min = np.clip(r_search_min_heuristic, R_BOUNDS[0], R_BOUNDS[1] * 0.95)
    r_search_max = np.clip(r_search_max_heuristic, r_search_min + (R_BOUNDS[1] - R_BOUNDS[0]) * 0.05, R_BOUNDS[1])

    if r_search_min >= r_search_max:
        print(f"警告: 启发式半径搜索范围无效 [{r_search_min_heuristic:.2f}, {r_search_max_heuristic:.2f}] -> [{r_search_min:.2f}, {r_search_max:.2f}]. 将使用 R_BOUNDS.")
        r_search_min=R_BOUNDS[0]
        r_search_max=R_BOUNDS[1]

    angle_offset_search_min = ANGLE_OFFSET_BOUNDS[0]
    angle_offset_search_max = ANGLE_OFFSET_BOUNDS[1]
    print(f"更新后的初始搜索范围: r in [{r_search_min:.2f}, {r_search_max:.2f}], angle_offset in [{angle_offset_search_min:.1f}, {angle_offset_search_max:.1f}] (r 是相对于新托盘中心的放置半径)")


    for k_init in range(NUM_INITIAL_SEARCHES):
        print(f"\n--- 初始搜索尝试 {k_init+1}/{NUM_INITIAL_SEARCHES} ---")
        r_val = np.random.uniform(r_search_min,r_search_max)
        angle_val = np.random.uniform(angle_offset_search_min,angle_offset_search_max)
        search_res = find_initial_grasp(initial_r=r_val,initial_angle_offset=angle_val,pressure_step=PRESSURE_STEP_INIT_SEARCH,max_pressure_init=max_pressure)
        last_search_attempt_result = search_res
        if search_res and search_res['status']=='FoundInitialPoint':
            params_entry=[search_res['r']]+search_res['pressures']+[search_res['angle_offset']]
            collected_initial_points.append(params_entry)
            print(f"*** 找到有效初始点: r={params_entry[0]:.2f},P={np.round(params_entry[1:4],0)},offset={params_entry[4]:.1f}° ***")
        else:
            print(f"--- 未找到有效初始接触点 (状态: {search_res.get('status', 'Unknown')}) ---")
    print(f"\n--- 初始接触点搜索完成，共找到 {len(collected_initial_points)} 个有效初始点 ---")

    if last_search_attempt_result and 'finger_meshes' in last_search_attempt_result:
        print("\n(PyVista) 可视化最后一次初始搜索尝试的状态...")
        title_init_pv_viz = f"PyVista - Initial Search SWAPPED & ORIENTED (Status: {last_search_attempt_result.get('status','N/A')})"
        plotter_init_pv = pv.Plotter(window_size=[900,700],title=title_init_pv_viz)
        plotter_init_pv.add_mesh(tray_pv,color=tray_color,opacity=0.5,name='tray_init_pv_oriented')
        if object_mesh_global_static:
            if 'colors' in object_mesh_global_static.point_data and object_mesh_global_static.point_data['colors'] is not None:
                plotter_init_pv.add_mesh(object_mesh_global_static,scalars='colors',rgba=True,style='points',point_size=5,name='obj_init_pv_color_swapped')
            else:
                plotter_init_pv.add_mesh(object_mesh_global_static,color=object_point_color,style='points',point_size=5,name='obj_init_pv_def_swapped')
        
        if pv_obb_object_mesh_global is not None: # 新增: 添加OBB到PyVista
            plotter_init_pv.add_mesh(pv_obb_object_mesh_global, style='wireframe', color=object_obb_color_pv, line_width=2, name='object_obb_init_pv')


        if show_axes:
            plotter_init_pv.add_axes_at_origin(labels_off=False,line_width=3)

        meshes_viz_i_pv = last_search_attempt_result.get('finger_meshes',[])
        for i,m_viz_i_pv in enumerate(meshes_viz_i_pv):
            if m_viz_i_pv is not None: plotter_init_pv.add_mesh(m_viz_i_pv,color=finger_color,style='surface',opacity=0.85,name=f'f_init_pv_{i}')
        status_txt_i_pv = f"LastSearch: r={last_search_attempt_result.get('r','?'):.2f},off={last_search_attempt_result.get('angle_offset','?'):.1f},P={np.array(last_search_attempt_result.get('pressures',['?']*3)).round(0)}\nStatus: {last_search_attempt_result.get('status','N/A')}"
        plotter_init_pv.add_text(status_txt_i_pv,position="upper_left",font_size=10,color='white')
        plotter_init_pv.camera_position='xy'
        print("(PyVista) 显示最后初始搜索结果。按Q关闭此窗口。")
        plotter_init_pv.show(cpos='xy', auto_close=False)

        visualize_poses_with_open3d(
            tray_geometry_transform_mm=T_actual_tray_geometry_world,
            tray_axes_transform_mm=T_tray_coord_system_in_world,
            bottle_axes_transform_mm=T_object_coord_system_in_world,
            bottle_points_world_mm=object_points_global_static,
            bottle_colors_rgb_float=sampled_colors_float_o3d,
            bottle_obb_world=world_obb_object_global, # 传递OBB
            tray_radius_mm=tray_radius,
            tray_height_mm=tray_height,
            window_title="Open3D - Initial SWAPPED & ORIENTED Poses & Axes"
        )
    else:
        print("警告：初始搜索结果不完整或物体未加载，无法进行PyVista和Open3D可视化。")


    proceed_bo = bool(collected_initial_points) or N_INITIAL_POINTS_BO > 0
    if not collected_initial_points and N_INITIAL_POINTS_BO > 0:
        print("未通过初始搜索找到有效三点接触，但BO将尝试自行生成初始点。")
    elif not proceed_bo:
        print("没有初始点（通过搜索或BO设定），跳过优化。")

    res_bo = None
    if proceed_bo:
        n_rand_bo_call = N_INITIAL_POINTS_BO
        x0_for_bo = None
        if collected_initial_points:
            x0_for_bo = collected_initial_points
            n_rand_bo_call = max(0, N_INITIAL_POINTS_BO - len(x0_for_bo))
        print(f"\n开始贝叶斯优化 (提供{len(x0_for_bo) if x0_for_bo else 0}点,BO随机探索{n_rand_bo_call}点,总调用目标{N_CALLS_BO}次)")
        try:
            res_bo = gp_minimize(func=evaluate_grasp,dimensions=dimensions,acq_func="EI",n_calls=N_CALLS_BO,n_initial_points=n_rand_bo_call,x0=x0_for_bo,random_state=123,noise=0.01)
        except Exception as e:
            print(f"\nBO过程错: {e}")
            traceback.print_exc()
        print("\n贝叶斯优化结束。")

    if res_bo:
        best_p_list_bo = res_bo.x
        best_c_bo = res_bo.fun
        best_p_dict_bo = dict(zip(param_names,best_p_list_bo))
        print("\n使用最优参数重新评估最终状态...")
        final_c_check_bo = evaluate_grasp(best_p_list_bo) # 使用列表传递参数
        best_gii_eval_bo = -final_c_check_bo if final_c_check_bo < -1e-9 else 0.
        print("\n找到的最优参数:")
        [print(f"  {n:<12} = {v:.4f}") for n,v in best_p_dict_bo.items()]
        print(f"优化找到成本: {best_c_bo:.4f}")
        print(f"重评成本: {final_c_check_bo:.4f}")
        if best_gii_eval_bo>0: print(f"重评GII: {best_gii_eval_bo:.4f}")
        else: print("重评状态无有效GII。")

        print("\n使用最优参数生成最终可视化...")
        final_pressures_bo_viz = [best_p_dict_bo['p1'],best_p_dict_bo['p2'],best_p_dict_bo['p3']]
        final_meshes_bo_viz = []
        all_preds_ok_bo_viz = True
        for i_f_bo_viz in range(3):
            disps_f_bo = predict_displacements_for_pressure(model_global,scaler_X_global,scaler_y_global,device_global,final_pressures_bo_viz[i_f_bo_viz])
            if disps_f_bo is None:
                all_preds_ok_bo_viz=False
                break
            c1u_f_bo=initial_coords_ref_global+disps_f_bo
            c2r_f_bo=initial_coords_ref_global+width_translation_vector_global
            c2u_f_bo=c2r_f_bo+disps_f_bo
            sc1_f_bo=sort_points_spatially(c1u_f_bo)
            sc2_f_bo=sort_points_spatially(c2u_f_bo)
            if sc1_f_bo is None or sc2_f_bo is None:
                all_preds_ok_bo_viz=False
                break
            verts_f_bo=np.vstack((sc1_f_bo,sc2_f_bo))
            if faces_np_global is None or faces_np_global.size==0:
                all_preds_ok_bo_viz=False
                break
            try:
                mesh_r_f_bo=pv.PolyData(verts_f_bo,faces=faces_np_global)
            except Exception:
                all_preds_ok_bo_viz=False
                break
            ang_d_f_bo=([0,120,240][i_f_bo_viz])+best_p_dict_bo['angle_offset']
            ang_r_f_bo=np.radians(ang_d_f_bo)
            rot_z_f_bo=create_rotation_matrix_z(ang_r_f_bo+np.pi/2.)
            pos_c_f_bo=np.array([best_p_dict_bo['r']*np.cos(ang_r_f_bo),best_p_dict_bo['r']*np.sin(ang_r_f_bo),0.])
            T3_f_bo=create_transformation_matrix_opt8(rot_z_f_bo,pos_c_f_bo)

            T_transform_finger_relative_to_tray_origin_bo = T3_f_bo @ T2_rotate_global @ T1_translate_global
            if T_pose_for_tray_display_and_finger_placement_global is None:
                all_preds_ok_bo_viz=False; break
            T_f_finger_bo_world = T_pose_for_tray_display_and_finger_placement_global @ T_transform_finger_relative_to_tray_origin_bo

            try:
                mesh_tf_f_bo=mesh_r_f_bo.transform(T_f_finger_bo_world,inplace=False)
                if mesh_tf_f_bo is None or mesh_tf_f_bo.n_points==0:
                    all_preds_ok_bo_viz=False
                    break
                mesh_tf_f_bo.clean(inplace=True)
                if mesh_tf_f_bo.n_cells>0:
                    mesh_tf_f_bo.compute_normals(cell_normals=True,point_normals=False,inplace=True,auto_orient_normals=True,non_manifold_traversal=False)
                final_meshes_bo_viz.append(mesh_tf_f_bo)
            except Exception:
                all_preds_ok_bo_viz=False
                break

        if all_preds_ok_bo_viz and len(final_meshes_bo_viz)==3:
            state_disp_bo=f"Cost={final_c_check_bo:.3f}"
            gii_disp_bo=f"GII={best_gii_eval_bo:.3f}" if best_gii_eval_bo>0 else "GII:N/A"
            plotter_final_pv = pv.Plotter(window_size=[1000,800],title="PyVista - Optimal Grasp SWAPPED & ORIENTED Configuration")
            plotter_final_pv.add_mesh(tray_pv,color=tray_color,opacity=0.5,name='tray_final_pv_oriented')
            if object_mesh_global_static:
                if 'colors' in object_mesh_global_static.point_data and object_mesh_global_static.point_data['colors'] is not None:
                    plotter_final_pv.add_mesh(object_mesh_global_static,scalars='colors',rgba=True,style='points',point_size=5,name='obj_final_pv_c_swapped')
                else:
                    plotter_final_pv.add_mesh(object_mesh_global_static,color=object_point_color,style='points',point_size=5,name='obj_final_pv_d_swapped')

            if pv_obb_object_mesh_global is not None: # 新增: 添加OBB到PyVista
                plotter_final_pv.add_mesh(pv_obb_object_mesh_global, style='wireframe', color=object_obb_color_pv, line_width=2, name='object_obb_final_pv')

            if show_axes:
                plotter_final_pv.add_axes_at_origin(labels_off=False,line_width=3)

            for i_fv_bo,m_fv_bo in enumerate(final_meshes_bo_viz):
                if m_fv_bo: plotter_final_pv.add_mesh(m_fv_bo,color=finger_color,style='surface',opacity=0.85,name=f'f_final_pv_{i_fv_bo}')
            params_txt_bo=f"r={best_p_dict_bo['r']:.2f},P1={best_p_dict_bo['p1']:.0f},P2={best_p_dict_bo['p2']:.0f},P3={best_p_dict_bo['p3']:.0f},off={best_p_dict_bo['angle_offset']:.1f}°"
            plotter_final_pv.add_text(f"Optimal:{params_txt_bo}\n{state_disp_bo}\n{gii_disp_bo}",position="upper_left",font_size=10,color='white')
            plotter_final_pv.camera_position='xy'
            print("\n(PyVista) 显示最优抓取配置。按Q退出此窗口.")
            plotter_final_pv.show(cpos='xy', auto_close=False)

            visualize_poses_with_open3d(
                tray_geometry_transform_mm=T_actual_tray_geometry_world,
                tray_axes_transform_mm=T_tray_coord_system_in_world,
                bottle_axes_transform_mm=T_object_coord_system_in_world,
                bottle_points_world_mm=object_points_global_static,
                bottle_colors_rgb_float=sampled_colors_float_o3d,
                bottle_obb_world=world_obb_object_global, # 传递OBB
                tray_radius_mm=tray_radius,
                tray_height_mm=tray_height,
                window_title="Open3D - Optimal SWAPPED & ORIENTED Poses & Axes"
            )
        else:
            print("未能为最优参数生成手指网格，无法进行PyVista最终可视化。")
    else:
        print("\nBO未成功或无结果，无法显示最优。")
    print("\n程序结束。")

