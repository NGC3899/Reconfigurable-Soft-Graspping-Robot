# -*- coding: utf-8 -*-
# 虚拟实验2：利用GraspNet预测的末端位姿与YCB数据集中的物体点云进行构型贝叶斯优化并可视化
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
import vtk # For VTK font constants

# --- 打印版本信息 ---
try:
    print(f"Opt_10_BO_experiment.txt (可视化美化版 v1.1, 已添加手动平移与旋转接口)")
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
    from skopt.space import Real, Integer
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
INPUT_DIM = 1
NODE_COUNT = 63
OUTPUT_DIM = NODE_COUNT * 3
HIDDEN_LAYER_1 = 128
HIDDEN_LAYER_2 = 256
HIDDEN_LAYER_3 = 128

# --- 3. 文件路径定义 ---
MODEL_PATH = 'best_mlp_model.pth'
X_SCALER_PATH = 'x_scaler.joblib'
Y_SCALER_PATH = 'y_scaler.joblib'
INITIAL_COORDS_PATH = 'initial_coordinates.txt'

GRASP_OUTPUTS_BASE_PATH = r"C:\Users\admin\Desktop\Figure\grasp_experiments\sugar_box" # <--- 修改为您的路径
RELATIVE_POSE_FILENAME = "relative_gripper_to_object_pose.txt"
HIGH_RES_OBJECT_DESKTOP_PATH = r"C:\Users\admin\Desktop\Figure\grasp_experiments\sugar_box" # <--- 修改为您的路径
HIGH_RES_OBJECT_FILENAME = "sugar_box.ply" # 示例，您会从外部读取

RELATIVE_POSE_FILE_PATH = os.path.join(GRASP_OUTPUTS_BASE_PATH, RELATIVE_POSE_FILENAME)
HIGH_RES_OBJECT_PLY_PATH = os.path.join(HIGH_RES_OBJECT_DESKTOP_PATH, HIGH_RES_OBJECT_FILENAME)

# --- 4. 配置参数 ---
tray_radius = 60.0
tray_height = 1.0
tray_center = np.array([0.0, 0.0, -tray_height / 2.0]) # 托盘几何中心在世界坐标系中的位置
finger_width = 10.0
TARGET_POINT_COUNT_FOR_SIM = 2500
show_axes = True
collision_threshold = 1.0
overlap_threshold = 1e-4
friction_coefficient = 0.5
eigenvalue_threshold = 1e-6
max_pressure = 40000.0
PRESSURE_STEP_EVAL_GRASP = 500.0
INITIAL_PRESSURE_EVAL_GRASP = 100.0
N_CALLS_BO = 20
N_INITIAL_POINTS_BO = 5
R_BOUNDS = (30, tray_radius * 0.95)
N_FINGER_SLOTS = 9
OBJECT_SCALE_FACTOR = 820
CHARACTERISTIC_LENGTH_FOR_GII = 30
DOT_PROD_TOLERANCE_LOCAL = 1e-6

# ======================= 新增：手动平移接口 =======================
# 在这里定义您希望对物体点云施加的额外平移量 (单位: 毫米)
# 这个平移是在从文件加载并应用初始变换矩阵之后应用的。
# 格式: [X, Y, Z]
# manual_object_translation_xyz = [18.0, -15.0, -3.0] # bowl
# manual_object_translation_xyz = [-20.0, -18.0, 18.0] # large_marker
# manual_object_translation_xyz = [-30.0, 0.0, -20.0] # tomato_soup_can
manual_object_translation_xyz = [10.0, 0.0, 0.0] # sugar_box
# manual_object_translation_xyz = [52.0, -25.0, 0.0] # power_drill
# manual_object_translation_xyz = [5.0, 0.0, 0.0] # mug
# =================================================================

# ======================= 新增：手动旋转接口 =======================
# 在这里定义您希望对物体点云施加的额外旋转量 (单位: 度)
# 这个旋转是在所有初始平移和变换之后，但在最终用于仿真之前，
# 围绕物体自身的质心进行的。
# 格式: [绕X轴旋转度数, 绕Y轴旋转度数, 绕Z轴旋转度数]
# manual_object_rotation_xyz_deg = [-90.0, 0.0, 0.0] # 示例：绕Z轴旋转0度
manual_object_rotation_xyz_deg = [0.0, 0.0, 0.0]
# =================================================================


# --- 新增：可视化美化参数 (来自 Opt_video.txt) ---
finger_color_viz = '#ff7f0e' # Matplotlib Orange
tray_color_viz = '#BDB7A4'   # Desaturated Tan/Khaki
object_point_color_viz = '#1f77b4' # Matplotlib Blue
object_obb_color_viz = '#2ca02c'
background_color_viz = '#EAEAEA' # Light Gray
text_color_viz = 'black'
font_family_viz = 'times'
# --- end 美化参数 ---

COST_MESH_FAILURE = 9.0
COST_OVERLAP = 5.0
COST_INTERSECTION = 4.0
COST_MAX_PRESSURE_NO_CONTACT = 6.0
COST_NO_CONTACT_OR_ITER_LIMIT = 3.0
COST_LOW_GII_OR_FEW_CONTACTS = 2.0

all_valid_finger_combinations_global = list(itertools.permutations(range(N_FINGER_SLOTS), 3))
num_valid_combinations = len(all_valid_finger_combinations_global)
print(f"总共有 {N_FINGER_SLOTS} 个手指槽位，生成了 {num_valid_combinations} 种有效的三手指位置组合。")

param_names_bo = ['r', 'combo_idx']
search_space_bo = [
    Real(name='r', low=R_BOUNDS[0], high=R_BOUNDS[1]),
    Integer(name='combo_idx', low=0, high=num_valid_combinations - 1)
]

# --- 5. 辅助函数 ---
def o3d_create_transformation_matrix(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t.flatten(); return T
def create_rotation_matrix(axis, angle_rad):
    axis = np.asarray(axis).astype(float); axis_norm = LA.norm(axis)
    if axis_norm < 1e-9 : return np.identity(3)
    axis /= axis_norm; a = np.cos(angle_rad / 2.0); b, c, d = -axis * np.sin(angle_rad / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d; bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)], [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)], [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

# --- 新增：为XYZ轴创建旋转矩阵的辅助函数 ---
def create_rotation_matrix_x(angle_rad): return create_rotation_matrix([1,0,0], angle_rad)
def create_rotation_matrix_y(angle_rad): return create_rotation_matrix([0,1,0], angle_rad)
def create_rotation_matrix_z(angle_rad): return create_rotation_matrix([0,0,1], angle_rad)
# --- 结束新增辅助函数 ---

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

# --- 新增/修改：可视化函数 ---

def setup_publication_plotter(title, window_size=[1000, 800], off_screen_default=False):
    """
    创建一个具有统一出版级风格的 PyVista 绘图器。
    """
    plotter_theme = pv.themes.DocumentTheme()
    plotter_theme.font.family = font_family_viz
    plotter_theme.font.color = pv.Color(text_color_viz)
    plotter_theme.font.size = 12
    plotter_theme.font.label_size = 10
    plotter_theme.background = pv.Color(background_color_viz)

    plotter = pv.Plotter(window_size=window_size, theme=plotter_theme, title=title, off_screen=off_screen_default)
    plotter.enable_anti_aliasing('msaa', multi_samples=8)
    plotter.enable_parallel_projection()

    # 设置光照
    plotter.remove_all_lights()
    plotter.enable_lightkit()

    return plotter

def visualize_poses_with_open3d(tray_geometry_transform_mm, tray_axes_transform_mm, bottle_axes_transform_mm, bottle_points_world_mm, bottle_colors_rgb_float, bottle_obb_world, tray_radius_mm, tray_height_mm, window_title="Open3D Relative Pose Visualization"):
    """
    使用 Open3D 进行位姿和坐标轴的可视化 (已美化)。
    """
    print(f"\n--- 打开 Open3D 窗口: {window_title} ---")
    geometries = []

    # 托盘
    o3d_tray_canonical_centered = o3d.geometry.TriangleMesh.create_cylinder(radius=tray_radius_mm, height=tray_height_mm, resolution=20, split=4)
    o3d_tray_canonical_centered.translate(np.array([0,0,-tray_height_mm/2.0]))
    o3d_tray_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_tray_canonical_centered)
    o3d_tray_wireframe.transform(tray_geometry_transform_mm)
    o3d_tray_wireframe.paint_uniform_color(pv.Color(tray_color_viz).float_rgb)
    geometries.append(o3d_tray_wireframe)

    # 物体点云
    o3d_bottle_pcd = o3d.geometry.PointCloud()
    o3d_bottle_pcd.points = o3d.utility.Vector3dVector(bottle_points_world_mm)
    if bottle_colors_rgb_float is not None and len(bottle_colors_rgb_float) == len(bottle_points_world_mm):
        o3d_bottle_pcd.colors = o3d.utility.Vector3dVector(bottle_colors_rgb_float)
    else:
        o3d_bottle_pcd.paint_uniform_color(pv.Color(object_point_color_viz).float_rgb)
    geometries.append(o3d_bottle_pcd)

    # 物体包围盒
    # if bottle_obb_world is not None:
        # bottle_obb_world.color = pv.Color(object_obb_color_viz).float_rgb # 使用美化颜色
        # geometries.append(bottle_obb_world)

    # 托盘坐标轴
    tray_axes_size = tray_radius_mm * 0.7
    o3d_tray_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=tray_axes_size, origin=[0,0,0])
    o3d_tray_axes.transform(tray_axes_transform_mm)
    geometries.append(o3d_tray_axes)

    # 物体坐标轴
    if bottle_points_world_mm.shape[0] > 0:
        temp_bottle_pcd_for_bbox = o3d.geometry.PointCloud(); temp_bottle_pcd_for_bbox.points = o3d.utility.Vector3dVector(bottle_points_world_mm)
        bbox_bottle = temp_bottle_pcd_for_bbox.get_axis_aligned_bounding_box()
        diag_len = LA.norm(bbox_bottle.get_max_bound() - bbox_bottle.get_min_bound())
        bottle_axes_size = diag_len * 0.25 if diag_len > 1e-2 else tray_axes_size * 0.8
    else:
        bottle_axes_size = tray_axes_size * 0.8
    bottle_axes_size = max(bottle_axes_size, 5.0)
    o3d_bottle_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=bottle_axes_size, origin=[0,0,0])
    o3d_bottle_axes.transform(bottle_axes_transform_mm)
    geometries.append(o3d_bottle_axes)

    # 世界坐标轴
    world_axes_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=tray_radius_mm * 1.2, origin=[0,0,0])
    geometries.append(world_axes_o3d)

    # 设置渲染选项
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=1000, height=800)
    opt = vis.get_render_option()
    opt.background_color = np.asarray(pv.Color(background_color_viz).float_rgb)
    opt.point_size = 3.5
    opt.light_on = True

    for geom in geometries:
        vis.add_geometry(geom)

    vis.run()
    vis.destroy_window()
    print(f"--- 关闭 Open3D 窗口: {window_title} ---")

# --- 全局变量定义 ---
initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global = [None]*5
object_points_global_static, object_centroid_global_static, object_mesh_global_static = [None]*3
world_obb_object_global = None; pv_obb_object_mesh_global = None
num_object_points_global_static, faces_np_global, width_translation_vector_global = 0, None, None
T1_translate_global, T2_rotate_global = None, None
T_pose_for_tray_display_and_finger_placement_global = None
tray_pv_global = None

# --- 调试可视化开关 ---
DEBUG_VISUALIZE_FAILED_GRASPS = False # 设置为 True 以在 BO 失败步骤时进行可视化
FAILED_GRASP_VIS_COUNT = 0 # 用于给失败可视化窗口编号
MAX_FAILED_GRASP_VIS = 10 # 最多显示多少个失败的可视化，防止过多窗口

def visualize_failed_bo_step(r_param, chosen_indices_param, pressures_param, finger_meshes_param, failure_reason_str):
    """
    使用美化参数来可视化BO失败的步骤。
    """
    global DEBUG_VISUALIZE_FAILED_GRASPS, FAILED_GRASP_VIS_COUNT, MAX_FAILED_GRASP_VIS
    global object_mesh_global_static, tray_pv_global, pv_obb_object_mesh_global, show_axes

    if not DEBUG_VISUALIZE_FAILED_GRASPS or FAILED_GRASP_VIS_COUNT >= MAX_FAILED_GRASP_VIS:
        if FAILED_GRASP_VIS_COUNT == MAX_FAILED_GRASP_VIS and DEBUG_VISUALIZE_FAILED_GRASPS:
            print("已达到最大失败可视化次数，后续失败将不再显示。")
            FAILED_GRASP_VIS_COUNT +=1
        return

    FAILED_GRASP_VIS_COUNT += 1

    plotter_title = f"BO失败调试 #{FAILED_GRASP_VIS_COUNT}: {failure_reason_str}"
    plotter_debug = setup_publication_plotter(plotter_title, window_size=[900, 700])

    if tray_pv_global:
        plotter_debug.add_mesh(tray_pv_global, color=tray_color_viz, opacity=0.7, name='tray_debug')
    if object_mesh_global_static:
        if 'colors' in object_mesh_global_static.point_data and object_mesh_global_static.point_data['colors'] is not None:
            plotter_debug.add_mesh(object_mesh_global_static, scalars='colors', rgb=True, style='points', point_size=3.5, render_points_as_spheres=True, name='obj_debug_color')
        else:
            plotter_debug.add_mesh(object_mesh_global_static, color=object_point_color_viz, style='points', point_size=3.5, render_points_as_spheres=True, name='obj_debug_default')
    # if pv_obb_object_mesh_global:
        # plotter_debug.add_mesh(pv_obb_object_mesh_global, style='wireframe', color=object_obb_color_viz, line_width=2, name='obj_obb_debug')
    if show_axes:
        plotter_debug.add_axes_at_origin(labels_off=True, line_width=2)

    if finger_meshes_param:
        for i, mesh in enumerate(finger_meshes_param):
            if mesh:
                plotter_debug.add_mesh(mesh, color=finger_color_viz, style='surface', opacity=0.9, smooth_shading=True, show_edges=True, edge_color='dimgray', name=f'finger_debug_{i}')
            else:
                print(f"  调试可视化警告: 手指 {i} 的网格为 None。")

    params_text = f"r={r_param:.3f}, 位置=({chosen_indices_param[0]},{chosen_indices_param[1]},{chosen_indices_param[2]})\n"
    pressures_text = f"压力=[{pressures_param[0]:.0f}, {pressures_param[1]:.0f}, {pressures_param[2]:.0f}]\n"
    reason_text = f"原因: {failure_reason_str}"
    plotter_debug.add_text(params_text + pressures_text + reason_text, position="upper_left", font=font_family_viz, font_size=10, color=text_color_viz)

    plotter_debug.camera_position = 'xy'
    plotter_debug.camera.zoom(1.2)
    print(f"\n(PyVista) 显示BO失败步骤 #{FAILED_GRASP_VIS_COUNT}。原因: {failure_reason_str}。请关闭窗口以继续...")
    plotter_debug.show(cpos='xy', auto_close=False)
    plotter_debug.close()

def evaluate_grasp(r, pos_idx1, pos_idx2, pos_idx3):
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
                print(f"  BO 评估: {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                if DEBUG_VISUALIZE_FAILED_GRASPS:
                    visualize_failed_bo_step(r, chosen_indices, current_pressures, deformed_finger_meshes_at_contact, failure_reason_for_vis)
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
                print(f"  BO 评估: {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                if DEBUG_VISUALIZE_FAILED_GRASPS:
                     visualize_failed_bo_step(r, chosen_indices, current_pressures, deformed_finger_meshes_at_contact, failure_reason_for_vis)
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
                    print(f"  BO 评估: {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                    if DEBUG_VISUALIZE_FAILED_GRASPS:
                        visualize_failed_bo_step(r, chosen_indices, current_pressures, current_step_finger_meshes, failure_reason_for_vis)
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
                print(f"  BO 评估: {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                if DEBUG_VISUALIZE_FAILED_GRASPS:
                    visualize_failed_bo_step(r, chosen_indices, current_pressures, current_step_finger_meshes, failure_reason_for_vis)
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
                        print(f"  BO 评估: {current_call_params_str} -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print:.4f}")
                        return (cost_to_return, return_pressures, return_meshes, return_contacts)
                    else:
                        cost_to_return = COST_LOW_GII_OR_FEW_CONTACTS
                        gii_value_to_print = gii if gii is not None else "计算失败"
                        failure_reason_for_vis = f"低GII ({gii_value_to_print if isinstance(gii_value_to_print, str) else gii_value_to_print:.4f})"
                        print(f"  BO 评估: {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print if isinstance(gii_value_to_print, str) else gii_value_to_print:.4f}")
                        if DEBUG_VISUALIZE_FAILED_GRASPS:
                             visualize_failed_bo_step(r, chosen_indices, return_pressures, return_meshes, failure_reason_for_vis)
                        return (cost_to_return, return_pressures, return_meshes, return_contacts)
                else:
                    cost_to_return = COST_LOW_GII_OR_FEW_CONTACTS
                    failure_reason_for_vis = "接触点不足 GII"
                    print(f"  BO 评估: {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                    if DEBUG_VISUALIZE_FAILED_GRASPS:
                        visualize_failed_bo_step(r, chosen_indices, return_pressures, return_meshes, failure_reason_for_vis)
                    return (cost_to_return, return_pressures, return_meshes, return_contacts)

            for i in range(3):
                if not finger_contact_established[i] and current_pressures[i] >= max_pressure:
                    cost_to_return = COST_MAX_PRESSURE_NO_CONTACT
                    failure_reason_for_vis = f"手指 {i} 最大压力无接触"
                    print(f"  BO 评估: {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                    if DEBUG_VISUALIZE_FAILED_GRASPS:
                        visualize_failed_bo_step(r, chosen_indices, current_pressures, deformed_finger_meshes_at_contact, failure_reason_for_vis)
                    return (cost_to_return, current_pressures, deformed_finger_meshes_at_contact, None)

            if not pressure_changed_this_step and not all(finger_contact_established):
                cost_to_return = COST_MAX_PRESSURE_NO_CONTACT
                failure_reason_for_vis = "停滞无接触"
                print(f"  BO 评估: {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
                if DEBUG_VISUALIZE_FAILED_GRASPS:
                    visualize_failed_bo_step(r, chosen_indices, current_pressures, deformed_finger_meshes_at_contact, failure_reason_for_vis)
                return (cost_to_return, current_pressures, deformed_finger_meshes_at_contact, None)

    except Exception as e_eval_grasp:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"DEBUG_BO: EXCEPTION in evaluate_grasp for {current_call_params_str}: {e_eval_grasp}")
        traceback.print_exc()
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        cost_to_return = 20.0
        failure_reason_for_vis = "evaluate_grasp内部异常"
        current_pressures_fallback = current_pressures if 'current_pressures' in locals() else np.full(3, INITIAL_PRESSURE_EVAL_GRASP, dtype=float)
        deformed_meshes_fallback = deformed_finger_meshes_at_contact

        print(f"  BO 评估: {current_call_params_str} ({failure_reason_for_vis}) -> 成本: {cost_to_return:.4f}, GII: {gii_value_to_print}")
        if DEBUG_VISUALIZE_FAILED_GRASPS:
             visualize_failed_bo_step(r, chosen_indices, current_pressures_fallback,
                                     deformed_meshes_fallback,
                                     failure_reason_for_vis)
        return (cost_to_return, None, None, None)


# --- 主脚本 ---
if __name__ == '__main__':
    # --- A. 加载GraspNet位姿 (物体相对于夹爪) ---
    T_gn_gripper_TO_gn_object_meters = load_transformation_matrix_from_txt(RELATIVE_POSE_FILE_PATH)
    if T_gn_gripper_TO_gn_object_meters is None: sys.exit("错误：未能加载物体相对于GraspNet夹爪的位姿。")

    # --- B. 定义托盘在世界坐标系中的初始 (或参考) 位姿 ---
    T_original_tray_pose_world = create_transformation_matrix_opt8(np.identity(3), tray_center)

    # --- C. 计算物体在托盘初始位姿下的“标准”世界位姿 ---
    T_gn_object_TO_gn_gripper_meters = np.linalg.inv(T_gn_gripper_TO_gn_object_meters)
    T_imported_obj_relative_to_tray_mm = copy.deepcopy(T_gn_object_TO_gn_gripper_meters)
    T_imported_obj_relative_to_tray_mm[:3, 3] *= 1000.0
    T_original_object_pose_world = T_original_tray_pose_world @ T_imported_obj_relative_to_tray_mm

    # --- D. 应用 Opt_10.txt (Opt_9_modified.txt) 的“位姿对调”逻辑 ---
    T_object_target_world_pose = T_original_tray_pose_world
    _T_tray_ref_before_fix = T_original_object_pose_world

    angle_rad_fix = np.pi / 2
    cos_fix = np.cos(angle_rad_fix); sin_fix = np.sin(angle_rad_fix)
    T_local_orientation_fix_for_tray_geometry = np.array([
        [cos_fix, 0, sin_fix, 0],
        [0,       1,       0, 0],
        [-sin_fix,0, cos_fix, 0],
        [0,       0,       0, 1]
    ])
    T_actual_tray_geometry_world = _T_tray_ref_before_fix @ T_local_orientation_fix_for_tray_geometry
    T_pose_for_tray_display_and_finger_placement_global = T_actual_tray_geometry_world

    T_tray_axes_vis_world = _T_tray_ref_before_fix
    T_object_axes_vis_world = T_object_target_world_pose

    print(f"\n--- 应用位姿对调逻辑后 ---")
    print(f"  物体点云的目标世界位姿 (T_object_target_world_pose):\n{T_object_target_world_pose}")
    print(f"  托盘几何体及手指放置的参考世界坐标系 (T_actual_tray_geometry_world):\n{T_actual_tray_geometry_world}")

    # --- E. 加载和处理高精度物体点云 ---
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

    # --- 应用手动定义的平移 ---
    manual_translation_vector = np.array(manual_object_translation_xyz)
    if manual_translation_vector.any(): # .any() 检查是否有非零元素
        print(f"\n--- 应用手动平移 ---")
        print(f"在初始位姿变换后，对物体点云应用额外平移: {manual_translation_vector} mm")
        object_points_transformed_full_mm += manual_translation_vector
        print(f"手动平移应用完毕。")
    # --- 手动平移应用结束 ---
    
    # ======================= 新增：应用手动旋转 =======================
    manual_rotation_deg = np.array(manual_object_rotation_xyz_deg)
    if manual_rotation_deg.any():
        print(f"\n--- 应用手动旋转 ---")
        print(f"将对物体点云应用额外旋转 (度): {manual_rotation_deg}")
        
        # 1. 计算当前点云的质心作为旋转中心
        if object_points_transformed_full_mm.shape[0] > 0:
            rotation_center = np.mean(object_points_transformed_full_mm, axis=0)
            print(f"旋转中心 (质心): {rotation_center.round(3)}")

            # 2. 创建旋转矩阵 (角度转弧度)
            angle_x_rad = np.radians(manual_rotation_deg[0])
            angle_y_rad = np.radians(manual_rotation_deg[1])
            angle_z_rad = np.radians(manual_rotation_deg[2])

            Rx = create_rotation_matrix_x(angle_x_rad)
            Ry = create_rotation_matrix_y(angle_y_rad)
            Rz = create_rotation_matrix_z(angle_z_rad)

            # 组合旋转矩阵 (应用顺序: Z -> Y -> X)
            R_combined = Rz @ Ry @ Rx
            
            # 3. 执行围绕质心的旋转
            # 3.1. 将点云平移,使质心位于原点
            points_translated_to_origin = object_points_transformed_full_mm - rotation_center
            # 3.2. 应用旋转
            points_rotated = points_translated_to_origin @ R_combined.T
            # 3.3. 将点云平移回原来的位置
            object_points_transformed_full_mm = points_rotated + rotation_center
            
            print("手动旋转应用完毕。")
        else:
            print("警告：点云为空，无法应用手动旋转。")
    # ======================= 手动旋转应用结束 =======================


    # --- F. 点云抽稀和颜色处理 ---
    final_sampled_points_mm = None; sampled_colors_uint8_pv = None; sampled_colors_float_o3d = None
    original_hr_colors_float_o3d = np.asarray(high_res_object_o3d.colors) if high_res_object_o3d.has_colors() else None
    if len(object_points_transformed_full_mm) > TARGET_POINT_COUNT_FOR_SIM:
        indices = np.random.choice(len(object_points_transformed_full_mm), TARGET_POINT_COUNT_FOR_SIM, replace=False)
        final_sampled_points_mm = object_points_transformed_full_mm[indices]
        if original_hr_colors_float_o3d is not None and indices.size > 0 and original_hr_colors_float_o3d.shape[0] == current_high_res_points_mm_orig_frame.shape[0]:
            try:
                sampled_colors_float_o3d = original_hr_colors_float_o3d[indices]
                if sampled_colors_float_o3d.ndim == 2 and sampled_colors_float_o3d.shape[1] == 3:
                    sampled_colors_uint8_pv = (sampled_colors_float_o3d * 255).astype(np.uint8)
                else: sampled_colors_uint8_pv = None; sampled_colors_float_o3d = None
            except IndexError:
                print("警告：颜色采样时索引不匹配（原始点云与采样索引），颜色可能不正确。")
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
    object_mesh_global_static = pv.PolyData(object_points_global_static)
    if sampled_colors_uint8_pv is not None: object_mesh_global_static.point_data['colors'] = sampled_colors_uint8_pv
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
        world_obb_object_global = o3d_temp_pcd_for_obb.get_oriented_bounding_box(); world_obb_object_global.color = pv.Color(object_obb_color_viz).float_rgb
        pv_obb_object_mesh_global = pv.Cube(center=(0,0,0), x_length=world_obb_object_global.extent[0], y_length=world_obb_object_global.extent[1], z_length=world_obb_object_global.extent[2])
        T_pv_obb_transform = np.eye(4); T_pv_obb_transform[:3,:3] = world_obb_object_global.R; T_pv_obb_transform[:3,3] = world_obb_object_global.center
        pv_obb_object_mesh_global.transform(T_pv_obb_transform, inplace=True)
    else: world_obb_object_global = None; pv_obb_object_mesh_global = None;

    # --- G. 初始化模型和手指参考 ---
    initial_coords_ref_global = load_initial_coordinates(INITIAL_COORDS_PATH, NODE_COUNT)
    model_global,scaler_X_global,scaler_y_global,device_global = load_prediction_components(MODEL_PATH,X_SCALER_PATH,Y_SCALER_PATH,INPUT_DIM,OUTPUT_DIM,HIDDEN_LAYER_1,HIDDEN_LAYER_2,HIDDEN_LAYER_3)
    if initial_coords_ref_global is None or model_global is None: sys.exit("错误：未能初始化手指模型或坐标。")
    faces_np_global = create_faces_array(NODE_COUNT)
    width_translation_vector_global = np.array([0, finger_width, 0])
    bottom_node_idx = np.argmin(initial_coords_ref_global[:,0])
    ref_mid_pt = initial_coords_ref_global[bottom_node_idx] + width_translation_vector_global/2.0
    T1_translate_global = create_transformation_matrix_opt8(None,-ref_mid_pt)
    rot_ref_to_local = np.array([[0,1,0],[0,0,1],[1,0,0]])
    T2_rotate_global = create_transformation_matrix_opt8(rot_ref_to_local,None)

    if object_points_global_static is None: sys.exit("错误：object_points_global_static 未初始化。")
    tray_pv_canonical = pv.Cylinder(center=(0,0,0), radius=tray_radius, height=tray_height, direction=(0,0,1), resolution=100)
    tray_pv_global = tray_pv_canonical.transform(T_actual_tray_geometry_world, inplace=False)

    # --- H. 贝叶斯优化 ---
    x0_for_bo = None
    n_rand_bo_call = N_INITIAL_POINTS_BO

    print(f"\n开始贝叶斯优化 (BO将随机探索{n_rand_bo_call}点, 总调用目标{N_CALLS_BO}次)")
    print(f"搜索空间维度: r, combo_idx (combo_idx 解码为三个不同的手指位置索引)")
    if DEBUG_VISUALIZE_FAILED_GRASPS:
        print(f"失败步骤可视化已启用。最多显示 {MAX_FAILED_GRASP_VIS} 次。")

    res_bo = None
    try:
        res_bo = gp_minimize(
            func=lambda p_bo: evaluate_grasp(
                r=p_bo[0],
                pos_idx1=all_valid_finger_combinations_global[int(p_bo[1])][0],
                pos_idx2=all_valid_finger_combinations_global[int(p_bo[1])][1],
                pos_idx3=all_valid_finger_combinations_global[int(p_bo[1])][2]
            )[0],
            dimensions=search_space_bo,
            acq_func="EI",
            n_calls=N_CALLS_BO,
            n_initial_points=n_rand_bo_call,
            x0=x0_for_bo,
            random_state=990,
            noise=0.01
        )
    except Exception as e:
        print(f"\nBO过程错: {e}")
        traceback.print_exc()
    print("\n贝叶斯优化结束。")

    # --- I. 处理和可视化优化结果 ---
    if res_bo:
        best_p_list_bo = res_bo.x
        best_c_bo = res_bo.fun

        best_r_bo = best_p_list_bo[0]
        best_combo_idx_bo = int(best_p_list_bo[1])
        best_pos_indices_bo = all_valid_finger_combinations_global[best_combo_idx_bo]

        print("\n使用最优参数重新评估最终状态以获取可视化数据...")
        original_debug_vis_state = DEBUG_VISUALIZE_FAILED_GRASPS
        DEBUG_VISUALIZE_FAILED_GRASPS = False
        final_cost_viz, final_pressures_viz, final_meshes_viz, final_contacts_viz = evaluate_grasp(
            r=best_r_bo,
            pos_idx1=best_pos_indices_bo[0],
            pos_idx2=best_pos_indices_bo[1],
            pos_idx3=best_pos_indices_bo[2]
        )
        DEBUG_VISUALIZE_FAILED_GRASPS = original_debug_vis_state

        best_gii_eval_bo = -final_cost_viz if final_cost_viz < -1e-9 else 0.

        print("\n找到的最优参数:")
        print(f"  r          = {best_r_bo:.4f}")
        print(f"  pos_idx1   = {best_pos_indices_bo[0]}")
        print(f"  pos_idx2   = {best_pos_indices_bo[1]}")
        print(f"  pos_idx3   = {best_pos_indices_bo[2]}")
        print(f"  (combo_idx = {best_combo_idx_bo})")
        print(f"优化找到成本 (BO func value): {best_c_bo:.4f}")
        print(f"重评成本 (from full eval): {final_cost_viz:.4f}")
        if best_gii_eval_bo > 0: print(f"重评GII (derived from cost): {best_gii_eval_bo:.4f}")
        else: print("重评状态无有效GII或GII为0。")

        print("\n使用最优参数生成最终可视化...")
        if final_meshes_viz and len(final_meshes_viz) == 3 and all(m is not None for m in final_meshes_viz):
            final_chosen_indices_bo_viz = best_pos_indices_bo

            pressures_display_str = "N/A"
            if final_pressures_viz is not None and hasattr(final_pressures_viz, 'size') and final_pressures_viz.size == 3:
                pressures_display_str = f"[{final_pressures_viz[0]:.0f}, {final_pressures_viz[1]:.0f}, {final_pressures_viz[2]:.0f}]"

            state_disp_bo=f"Cost={final_cost_viz:.3f}"; gii_disp_bo=f"GII={best_gii_eval_bo:.3f}" if best_gii_eval_bo>0 else "GII:N/A"

            # 使用新的美化绘图器
            plotter_final_pv = setup_publication_plotter("PyVista - Optimal Grasp", off_screen_default=False)

            plotter_final_pv.add_mesh(tray_pv_global, color=pv.Color(tray_color_viz, opacity=200), smooth_shading=True, name='tray_final_pv')

            if object_mesh_global_static:
                if 'colors' in object_mesh_global_static.point_data and object_mesh_global_static.point_data['colors'] is not None:
                    plotter_final_pv.add_mesh(object_mesh_global_static, scalars='colors', rgba=True, style='points', point_size=3.5, render_points_as_spheres=True, name='obj_final_pv')
                else:
                    plotter_final_pv.add_mesh(object_mesh_global_static, color=object_point_color_viz, style='points', point_size=3.5, render_points_as_spheres=True, name='obj_final_pv')

            # if pv_obb_object_mesh_global is not None:
                # plotter_final_pv.add_mesh(pv_obb_object_mesh_global, style='wireframe', color=object_obb_color_viz, line_width=2, name='object_obb_final_pv')

            if show_axes:
                plotter_final_pv.add_axes_at_origin(labels_off=True, line_width=1.5)

            for i_fv_bo,m_fv_bo in enumerate(final_meshes_viz):
                if m_fv_bo:
                    plotter_final_pv.add_mesh(m_fv_bo, color=finger_color_viz, style='surface', opacity=0.95, smooth_shading=True, show_edges=True, edge_color='dimgray', line_width=0.5, name=f'f_final_pv_{i_fv_bo}')

            params_txt_bo=f"r={best_r_bo:.2f}, P={pressures_display_str}, indices=({final_chosen_indices_bo_viz[0]},{final_chosen_indices_bo_viz[1]},{final_chosen_indices_bo_viz[2]})"
            plotter_final_pv.add_text(f"Optimal:{params_txt_bo}\n{state_disp_bo}\n{gii_disp_bo}", position="upper_left", font=font_family_viz, font_size=10, color=text_color_viz)

            plotter_final_pv.camera_position='xy'
            plotter_final_pv.camera.zoom(1.2)
            print("\n(PyVista) 显示最优抓取配置。请关闭窗口以继续。")
            plotter_final_pv.show(cpos='xy', auto_close=False)
            plotter_final_pv.close()

            visualize_poses_with_open3d(
                tray_geometry_transform_mm=T_actual_tray_geometry_world,
                tray_axes_transform_mm=T_tray_axes_vis_world,
                bottle_axes_transform_mm=T_object_axes_vis_world,
                bottle_points_world_mm=object_points_global_static,
                bottle_colors_rgb_float=sampled_colors_float_o3d,
                bottle_obb_world=world_obb_object_global,
                tray_radius_mm=tray_radius,
                tray_height_mm=tray_height,
                window_title="Open3D - Optimal Poses & Axes"
            )
        else:
            print("未能为最优参数生成手指网格或获取最终压力，无法进行PyVista最终可视化。")
            if final_cost_viz is not None : print(f"  (重评成本为: {final_cost_viz:.3f})")

    else:
        print("\nBO未成功或无结果，无法显示最优。")
    print("\n程序结束。")
