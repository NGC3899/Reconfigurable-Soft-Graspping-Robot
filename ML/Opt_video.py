# -*- coding: utf-8 -*-
import numpy as np
import pyvista as pv
import numpy.linalg as LA
import torch
import torch.nn as nn
import joblib
import sys
from scipy.spatial.distance import cdist
import traceback # 保留用于调试
import time
import os
import copy
import open3d as o3d # 确保 Open3D 已导入
import random # For selecting random distinct positions
import vtk # For VTK font constants
import imageio # 用于创建视频
# import shutil # 如果需要后续清理图片帧，可以取消注释

# --- 打印版本信息 ---
try:
    print(f"Simplified_Initial_Grasp_Search_Video_v6_final_hold_labels.py - Visualizing Initial Grasps with Enhancements")
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

# --- 3. 文件路径定义 ---
MODEL_PATH = 'best_mlp_model.pth'
X_SCALER_PATH = 'x_scaler.joblib'
Y_SCALER_PATH = 'y_scaler.joblib'
INITIAL_COORDS_PATH = 'initial_coordinates.txt'

GRASP_OUTPUTS_BASE_PATH = r"C:\Users\admin\Desktop\grasp_outputs" # <--- 修改为您的路径
RELATIVE_POSE_FILENAME = "relative_gripper_to_object_pose_CONDITIONAL_REORIENT.txt"

HIGH_RES_OBJECT_DESKTOP_PATH = r"C:\Users\admin\Desktop" # <--- 修改为您的路径
HIGH_RES_OBJECT_FILENAME = "Bird_Model.ply" # 确保此文件存在

RELATIVE_POSE_FILE_PATH = os.path.join(GRASP_OUTPUTS_BASE_PATH, RELATIVE_POSE_FILENAME)
HIGH_RES_OBJECT_PLY_PATH = os.path.join(HIGH_RES_OBJECT_DESKTOP_PATH, HIGH_RES_OBJECT_FILENAME)

# --- 4. 配置参数 ---
tray_radius = 60.0
tray_height = 1.0
tray_center = np.array([0.0, 0.0, -tray_height / 2.0])
finger_width = 10.0
TARGET_POINT_COUNT_FOR_SIM = 1000
show_axes = True

# --- 美化参数 ---
finger_color_viz = '#ff7f0e' # Matplotlib Orange
tray_color_viz = '#BDB7A4'   # Desaturated Tan/Khaki
object_point_color_viz = '#1f77b4' # Matplotlib Blue
contact_highlight_color_viz = '#9400D3' # DarkViolet for contact points
contact_label_color_viz = 'black' # 接触点标签颜色
object_obb_color_viz = '#2ca02c' # 保留以防万一
background_color_viz = '#EAEAEA' # Light Gray
text_color_viz = 'black'
font_family_viz = 'times' 
# --- end 美化参数 ---

collision_threshold = 1.0
overlap_threshold = 1e-4
max_pressure = 40000.0
NUM_INITIAL_SEARCHES = 2 
PRESSURE_STEP_INIT_SEARCH = 300.0
R_BOUNDS = (tray_radius * 0.3, tray_radius * 0.95)
P_BOUNDS = (0.0, max_pressure)
N_FINGER_SLOTS = 9
OBJECT_SCALE_FACTOR = 0.8

# --- 视频导出相关配置 ---
SAVE_VIDEO_FRAMES = True
VIDEO_OUTPUT_BASE_DIR = "initial_grasp_videos_final_hold_labels" # 更新输出目录名
VIDEO_FPS = 10 
VIDEO_CAMERA_INITIAL_ORBIT_DEGREES = -45.0 # 视频开始时相机旋转度数
VIDEO_HOLD_LAST_FRAME_DURATION = 5 # 最后状态定格秒数 (新增)
CLEANUP_FRAMES_AFTER_VIDEO = True

# --- 5. 辅助函数 ---
def o3d_create_transformation_matrix(R, t): T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t.flatten(); return T
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
def load_transformation_matrix_from_txt(file_path):
    if not os.path.exists(file_path): print(f"错误: 文件 '{file_path}' 未找到."); return None
    try: matrix = np.loadtxt(file_path)
    except Exception as e: print(f"加载 '{file_path}' 出错: {e}"); return None
    if matrix.shape == (4,4): print(f"成功从 '{file_path}' 加载变换矩阵."); return matrix
    else: print(f"错误: '{file_path}' 矩阵形状非(4,4)，而是{matrix.shape}."); return None
def load_initial_coordinates(file_path, expected_nodes):
    try: coords = np.loadtxt(file_path, dtype=np.float32, usecols=(0, 1, 2))
    except Exception as e: print(f"加载初始坐标出错: {e}"); return None
    if coords.shape == (expected_nodes, 3): print(f"成功加载 {coords.shape[0]} 初始节点坐标."); return coords
    else: print(f"错误: 坐标形状 {coords.shape} 与预期 ({expected_nodes},3) 不符."); return None
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
        except Exception as e_cdist: print(f"Error during cdist in sort_points_spatially: {e_cdist}"); break
        if distances.size == 0: break
        nearest_neighbor_relative_index = np.argmin(distances)
        nearest_neighbor_absolute_index = remaining_indices[nearest_neighbor_relative_index]
        sorted_indices.append(nearest_neighbor_absolute_index); current_index = nearest_neighbor_absolute_index
        if nearest_neighbor_absolute_index in remaining_indices: remaining_indices.pop(remaining_indices.index(nearest_neighbor_absolute_index))
    if len(sorted_indices) != num_points: print(f"Warning: Spatial sort processed {len(sorted_indices)} of {num_points} points.")
    return points[sorted_indices]

def get_rotation_matrix_between_vectors(vec1, vec2):
    norm_vec1 = np.linalg.norm(vec1); norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 < 1e-9 or norm_vec2 < 1e-9: return np.identity(3)
    a = vec1 / norm_vec1; b = vec2 / norm_vec2; v = np.cross(a, b)
    c = np.dot(a, b); s = np.linalg.norm(v)
    if s < 1e-9:
        if c > 0: return np.identity(3)
        else:
            if np.abs(a[0]) > 0.9 or np.abs(a[1]) > 0.9 : axis_ortho = np.array([0.0, 0.0, 1.0])
            else: axis_ortho = np.array([1.0, 0.0, 0.0])
            if np.linalg.norm(np.cross(a, axis_ortho)) < 1e-6: axis_ortho = np.array([0.0, 1.0, 0.0])
            return create_rotation_matrix(axis_ortho, np.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r_matrix = np.identity(3) + vx + vx @ vx * ((1 - c) / (s ** 2)); return r_matrix

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
    if model is None: print("模型未加载。"); return None
    pressure_value=np.clip(pressure_value,P_BOUNDS[0],P_BOUNDS[1]); input_p=np.array([[pressure_value]],dtype=np.float32)
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
    return None

# --- 全局变量 ---
initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global = [None]*5
object_points_global_static, object_centroid_global_static, object_mesh_global_static = [None]*3
num_object_points_global_static, faces_np_global, width_translation_vector_global = 0, None, None
T1_translate_global, T2_rotate_global = None, None
T_pose_for_tray_display_and_finger_placement_global = None
tray_pv = None

# --- 可视化辅助函数 ---
def setup_publication_plotter(title, window_size=[1000,800], off_screen_default=False):
    plotter_theme = pv.themes.DocumentTheme()
    plotter_theme.font.family = font_family_viz
    plotter_theme.font.color = text_color_viz
    plotter_theme.font.size = 12
    plotter_theme.font.label_size = 10
    plotter_theme.background = pv.Color(background_color_viz)
    plotter = pv.Plotter(window_size=window_size, theme=plotter_theme, title=title, off_screen=off_screen_default)
    plotter.enable_anti_aliasing('msaa', multi_samples=8)
    plotter.enable_parallel_projection()
    plotter.specular_power = 10.0
    plotter.remove_all_lights(); plotter.enable_lightkit()
    if plotter.renderer.cube_axes_actor is not None:
        actor = plotter.renderer.cube_axes_actor
        if hasattr(actor, 'GetXAxisCaptionActor2D'):
            for i in range(3):
                cap_prop = actor.GetCaptionTextProperty(i) if hasattr(actor, 'GetCaptionTextProperty') else None
                if cap_prop: cap_prop.SetFontFamily(vtk.VTK_TIMES); cap_prop.SetFontSize(10); cap_prop.SetColor(0,0,0); cap_prop.SetBold(0)
                else:
                    title_prop = actor.GetTitleTextProperty(i) if hasattr(actor, 'GetTitleTextProperty') else None
                    if title_prop: title_prop.SetFontFamily(vtk.VTK_TIMES); title_prop.SetFontSize(10); title_prop.SetColor(0,0,0); title_prop.SetBold(0)
        actor.SetXTitle(""); actor.SetYTitle(""); actor.SetZTitle("")
        if hasattr(actor, 'GetProperty'): actor.GetProperty().SetLineWidth(1.0)
    return plotter

def create_video_from_frames(image_files, video_path, fps):
    if not image_files: print(f"警告: 没有图片帧可用于创建视频 {video_path}"); return
    print(f"正在创建视频: {video_path} (包含 {len(image_files)} 帧, {fps} FPS)...")
    try:
        with imageio.get_writer(video_path, fps=fps, macro_block_size=1) as writer:
            for filename in image_files: writer.append_data(imageio.imread(filename))
        print(f"视频保存成功: {video_path}")
        if CLEANUP_FRAMES_AFTER_VIDEO:
            print(f"正在清理图片帧: {os.path.dirname(video_path)}")
            for filename in image_files:
                try: os.remove(filename)
                except OSError as e: print(f"  无法删除帧 {filename}: {e}")
            try:
                if not os.listdir(os.path.dirname(video_path)): os.rmdir(os.path.dirname(video_path)); print(f"已删除帧文件夹: {os.path.dirname(video_path)}")
            except OSError as e: print(f"  无法删除帧文件夹 {os.path.dirname(video_path)} (可能不为空): {e}")
    except Exception as e: print(f"创建视频 {video_path} 时发生错误: {e}")

def visualize_poses_with_open3d(
                                tray_geometry_transform_mm, tray_axes_transform_mm,     
                                bottle_axes_transform_mm, bottle_points_world_mm,     
                                bottle_colors_rgb_float, bottle_obb_world,          
                                tray_radius_mm, tray_height_mm,
                                window_title="Open3D Relative Pose Visualization"):
    print(f"\n--- 打开 Open3D 窗口: {window_title} ---")
    geometries = []
    o3d_tray_canonical_centered = o3d.geometry.TriangleMesh.create_cylinder(radius=tray_radius_mm, height=tray_height_mm, resolution=20, split=4)
    o3d_tray_canonical_centered.translate(np.array([0,0,-tray_height_mm/2.0]))
    o3d_tray_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_tray_canonical_centered)
    o3d_tray_wireframe.transform(tray_geometry_transform_mm)
    o3d_tray_wireframe.paint_uniform_color(pv.Color(tray_color_viz).float_rgb)
    geometries.append(o3d_tray_wireframe)
    o3d_bottle_pcd = o3d.geometry.PointCloud()
    o3d_bottle_pcd.points = o3d.utility.Vector3dVector(bottle_points_world_mm)
    if bottle_colors_rgb_float is not None and len(bottle_colors_rgb_float) == len(bottle_points_world_mm):
        o3d_bottle_pcd.colors = o3d.utility.Vector3dVector(bottle_colors_rgb_float)
    else: o3d_bottle_pcd.paint_uniform_color(pv.Color(object_point_color_viz).float_rgb)
    geometries.append(o3d_bottle_pcd)
    if bottle_obb_world is not None: geometries.append(bottle_obb_world)
    tray_axes_size = tray_radius_mm * 0.7
    o3d_tray_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=tray_axes_size, origin=[0,0,0])
    o3d_tray_axes.transform(tray_axes_transform_mm); geometries.append(o3d_tray_axes)
    if bottle_points_world_mm.shape[0] > 0:
        temp_bottle_pcd_for_bbox = o3d.geometry.PointCloud(); temp_bottle_pcd_for_bbox.points = o3d.utility.Vector3dVector(bottle_points_world_mm)
        bbox_bottle = temp_bottle_pcd_for_bbox.get_axis_aligned_bounding_box()
        diag_len = LA.norm(bbox_bottle.get_max_bound() - bbox_bottle.get_min_bound())
        bottle_axes_size = diag_len * 0.25 if diag_len > 1e-2 else tray_axes_size * 0.8
    else: bottle_axes_size = tray_axes_size * 0.8
    bottle_axes_size = max(bottle_axes_size, 5.0)
    o3d_bottle_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=bottle_axes_size, origin=[0,0,0])
    o3d_bottle_axes.transform(bottle_axes_transform_mm); geometries.append(o3d_bottle_axes)
    world_axes_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=tray_radius_mm * 1.2, origin=[0,0,0])
    geometries.append(world_axes_o3d)
    o3d.visualization.draw_geometries(geometries, window_name=window_title, width=1000, height=800)
    print(f"--- 关闭 Open3D 窗口: {window_title} ---")

# --- 初始点检测函数 (核心逻辑，包含视频帧生成和接触点高亮) ---
def find_initial_grasp(initial_r, initial_finger_indices, pressure_step, max_pressure_init, attempt_idx, 
                       video_camera_initial_orbit_degrees=0.0): 
    global initial_coords_ref_global, model_global, scaler_X_global, scaler_y_global, device_global, \
           object_points_global_static, num_object_points_global_static, faces_np_global, \
           width_translation_vector_global, T1_translate_global, T2_rotate_global, \
           T_pose_for_tray_display_and_finger_placement_global, N_FINGER_SLOTS, \
           VIDEO_FPS, VIDEO_OUTPUT_BASE_DIR, SAVE_VIDEO_FRAMES, CLEANUP_FRAMES_AFTER_VIDEO, \
           tray_pv, object_mesh_global_static, contact_highlight_color_viz, object_centroid_global_static, \
           contact_label_color_viz, font_family_viz, VIDEO_HOLD_LAST_FRAME_DURATION # 新增全局变量

    print(f"\n--- 开始寻找初始接触点 (尝试 {attempt_idx+1}): r={initial_r:.2f}, indices={initial_finger_indices}, initial_cam_orbit={video_camera_initial_orbit_degrees}° ---")
    current_pressures_init = np.array([0.0, 0.0, 0.0])
    finger_contact_achieved_init = [False, False, False]
    init_iter_count = 0
    last_meshes = [None] * 3
    last_point_status = ['Non-Contact'] * num_object_points_global_static if num_object_points_global_static > 0 else []
    last_closest_contacts = [None] * 3
    dot_prod_tolerance = 1e-6
    max_iterations = int( (max_pressure_init / pressure_step) * 3 * 1.5) + 20

    image_files = []; frame_output_dir = ""; video_plotter = None; result_dict = {}
    
    if SAVE_VIDEO_FRAMES:
        param_str_for_filename = f"r{initial_r:.1f}_pos{''.join(map(str, initial_finger_indices))}"
        frame_output_dir = os.path.join(VIDEO_OUTPUT_BASE_DIR, f"attempt_{attempt_idx:03d}_{param_str_for_filename}")
        os.makedirs(frame_output_dir, exist_ok=True)
        video_plotter = setup_publication_plotter(
            f"Initial Search Attempt {attempt_idx+1} - Frame",
            window_size=[800, 600], off_screen_default=True
        )
        if tray_pv is not None: video_plotter.add_mesh(tray_pv, color=pv.Color(tray_color_viz, opacity=200), smooth_shading=True, name='tray_video')
        object_actor_name_video = 'obj_video_main' 
        if object_mesh_global_static:
            if 'colors' in object_mesh_global_static.point_data and object_mesh_global_static.point_data['colors'] is not None:
                video_plotter.add_mesh(object_mesh_global_static, scalars='colors', rgba=True, style='points', point_size=3.5, render_points_as_spheres=True, name=object_actor_name_video)
            else:
                video_plotter.add_mesh(object_mesh_global_static, color=object_point_color_viz, style='points', point_size=3.5, render_points_as_spheres=True, name=object_actor_name_video)
        if show_axes: video_plotter.add_axes_at_origin(line_width=1.5, labels_off=True)
        
        video_plotter.camera.azimuth = 45 
        video_plotter.camera.elevation = 25
        video_plotter.camera.zoom(1.3)
        if object_centroid_global_static is not None:
            video_plotter.set_focus(object_centroid_global_static)
        
        if video_camera_initial_orbit_degrees != 0.0:
            video_plotter.camera.azimuth += video_camera_initial_orbit_degrees
            print(f"  应用初始相机旋转 {video_camera_initial_orbit_degrees}°. 新方位角: {video_plotter.camera.azimuth:.1f}°")

    try:
        while True:
            init_iter_count += 1
            if init_iter_count > max_iterations:
                result_dict = { 'status': 'IterationLimit', 'r': initial_r, 'finger_indices': initial_finger_indices, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }; break
            pressure_changed_init = False
            for i in range(3):
                if not finger_contact_achieved_init[i] and current_pressures_init[i] < max_pressure_init:
                    current_pressures_init[i] += pressure_step
                    current_pressures_init[i] = min(current_pressures_init[i], max_pressure_init)
                    pressure_changed_init = True
            if not pressure_changed_init and all(finger_contact_achieved_init): pass
            elif not pressure_changed_init and not all(finger_contact_achieved_init):
                result_dict = { 'status': 'MaxPressureReachedOrNoChange', 'r': initial_r, 'finger_indices': initial_finger_indices, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }; break

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
                current_pos_index = initial_finger_indices[i]; current_angle_deg = current_pos_index * (360.0 / N_FINGER_SLOTS)
                angle_rad = np.radians(current_angle_deg); rot_angle_z_placing = angle_rad + np.pi / 2.0
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
                    deformed_finger_meshes_init[i] = final_transformed_finger_mesh
                except Exception: valid_preds = False; break
            if not valid_preds:
                result_dict = { 'status': 'PredictionFailed', 'r': initial_r, 'finger_indices': initial_finger_indices, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }; break
            last_meshes = [mesh.copy() if mesh is not None else None for mesh in deformed_finger_meshes_init]

            current_point_status = ['Non-Contact'] * num_object_points_global_static if num_object_points_global_static > 0 else []
            has_overlap_init = False; num_contact_points_init = 0
            finger_dot_products_init = [[] for _ in range(3)]; contacting_fingers_indices_this_iter = set()
            contacting_object_point_indices = [] 

            if object_points_global_static is not None and num_object_points_global_static > 0:
                for obj_pt_idx, obj_point in enumerate(object_points_global_static):
                    closest_dist_for_this_pt = float('inf'); finger_idx_for_this_pt = -1; normal_for_this_pt = None; pt_on_mesh_for_this_pt = None
                    for finger_idx, finger_mesh in enumerate(deformed_finger_meshes_init):
                        if finger_mesh is None or finger_mesh.n_cells == 0: continue
                        has_normals_loc = 'Normals' in finger_mesh.cell_data and finger_mesh.cell_data['Normals'] is not None
                        try:
                            closest_cell_id_loc, pt_on_mesh_candidate = finger_mesh.find_closest_cell(obj_point, return_closest_point=True)
                            if closest_cell_id_loc < 0 or closest_cell_id_loc >= finger_mesh.n_cells: continue
                            dist = LA.norm(obj_point - pt_on_mesh_candidate)
                            if dist < closest_dist_for_this_pt:
                                closest_dist_for_this_pt = dist; finger_idx_for_this_pt = finger_idx; pt_on_mesh_for_this_pt = pt_on_mesh_candidate; current_normal = None
                                if has_normals_loc and closest_cell_id_loc < len(finger_mesh.cell_data['Normals']): current_normal = finger_mesh.cell_normals[closest_cell_id_loc]
                                normal_for_this_pt = current_normal
                        except Exception: continue
                    if finger_idx_for_this_pt != -1:
                        dist_final_for_pt = closest_dist_for_this_pt; normal_final_for_pt = normal_for_this_pt
                        if dist_final_for_pt < overlap_threshold: current_point_status[obj_pt_idx] = 'Overlap'; has_overlap_init = True
                        elif dist_final_for_pt < collision_threshold:
                            current_point_status[obj_pt_idx] = 'Contact'; num_contact_points_init += 1; contacting_fingers_indices_this_iter.add(finger_idx_for_this_pt)
                            contacting_object_point_indices.append(obj_pt_idx)
                            if normal_final_for_pt is not None and LA.norm(normal_final_for_pt) > 1e-9:
                                vector_to_point = obj_point - pt_on_mesh_for_this_pt
                                if LA.norm(vector_to_point) > 1e-9:
                                    dot_prod = np.dot(vector_to_point/LA.norm(vector_to_point), normal_final_for_pt/LA.norm(normal_final_for_pt))
                                    finger_dot_products_init[finger_idx_for_this_pt].append(dot_prod)
                    if has_overlap_init: break
            last_point_status = current_point_status # 更新全局状态以供返回

            if SAVE_VIDEO_FRAMES and video_plotter is not None:
                print(f"  初始点搜索 Iter {init_iter_count}: P={current_pressures_init.round(0)}, Contacts={num_contact_points_init}, saving frame...")
                for finger_plot_idx_vid in range(3): video_plotter.remove_actor(f'finger_video_{finger_plot_idx_vid}')
                video_plotter.remove_actor("iteration_text_video")
                video_plotter.remove_actor("contact_points_video") 
                video_plotter.remove_actor("contact_point_labels_video") # 移除上一帧的接触点标签

                for finger_plot_idx_vid, finger_mesh_viz_vid in enumerate(deformed_finger_meshes_init):
                    if finger_mesh_viz_vid is not None:
                        video_plotter.add_mesh(finger_mesh_viz_vid, color=finger_color_viz, style='surface', opacity=0.95,
                                             smooth_shading=True, show_edges=True, edge_color='dimgray', line_width=0.5,
                                             name=f'finger_video_{finger_plot_idx_vid}')
                
                contact_point_labels = []
                if contacting_object_point_indices and object_points_global_static is not None:
                    contact_pts_coords = object_points_global_static[contacting_object_point_indices]
                    if contact_pts_coords.size > 0: 
                        contact_pts_polydata = pv.PolyData(contact_pts_coords)
                        video_plotter.add_mesh(contact_pts_polydata, color=contact_highlight_color_viz, 
                                               style='points', point_size=5.0, render_points_as_spheres=True,
                                               name="contact_points_video")
                        # 为接触点添加坐标标签
                        for pt_coord in contact_pts_coords:
                            contact_point_labels.append(f"({pt_coord[0]:.1f}, {pt_coord[1]:.1f}, {pt_coord[2]:.1f})")
                        if contact_point_labels:
                             video_plotter.add_point_labels(
                                 contact_pts_coords, 
                                 contact_point_labels,
                                 name="contact_point_labels_video",
                                 font_size=8, # 较小字体
                                 font_family=font_family_viz, # 'times'
                                 text_color=contact_label_color_viz, # 黑色
                                 shape_opacity=0, # 不显示标签背景形状
                                 show_points=False, # 不在标签位置额外绘制点
                                 always_visible=True # 尝试让标签总是可见
                             )
                
                iter_text_vid = f"Attempt: {attempt_idx+1}, Iter: {init_iter_count}\nP: {current_pressures_init.round(0)}\nContacts: {num_contact_points_init}"
                video_plotter.add_text(iter_text_vid, position="upper_left", font=font_family_viz, font_size=10, color=text_color_viz, name="iteration_text_video")
                                
                img_path = os.path.join(frame_output_dir, f"frame_{init_iter_count:04d}.png"); image_files.append(img_path)
                video_plotter.screenshot(img_path)

            if has_overlap_init:
                result_dict = { 'status': 'Overlap', 'r': initial_r, 'finger_indices': initial_finger_indices, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }; break
            finger_intersects_init = [False] * 3
            if num_contact_points_init > 0:
                for i in range(3):
                    if finger_dot_products_init[i]:
                        has_pos_dp = any(dp > dot_prod_tolerance for dp in finger_dot_products_init[i])
                        has_neg_dp = any(dp < -dot_prod_tolerance for dp in finger_dot_products_init[i])
                        if has_pos_dp and has_neg_dp: finger_intersects_init[i] = True
            if any(finger_intersects_init):
                result_dict = { 'status': 'Intersection', 'r': initial_r, 'finger_indices': initial_finger_indices, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }; break
            for i in range(3):
                if i in contacting_fingers_indices_this_iter: finger_contact_achieved_init[i] = True
            print(f"  初始点搜索 Iter {init_iter_count}: 接触状态={list(finger_contact_achieved_init)}")
            if all(finger_contact_achieved_init):
                result_dict = { 'status': 'FoundInitialPoint', 'r': initial_r, 'finger_indices': initial_finger_indices, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }; break
            if not pressure_changed_init and not all(finger_contact_achieved_init):
                result_dict = { 'status': 'MaxPressureReachedOrNoChangeLoopEnd', 'r': initial_r, 'finger_indices': initial_finger_indices, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }; break
        if not result_dict: # 如果循环因其他原因结束（例如，valid_preds为False后）
             result_dict = { 'status': 'LoopEndUnknown', 'r': initial_r, 'finger_indices': initial_finger_indices, 'pressures': current_pressures_init.tolist(), 'finger_meshes': last_meshes, 'object_point_status': last_point_status, 'contact_info': last_closest_contacts }
    finally:
        if SAVE_VIDEO_FRAMES and image_files:
            # 在创建视频前，复制最后一帧以实现定格效果
            if VIDEO_HOLD_LAST_FRAME_DURATION > 0 and image_files:
                last_frame_path = image_files[-1]
                num_hold_frames = int(VIDEO_FPS * VIDEO_HOLD_LAST_FRAME_DURATION)
                print(f"  为最后状态定格 {VIDEO_HOLD_LAST_FRAME_DURATION} 秒 (复制 {num_hold_frames} 帧)...")
                image_files.extend([last_frame_path] * num_hold_frames)

            status_for_filename = result_dict.get('status', 'UnknownStatus')
            param_str_for_filename = f"r{initial_r:.1f}_pos{''.join(map(str, initial_finger_indices))}"
            video_file_name = f"attempt_{attempt_idx:03d}_{param_str_for_filename}_{status_for_filename}.mp4"
            video_path = os.path.join(VIDEO_OUTPUT_BASE_DIR, video_file_name)
            create_video_from_frames(image_files, video_path, VIDEO_FPS)
        if video_plotter is not None:
            video_plotter.close()
            del video_plotter # 明确删除，帮助垃圾回收
    return result_dict

# --- 主脚本 ---
if __name__ == '__main__':
    T_gn_gripper_TO_gn_object_meters = load_transformation_matrix_from_txt(RELATIVE_POSE_FILE_PATH)
    if T_gn_gripper_TO_gn_object_meters is None: sys.exit()
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
    print(f"\n已加载并处理 '{HIGH_RES_OBJECT_FILENAME}'。用于仿真点数: {num_object_points_global_static}")

    world_obb_object_global_for_o3d = None
    if num_object_points_global_static > 0:
        o3d_temp_pcd_for_obb = o3d.geometry.PointCloud(); o3d_temp_pcd_for_obb.points = o3d.utility.Vector3dVector(object_points_global_static)
        world_obb_object_global_for_o3d = o3d_temp_pcd_for_obb.get_oriented_bounding_box()
        world_obb_object_global_for_o3d.color = pv.Color(object_obb_color_viz).float_rgb

    initial_coords_ref_global = load_initial_coordinates(INITIAL_COORDS_PATH, NODE_COUNT)
    model_global,scaler_X_global,scaler_y_global,device_global = load_prediction_components(MODEL_PATH,X_SCALER_PATH,Y_SCALER_PATH,INPUT_DIM,OUTPUT_DIM,HIDDEN_LAYER_1,HIDDEN_LAYER_2,HIDDEN_LAYER_3)
    if initial_coords_ref_global is None or model_global is None: sys.exit("错误：未能初始化手指模型或坐标。")
    faces_np_global = create_faces_array(NODE_COUNT)
    width_translation_vector_global = np.array([0, finger_width, 0])
    bottom_node_idx = np.argmin(initial_coords_ref_global[:,0]); ref_mid_pt = initial_coords_ref_global[bottom_node_idx] + width_translation_vector_global/2.0
    T1_translate_global = create_transformation_matrix_opt8(None,-ref_mid_pt)
    rot_ref_to_local = np.array([[0,1,0],[0,0,1],[1,0,0]]); T2_rotate_global = create_transformation_matrix_opt8(rot_ref_to_local,None)
    
    if object_points_global_static is None: sys.exit("错误：object_points_global_static 未初始化。")
    
    tray_pv_canonical = pv.Cylinder(center=(0,0,0), radius=tray_radius, height=tray_height, direction=(0,0,1), resolution=30)
    tray_pv = tray_pv_canonical.transform(T_actual_tray_geometry_world, inplace=False)

    if SAVE_VIDEO_FRAMES:
        os.makedirs(VIDEO_OUTPUT_BASE_DIR, exist_ok=True)
        print(f"视频帧和视频将保存在: {os.path.abspath(VIDEO_OUTPUT_BASE_DIR)}")

    print(f"\n--- 开始在 {NUM_INITIAL_SEARCHES} 个随机 (r, finger_indices) 组合上寻找初始接触点并生成视频 ---")
    last_search_attempt_result = None

    x_coords_rel_new_obj_centroid = object_points_global_static[:, 0] - object_centroid_global_static[0]
    y_coords_rel_new_obj_centroid = object_points_global_static[:, 1] - object_centroid_global_static[1]
    distances_from_new_obj_centroid_xy = np.sqrt(x_coords_rel_new_obj_centroid**2 + y_coords_rel_new_obj_centroid**2)
    if distances_from_new_obj_centroid_xy.size > 0:
        effective_radius_of_object_at_old_tray_pos = np.max(distances_from_new_obj_centroid_xy)
        r_search_min_heuristic = effective_radius_of_object_at_old_tray_pos * 0.7
        r_search_max_heuristic = tray_radius * 1.1
    else: r_search_min_heuristic = R_BOUNDS[0]; r_search_max_heuristic = R_BOUNDS[1]
    r_search_min = np.clip(r_search_min_heuristic, R_BOUNDS[0], R_BOUNDS[1] * 0.95)
    r_search_max = np.clip(r_search_max_heuristic, r_search_min + (R_BOUNDS[1] - R_BOUNDS[0]) * 0.05, R_BOUNDS[1])
    if r_search_min >= r_search_max: r_search_min=R_BOUNDS[0]; r_search_max=R_BOUNDS[1]
    print(f"初始搜索范围: r in [{r_search_min:.2f}, {r_search_max:.2f}]")

    for k_init in range(NUM_INITIAL_SEARCHES):
        r_val = np.random.uniform(r_search_min,r_search_max)
        possible_indices = list(range(N_FINGER_SLOTS))
        chosen_k_init_indices = random.sample(possible_indices, 3); chosen_k_init_indices.sort()
        
        search_res = find_initial_grasp(
            initial_r=r_val,
            initial_finger_indices=chosen_k_init_indices,
            pressure_step=PRESSURE_STEP_INIT_SEARCH,
            max_pressure_init=max_pressure,
            attempt_idx=k_init,
            video_camera_initial_orbit_degrees=VIDEO_CAMERA_INITIAL_ORBIT_DEGREES 
        )
        last_search_attempt_result = search_res
        
        if search_res:
            print(f"--- 初始搜索尝试 {k_init+1}/{NUM_INITIAL_SEARCHES} 完成，状态: {search_res.get('status', 'Unknown')} ---")
        else:
            print(f"--- 初始搜索尝试 {k_init+1}/{NUM_INITIAL_SEARCHES} 未返回有效结果 ---")

    print(f"\n--- 所有 {NUM_INITIAL_SEARCHES} 次初始点检测已完成 ---")

    if last_search_attempt_result and 'finger_meshes' in last_search_attempt_result and last_search_attempt_result['finger_meshes'] is not None :
        print("\n(PyVista) 可视化最后一次初始搜索尝试的状态...")
        title_init_pv_viz = f"PyVista - Last Initial Search Attempt (Status: {last_search_attempt_result.get('status','N/A')})"
        plotter_init_pv = setup_publication_plotter(title_init_pv_viz, off_screen_default=False)

        plotter_init_pv.add_mesh(tray_pv, color=pv.Color(tray_color_viz, opacity=200), smooth_shading=True, name='tray_init_pv')
        
        main_object_actor_name = "main_object_interactive"
        contact_points_actor_name = "contact_points_interactive"

        if object_mesh_global_static:
            if 'colors' in object_mesh_global_static.point_data and object_mesh_global_static.point_data['colors'] is not None:
                plotter_init_pv.add_mesh(object_mesh_global_static, scalars='colors', rgba=True, style='points', point_size=3.5, render_points_as_spheres=True, name=main_object_actor_name)
            else:
                plotter_init_pv.add_mesh(object_mesh_global_static, color=object_point_color_viz, style='points', point_size=3.5, render_points_as_spheres=True, name=main_object_actor_name)

            final_point_status = last_search_attempt_result.get('object_point_status', [])
            contacting_indices_interactive = [idx for idx, status in enumerate(final_point_status) if status == 'Contact']
            if contacting_indices_interactive and object_points_global_static is not None:
                 contact_pts_coords_interactive = object_points_global_static[contacting_indices_interactive]
                 if contact_pts_coords_interactive.size > 0:
                    contact_pts_polydata_interactive = pv.PolyData(contact_pts_coords_interactive)
                    plotter_init_pv.add_mesh(contact_pts_polydata_interactive, color=contact_highlight_color_viz, 
                                           style='points', point_size=5.0, render_points_as_spheres=True,
                                           name=contact_points_actor_name)
                    # 为交互式可视化也添加接触点标签
                    contact_labels_interactive = [f"({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})" for p in contact_pts_coords_interactive]
                    plotter_init_pv.add_point_labels(contact_pts_coords_interactive, contact_labels_interactive,
                                                     name="contact_labels_interactive", font_size=8, font_family=font_family_viz,
                                                     text_color=contact_label_color_viz, shape_opacity=0, show_points=False, always_visible=True)

        if show_axes:
            plotter_init_pv.add_axes_at_origin(line_width=1.5, labels_off=True)

        meshes_viz_i_pv = last_search_attempt_result.get('finger_meshes',[])
        if meshes_viz_i_pv:
            for i,m_viz_i_pv in enumerate(meshes_viz_i_pv):
                if m_viz_i_pv is not None:
                    plotter_init_pv.add_mesh(m_viz_i_pv, color=finger_color_viz, style='surface', opacity=0.95,
                                             smooth_shading=True, show_edges=True, edge_color='dimgray', line_width=0.5,
                                             name=f'f_init_pv_{i}')
        
        r_disp = last_search_attempt_result.get('r','?'); indices_disp = last_search_attempt_result.get('finger_indices', ['?']*3)
        pressures_disp = np.array(last_search_attempt_result.get('pressures',['?']*3)).round(0); status_disp = last_search_attempt_result.get('status','N/A')
        status_txt_i_pv = f"Last Search: r={r_disp:.2f}, indices=({indices_disp[0]},{indices_disp[1]},{indices_disp[2]}), P={pressures_disp}\nStatus: {status_disp}"
        plotter_init_pv.add_text(status_txt_i_pv, position="upper_left", font=font_family_viz, font_size=10, color=text_color_viz, name="status_text_init")
        
        plotter_init_pv.camera.azimuth = 45
        plotter_init_pv.camera.elevation = 25
        plotter_init_pv.camera.zoom(1.3)
        print("(PyVista) 显示最后初始搜索结果。按Q关闭此窗口。")
        plotter_init_pv.show(cpos=None, auto_close=False)

        visualize_poses_with_open3d(
            tray_geometry_transform_mm=T_actual_tray_geometry_world, tray_axes_transform_mm=T_tray_coord_system_in_world,
            bottle_axes_transform_mm=T_object_coord_system_in_world, bottle_points_world_mm=object_points_global_static,
            bottle_colors_rgb_float=sampled_colors_float_o3d, bottle_obb_world=world_obb_object_global_for_o3d,
            tray_radius_mm=tray_radius, tray_height_mm=tray_height,
            window_title="Open3D - Last Initial Search Poses & Axes"
        )
    else:
        print("警告：最后一次初始搜索结果不完整或物体未加载，无法进行PyVista和Open3D可视化。")

    print("\n脚本结束。")
