# -*- coding: utf-8 -*-
import numpy as np
import pyvista as pv
import numpy.linalg as LA
import torch
import torch.nn as nn
import joblib
import sys
from scipy.spatial.distance import cdist # 用于高效计算距离

# --- 1. ML 模型定义 ---
class MLPRegression(nn.Module):
    def __init__(self, input_dim, output_dim, h1, h2, h3):
        super(MLPRegression, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, h3), nn.Tanh(),
            nn.Linear(h3, output_dim)
        )
    def forward(self, x): return self.network(x)

# --- 2. ML 相关参数定义 ---
INPUT_DIM = 1; NODE_COUNT = 63; OUTPUT_DIM = NODE_COUNT * 3
HIDDEN_LAYER_1 = 128; HIDDEN_LAYER_2 = 256; HIDDEN_LAYER_3 = 128 # <<< 确认

# --- 3. 文件路径定义 ---
MODEL_PATH = 'best_mlp_model.pth'; X_SCALER_PATH = 'x_scaler.joblib'
Y_SCALER_PATH = 'y_scaler.joblib'; INITIAL_COORDS_PATH = 'initial_coordinates.txt'

# --- 4. 配置参数 ---
tray_radius = 50.0; tray_height = 10.0
tray_center = np.array([0.0, 0.0, -tray_height / 2.0])
finger_width = 10.0
finger_placement_radius = tray_radius * 0.7 # <<< 可调半径 r
sphere_radius = 15.0; num_sphere_points = 400
sphere_rotation_angle_z = np.radians(30)
sphere_translation = np.array([0, 0, sphere_radius + 5.0])
show_axes = True; finger_color = 'lightcoral'; tray_color = 'tan'; sphere_color = 'blue'

# --- 5. 辅助函数 ---
def create_rotation_matrix(axis, angle_rad):
    axis = np.asarray(axis).astype(float); axis /= LA.norm(axis)
    a = np.cos(angle_rad / 2.0); b, c, d = -axis * np.sin(angle_rad / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d; bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
def create_rotation_matrix_x(a): return create_rotation_matrix([1,0,0], a)
def create_rotation_matrix_y(a): return create_rotation_matrix([0,1,0], a)
def create_rotation_matrix_z(a): return create_rotation_matrix([0,0,1], a)
def create_translation_matrix(t): matrix=np.identity(4); matrix[:3, 3]=t; return matrix
def create_transformation_matrix(r, t):
    matrix=np.identity(4)
    if r is not None: matrix[:3, :3]=r
    if t is not None: matrix[:3, 3]=t
    return matrix
def generate_sphere_points(radius, num_points):
    points = []; phi = np.pi * (3. - np.sqrt(5.))
    for i in range(num_points):
        z = 1 - (i / float(num_points - 1)) * 2
        radius_at_z = np.sqrt(1 - z * z); theta = phi * i
        x = np.cos(theta) * radius_at_z; y = np.sin(theta) * radius_at_z
        points.append((x, y, z));
    return np.array(points) * radius
def transform_points(points, matrix):
    if points.shape[1] != 3: raise ValueError("Input points shape");
    if matrix.shape != (4, 4): raise ValueError("Matrix shape");
    num_pts = points.shape[0]; h_points = np.hstack((points, np.ones((num_pts, 1))))
    t_h = h_points @ matrix.T; return t_h[:, :3] / t_h[:, 3, np.newaxis]

def load_initial_coordinates(file_path, expected_nodes):
    try:
        coords = np.loadtxt(file_path, dtype=np.float32, usecols=(0, 1, 2))
        if coords.shape == (expected_nodes, 3):
            print(f"成功加载 {coords.shape[0]} 个初始节点坐标。")
            return coords
        else: print(f"错误：坐标形状 {coords.shape} 与预期 ({expected_nodes}, 3) 不符。"); return None
    except FileNotFoundError: print(f"错误：找不到文件 {file_path}"); return None
    except Exception as e: print(f"加载初始坐标时出错: {e}"); return None

def create_faces_array(num_nodes_per_curve):
    faces = []; num_quads = num_nodes_per_curve - 1
    if num_quads <= 0: return np.array([], dtype=int)
    for i in range(num_quads):
        p1, p2 = i, i + 1; p3, p4 = (i + 1) + num_nodes_per_curve, i + num_nodes_per_curve
        faces.append([4, p1, p2, p3, p4]);
    return np.hstack(faces)

def sort_points_spatially(points):
    """使用近邻搜索对点进行空间排序"""
    if points is None or points.shape[0] < 2:
        return points

    num_points = points.shape[0]
    sorted_indices = []
    remaining_indices = list(range(num_points))

    # 启发式选择起点：选择 X 坐标最小的点作为起点
    # (这个假设需要根据点的实际分布调整，例如选择 Z 最小或离某个角点最近的点)
    start_node_index = np.argmin(points[:, 0]) # 假设 X 最小是起点
    current_index = start_node_index

    sorted_indices.append(current_index)
    remaining_indices.pop(remaining_indices.index(current_index)) # 从列表中移除

    while remaining_indices:
        last_point = points[current_index, np.newaxis] # (1, 3) shape
        remaining_points = points[remaining_indices]   # (N_rem, 3) shape

        # 计算最后一个已排序点到所有剩余点的距离
        distances = cdist(last_point, remaining_points)[0] # Shape (N_rem,)

        # 找到最近邻点的索引 (在 remaining_points 中的相对索引)
        nearest_neighbor_relative_index = np.argmin(distances)
        # 获取该点在原始 points 数组中的绝对索引
        nearest_neighbor_absolute_index = remaining_indices[nearest_neighbor_relative_index]

        # 添加到排序列表，并更新当前点
        sorted_indices.append(nearest_neighbor_absolute_index)
        current_index = nearest_neighbor_absolute_index
        # 从剩余索引中移除
        remaining_indices.pop(nearest_neighbor_relative_index) # 按相对索引移除

    # 根据排序后的索引重新排列原始点数组
    return points[sorted_indices]

# --- ML 加载和预测函数 ---
def load_prediction_components(model_path, x_scaler_path, y_scaler_path, input_dim, output_dim, h1, h2, h3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"ML使用设备: {device}")
    model = MLPRegression(input_dim, output_dim, h1, h2, h3)
    try: model.load_state_dict(torch.load(model_path, map_location=device)); model.to(device); model.eval()
    except Exception as e: print(f"加载模型 {model_path} 时出错: {e}"); return None, None, None, None
    scaler_X, scaler_y = None, None
    try: scaler_X = joblib.load(x_scaler_path); print(f"X Scaler 加载成功。")
    except FileNotFoundError: print(f"警告: 未找到 X scaler '{x_scaler_path}'。")
    except Exception as e: print(f"加载 X Scaler '{x_scaler_path}' 时出错: {e}"); return None, None, None, None
    try: scaler_y = joblib.load(y_scaler_path); print(f"Y Scaler 加载成功。")
    except FileNotFoundError: print(f"警告: 未找到 Y scaler '{y_scaler_path}'。")
    except Exception as e: print(f"加载 Y Scaler '{y_scaler_path}' 时出错: {e}"); return None, None, None, None
    return model, scaler_X, scaler_y, device

def predict_displacements_for_pressure(model, scaler_X, scaler_y, device, pressure_value):
    if model is None: print("模型未加载。"); return None
    input_p = np.array([[pressure_value]], dtype=np.float32)
    if scaler_X:
        try: input_p_scaled = scaler_X.transform(input_p)
        except Exception as e: print(f"X scaler 标准化出错: {e}"); return None
    else: input_p_scaled = input_p
    input_tensor = torch.tensor(input_p_scaled, dtype=torch.float32).to(device)
    predicted_original_scale = None
    with torch.no_grad():
        try:
            predicted_scaled_tensor = model(input_tensor); predicted_scaled = predicted_scaled_tensor.cpu().numpy()
            if scaler_y:
                try: predicted_original_scale = scaler_y.inverse_transform(predicted_scaled)
                except Exception as e: print(f"Y scaler 反标准化出错: {e}"); return None
            else: predicted_original_scale = predicted_scaled
        except Exception as e: print(f"模型预测出错: {e}"); return None
    if predicted_original_scale is not None:
        if predicted_original_scale.shape[1] != OUTPUT_DIM: print(f"错误：模型输出维度错误"); return None
        return predicted_original_scale.reshape(NODE_COUNT, 3)
    else: return None

# --- 6. 主脚本 ---
if __name__ == '__main__':
    # --- 加载初始坐标参照系 ---
    initial_coords_ref = load_initial_coordinates(INITIAL_COORDS_PATH, NODE_COUNT)
    if initial_coords_ref is None: sys.exit()

    # --- 初始化 ML 组件 ---
    model, scaler_X, scaler_y, device = load_prediction_components( MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH, INPUT_DIM, OUTPUT_DIM, HIDDEN_LAYER_1, HIDDEN_LAYER_2, HIDDEN_LAYER_3 )
    if model is None: sys.exit()

    # --- 定义参考系信息 ---
    width_translation_vector = np.array([0, finger_width, 0]) # <<< 确认宽度方向
    bottom_node_index_ref = np.argmin(initial_coords_ref[:, 0]) # <<< 确认 X 最小代表底部
    ref_bottom_midpoint = initial_coords_ref[bottom_node_index_ref] + width_translation_vector / 2.0

    # --- 定义坐标系变换 ---
    T1_translate = create_translation_matrix(-ref_bottom_midpoint)
    rotation_ref_to_local = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) # Y_ref->X_loc, Z_ref->Y_loc, X_ref->Z_loc
    T2_rotate = create_transformation_matrix(rotation_ref_to_local, None)

    # --- 创建基础场景几何 ---
    tray = pv.Cylinder(center=tray_center, radius=tray_radius, height=tray_height, direction=(0, 0, 1), resolution=100)
    sphere_points_local = generate_sphere_points(sphere_radius, num_sphere_points)
    sphere_transform_matrix = create_transformation_matrix(create_rotation_matrix_z(sphere_rotation_angle_z), sphere_translation)
    sphere_points_global = transform_points(sphere_points_local, sphere_transform_matrix)
    sphere_point_cloud_actor = pv.PolyData(sphere_points_global)

    # --- 预生成面片连接信息 ---
    faces_np = create_faces_array(NODE_COUNT)
    if faces_np.size == 0: print("错误：无法生成面片连接信息。"); sys.exit()

    # --- 交互式循环 ---
    plotter = pv.Plotter(window_size=[1000, 800]); plotter.camera_position = 'xy'

    while True:
        try:
            input_pressure_str = input(f"\n请输入气压 P 值 (输入 'quit' 退出): ")
            if input_pressure_str.lower() == 'quit': break
            input_pressure = float(input_pressure_str)

            # --- 1. 预测位移 ---
            displacements_matrix = predict_displacements_for_pressure(model, scaler_X, scaler_y, device, input_pressure)
            if displacements_matrix is None: continue

            # --- 2. 计算参考坐标系下的变形坐标 ---
            deformed_curve1_ref_unordered = initial_coords_ref + displacements_matrix
            curve2_ref = initial_coords_ref + width_translation_vector
            deformed_curve2_ref_unordered = curve2_ref + displacements_matrix

            # --- 3. 对变形后的曲线进行空间排序 ---
            print("正在对变形后的节点进行空间排序...")
            sorted_deformed_curve1_ref = sort_points_spatially(deformed_curve1_ref_unordered)
            sorted_deformed_curve2_ref = sort_points_spatially(deformed_curve2_ref_unordered)
            print("排序完成。")

            # --- 4. 合并排序后的顶点并创建参考网格 ---
            sorted_deformed_vertices_ref = np.vstack((sorted_deformed_curve1_ref, sorted_deformed_curve2_ref))
            # 使用排序后的顶点和标准的面片连接创建网格
            deformed_mesh_ref = pv.PolyData(sorted_deformed_vertices_ref, faces=faces_np)
            deformed_mesh_ref.clean(inplace=True)

            # --- 5. 对每个手指应用变换并可视化 ---
            plotter.clear_actors()
            plotter.add_mesh(tray, color=tray_color, opacity=0.5, name='tray')
            plotter.add_mesh(sphere_point_cloud_actor, color=sphere_color, point_size=5, render_points_as_spheres=True, name='sphere')

            angles_deg = [0, 120, 240]; angles_rad = [np.radians(deg) for deg in angles_deg]
            for i, angle_rad in enumerate(angles_rad):
                # a) 计算 T3: Local -> World (放置变换)
                rot_angle_z_placing = angle_rad + np.pi / 2.0
                rot_z_placing = create_rotation_matrix_z(rot_angle_z_placing)
                target_pos_on_circle = np.array([ finger_placement_radius * np.cos(angle_rad), finger_placement_radius * np.sin(angle_rad), 0.0 ])
                T3_place = create_transformation_matrix(rot_z_placing, target_pos_on_circle)

                # b) 组合最终变换 T_final = T3 @ T2 @ T1
                T_final = T3_place @ T2_rotate @ T1_translate

                # c) 应用最终变换
                final_transformed_mesh = deformed_mesh_ref.transform(T_final, inplace=False)

                # d) 添加到绘图器
                plotter.add_mesh(final_transformed_mesh, color=finger_color, style='surface',
                                 edge_color='grey', opacity=0.85, smooth_shading=True,
                                 name=f'finger_{i}', label=f'变形手指 {i+1} (P={input_pressure})')

            # --- 添加其他可视化元素 ---
            if show_axes: plotter.add_axes_at_origin(labels_off=False)
            plotter.add_legend(bcolor=None, border=False)
            plotter.enable_anti_aliasing()
            plotter.add_text(f"当前气压 P = {input_pressure}", position="upper_edge", font_size=12)
            print("正在更新显示场景...")
            if not plotter.iren.initialized: plotter.show()
            else: plotter.render()

        except ValueError: print("无效输入，请输入一个数字或 'quit'。")
        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            import traceback; traceback.print_exc()

    print("程序退出。")
    # plotter.close()