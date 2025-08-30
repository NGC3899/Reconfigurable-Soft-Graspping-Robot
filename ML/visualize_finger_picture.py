# -*- coding: utf-8 -*-
import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
import joblib
import sys
from scipy.spatial.distance import cdist

# --- 1. 用户可配置参数 ---

# --- 文件路径 (请确保这些文件存在) ---
MODEL_PATH = 'best_mlp_model.pth'
X_SCALER_PATH = 'x_scaler.joblib'
Y_SCALER_PATH = 'y_scaler.joblib'
INITIAL_COORDS_PATH = 'initial_coordinates.txt'

# --- 模型和几何参数 (与 Opt_10_BO.txt 保持一致) ---
INPUT_DIM = 1
NODE_COUNT = 63
OUTPUT_DIM = NODE_COUNT * 3
HIDDEN_LAYER_1 = 128
HIDDEN_LAYER_2 = 256
HIDDEN_LAYER_3 = 128
FINGER_WIDTH = 10.0 # 手指宽度

# --- 可视化参数 ---
PRESSURE_TO_VISUALIZE = 1000.0
MAX_PRESSURE = 40000.0

# --- 来自参考脚本的美化参数 ---
FINGER_COLOR_VIZ = '#ff7f0e'      # Matplotlib Orange, 严格匹配参考脚本
BACKGROUND_COLOR_VIZ = '#EAEAEA'  # 浅灰色背景
TEXT_COLOR_VIZ = 'black'          # 黑色文字
FONT_FAMILY_VIZ = 'times'         # Times New Roman 字体

# --- 2. 从 Opt_10_BO.txt 文件中借鉴的必要代码 ---

# --- ML 模型定义 ---
class MLPRegression(nn.Module):
    def __init__(self, input_dim, output_dim, h1, h2, h3):
        super(MLPRegression, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, h3), nn.Tanh(),
            nn.Linear(h3, output_dim) )
    def forward(self, x): return self.network(x)

# --- 辅助函数 ---
def load_prediction_components(model_path, x_scaler_path, y_scaler_path, input_dim, output_dim, h1, h2, h3):
    """加载模型和缩放器"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ML使用设备: {device}")
    model = MLPRegression(input_dim, output_dim, h1, h2, h3)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"模型 {model_path} 加载成功。")
    except Exception as e:
        print(f"加载模型 {model_path} 时出错: {e}")
        return None, None, None, None
    
    scaler_X, scaler_y = None, None
    try:
        scaler_X = joblib.load(x_scaler_path)
        print(f"X Scaler {x_scaler_path} 加载成功。")
    except FileNotFoundError:
        print(f"警告: 未找到 X scaler '{x_scaler_path}'。")
    except Exception as e:
        print(f"加载 X Scaler '{x_scaler_path}' 时出错: {e}")
        return None, None, None, None
        
    try:
        scaler_y = joblib.load(y_scaler_path)
        print(f"Y Scaler {y_scaler_path} 加载成功。")
    except FileNotFoundError:
        print(f"警告: 未找到 Y scaler '{y_scaler_path}'。")
    except Exception as e:
        print(f"加载 Y Scaler '{y_scaler_path}' 时出错: {e}")
        return None, None, None, None
        
    return model, scaler_X, scaler_y, device

def predict_displacements_for_pressure(model, scaler_X, scaler_y, device, pressure_value):
    """根据压力值预测位移"""
    if model is None:
        print("模型未加载，无法预测。")
        return None
        
    pressure_value = np.clip(pressure_value, 0.0, MAX_PRESSURE)
    input_p = np.array([[pressure_value]], dtype=np.float32)
    
    if scaler_X:
        try:
            input_p_scaled = scaler_X.transform(input_p)
        except Exception as e:
            print(f"X scaler 标准化出错: {e}")
            return None
    else:
        input_p_scaled = input_p
        
    input_tensor = torch.tensor(input_p_scaled, dtype=torch.float32).to(device)
    predicted_original_scale = None
    
    with torch.no_grad():
        try:
            predicted_scaled_tensor = model(input_tensor)
            predicted_scaled = predicted_scaled_tensor.cpu().numpy()
            
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

def load_initial_coordinates(file_path, expected_nodes):
    """加载初始坐标"""
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
    """创建面片数组"""
    faces = []
    num_quads = num_nodes_per_curve - 1
    if num_quads <= 0:
        return np.array([], dtype=int)
        
    for i in range(num_quads):
        p1, p2 = i, i + 1
        p3, p4 = (i + 1) + num_nodes_per_curve, i + num_nodes_per_curve
        faces.append([4, p1, p2, p3, p4])
        
    return np.hstack(faces)

def sort_points_spatially(points):
    """对点进行空间排序，以确保网格拓扑正确"""
    if points is None: return None
    points = np.asarray(points)
    if points.shape[0] < 2: return points
    
    num_points = points.shape[0]
    sorted_indices = []
    remaining_indices = list(range(num_points))
    
    start_node_index = np.argmin(points[:,0] + points[:,1] + points[:,2])
    current_index = start_node_index
    
    sorted_indices.append(current_index)
    if current_index in remaining_indices:
        remaining_indices.pop(remaining_indices.index(current_index))
        
    while remaining_indices:
        last_point = points[current_index, np.newaxis]
        remaining_points_array = points[remaining_indices]
        
        if remaining_points_array.ndim == 1:
            remaining_points_array = remaining_points_array[np.newaxis, :]
        if remaining_points_array.shape[0] == 0:
            break
            
        distances = cdist(last_point, remaining_points_array)[0]
        if distances.size == 0:
            break
            
        nearest_neighbor_relative_index = np.argmin(distances)
        nearest_neighbor_absolute_index = remaining_indices[nearest_neighbor_relative_index]
        
        sorted_indices.append(nearest_neighbor_absolute_index)
        current_index = nearest_neighbor_absolute_index
        
        if nearest_neighbor_absolute_index in remaining_indices:
            remaining_indices.pop(remaining_indices.index(nearest_neighbor_absolute_index))
            
    if len(sorted_indices) != num_points:
        print(f"警告: 空间排序只处理了 {len(sorted_indices)} / {num_points} 个点。")
        
    return points[sorted_indices]

def setup_publication_plotter(title, window_size=[1000, 800]):
    """
    创建一个具有统一出版级风格的 PyVista 绘图器。
    严格仿照参考脚本中的风格。
    """
    plotter_theme = pv.themes.DocumentTheme()
    plotter_theme.font.family = FONT_FAMILY_VIZ
    plotter_theme.font.color = pv.Color(TEXT_COLOR_VIZ)
    plotter_theme.background = pv.Color(BACKGROUND_COLOR_VIZ)

    plotter = pv.Plotter(window_size=window_size, theme=plotter_theme, title=title)
    
    plotter.enable_anti_aliasing('msaa', multi_samples=8)
    plotter.enable_parallel_projection()

    plotter.remove_all_lights()
    plotter.enable_lightkit()

    return plotter

# --- 3. 主执行逻辑 ---
if __name__ == '__main__':
    # --- 加载所需组件 ---
    model, scaler_X, scaler_y, device = load_prediction_components(
        MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH, INPUT_DIM, OUTPUT_DIM, HIDDEN_LAYER_1, HIDDEN_LAYER_2, HIDDEN_LAYER_3
    )
    initial_coords = load_initial_coordinates(INITIAL_COORDS_PATH, NODE_COUNT)

    if model is None or initial_coords is None:
        sys.exit("错误: 核心组件加载失败，程序终止。")

    # --- 生成变形后的手指几何体 ---
    print(f"\n正在为压力值 {PRESSURE_TO_VISUALIZE} 计算位移...")
    displacements = predict_displacements_for_pressure(model, scaler_X, scaler_y, device, PRESSURE_TO_VISUALIZE)
    
    if displacements is None:
        sys.exit("错误: 位移预测失败，程序终止。")
    print("位移计算成功。")

    # --- 创建网格 ---
    print("正在创建变形后的手指网格...")
    deformed_curve1_unordered = initial_coords + displacements
    
    width_vector = np.array([0, FINGER_WIDTH, 0])
    curve2_initial = initial_coords + width_vector
    deformed_curve2_unordered = curve2_initial + displacements

    deformed_curve1 = sort_points_spatially(deformed_curve1_unordered)
    deformed_curve2 = sort_points_spatially(deformed_curve2_unordered)

    if deformed_curve1 is None or deformed_curve2 is None:
        sys.exit("错误: 点云空间排序失败，无法创建网格。")

    finger_vertices = np.vstack((deformed_curve1, deformed_curve2))
    finger_faces = create_faces_array(NODE_COUNT)
    
    if finger_faces.size == 0:
        sys.exit("错误: 未能创建面片数组。")

    finger_mesh = pv.PolyData(finger_vertices, faces=finger_faces)
    finger_mesh.clean(inplace=True)
    print(f"原始网格创建完成，包含 {finger_mesh.n_points} 个顶点和 {finger_mesh.n_cells} 个面片。")

    # --- 高质量可视化处理 ---
    print("正在将网格三角化...")
    finger_mesh.triangulate(inplace=True)
    
    print("正在细分网格以提升平滑度...")
    finger_mesh_smooth = finger_mesh.subdivide(2, subfilter='loop')
    print(f"细分后，网格包含 {finger_mesh_smooth.n_points} 个顶点和 {finger_mesh_smooth.n_cells} 个面片。")

    # --- 使用新的函数设置绘图器并进行高质量渲染 ---
    print("正在绘制优化后的手指网格...")
    plotter = setup_publication_plotter(title="Deformed Finger Visualization")
    
    plotter.add_mesh(
        finger_mesh_smooth,
        color=FINGER_COLOR_VIZ,
        style='surface',
        smooth_shading=True,
        opacity=0.95,
        show_edges=True,
        edge_color='dimgray',
        line_width=0.5
    )
    
    '''
    # --- 【核心修改】添加并自定义坐标轴 ---
    plotter.show_bounds(
        grid='front',             # 只在前景显示网格线
        location='outer',         # 在外侧显示刻度
        ticks='inside',           # 刻度线朝内，更简洁
        xlabel='',                # 移除 X 轴标题
        ylabel='',                # 移除 Y 轴标题
        zlabel='',                # 移除 Z 轴标题
        n_xlabels=5,              # 控制 X 轴刻度数量
        n_ylabels=5,              # 控制 Y 轴刻度数量
        n_zlabels=5,              # 控制 Z 轴刻度数量
        font_family='times',      # 设置字体为新罗马
        color=TEXT_COLOR_VIZ      # 设置坐标轴和标签颜色
    )

    '''
    plotter.add_text(f"手指可视化 (压力: {PRESSURE_TO_VISUALIZE})", position="upper_edge", font=FONT_FAMILY_VIZ, color=TEXT_COLOR_VIZ)
    plotter.camera.zoom(1.2)

    # 显示绘图窗口
    print("\n正在显示交互式3D绘图... 您可以旋转和缩放。")
    plotter.show(cpos='xy')
    print("绘图窗口已关闭。")

