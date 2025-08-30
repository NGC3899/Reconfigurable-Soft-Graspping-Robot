import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
# from mpl_toolkits.mplot3d import Axes3D # 如果需要3D绘图，请取消此行注释

# --- 1. 与训练时完全相同的模型定义 ---
class MLPRegression(nn.Module):
    """一个简单的多层感知机用于回归"""
    def __init__(self, input_dim, output_dim, h1, h2, h3):
        super(MLPRegression, self).__init__()
        # !!! 确保这里的层结构和激活函数与训练时完全一致 !!!
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, h3),
            nn.Tanh(),
            nn.Linear(h3, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# --- 2. 定义与训练时相同的参数 ---
INPUT_DIM = 1
NODE_COUNT = 63 # 节点数量
OUTPUT_DIM = NODE_COUNT * 3 # 189

# !!! 重要：这里的隐藏层大小必须与您训练并保存 mlp_model_v2.pth 时使用的完全一致 !!!
HIDDEN_LAYER_1 = 128
HIDDEN_LAYER_2 = 256
HIDDEN_LAYER_3 = 128

# --- 3. 定义文件路径 ---
MODEL_SAVE_DIR = r'D:\FEM_MLP_Model'
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'mlp_model_v2.pth')
X_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, 'x_scaler_v2.joblib')
Y_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, 'y_scaler_v2.joblib')

# !!! 确保这个初始坐标文件路径是正确的 !!!
INITIAL_COORDS_PATH = r'initial_coordinates.txt' # 假设此文件在脚本运行的同级目录下

# --- 4. 加载模型和 Scaler 的函数 ---
def load_prediction_components(model_path, x_scaler_path, y_scaler_path, input_dim, output_dim, h1, h2, h3):
    """加载模型和标准化器"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用的设备: {device}")

    for path in [model_path, x_scaler_path, y_scaler_path]:
        if not os.path.exists(path):
            print(f"错误：必需文件未找到: {path}")
            return None, None, None, None

    model = MLPRegression(input_dim, output_dim, h1, h2, h3)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # 设置为评估模式
        print(f"模型已成功从 {model_path} 加载。")
    except Exception as e:
        print(f"加载模型 {model_path} 时出错: {e}")
        return None, None, None, None
    
    try:
        scaler_X = joblib.load(x_scaler_path)
        scaler_y = joblib.load(y_scaler_path)
        print(f"Scaler 已成功从 {x_scaler_path} 和 {y_scaler_path} 加载。")
    except Exception as e:
        print(f"加载 Scaler 时出错: {e}")
        return None, None, None, None
        
    return model, scaler_X, scaler_y, device

# --- 5. 进行预测的函数 ---
def predict_displacements(model, scaler_X, scaler_y, device, pressure_value):
    """使用加载的模型和 Scaler 对给定的压力值进行预测"""
    if model is None: return None
    input_p = np.array([[pressure_value]], dtype=np.float32)
    input_p_scaled = scaler_X.transform(input_p)
    input_tensor = torch.tensor(input_p_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        predicted_scaled_tensor = model(input_tensor)
    predicted_scaled = predicted_scaled_tensor.cpu().numpy()
    predicted_original_scale = scaler_y.inverse_transform(predicted_scaled)
    return predicted_original_scale.flatten()

# --- 6. 加载初始坐标的函数 ---
def load_initial_coordinates(file_path, expected_nodes):
    """从文本文件加载初始坐标 (X0, Y0, Z0)"""
    if not os.path.exists(file_path):
        print(f"错误：找不到初始坐标文件 {file_path}")
        return None
    try:
        coords = np.loadtxt(file_path, dtype=np.float32)
        if coords.shape == (expected_nodes, 3):
            print(f"成功从 {file_path} 加载 {coords.shape[0]} 个节点的初始坐标。")
            return coords
        else:
            print(f"错误：坐标文件 {file_path} 的形状 {coords.shape} 与预期 ({expected_nodes}, 3) 不符。")
            return None
    except Exception as e:
        print(f"加载初始坐标文件 {file_path} 时出错: {e}")
        return None

# --- 7. 新增：将预测结果保存到Excel的函数 ---
def save_predictions_to_excel(predicted_u, initial_coords, pressure, save_dir):
    """将所有节点的预测位移按顺序保存到新的Excel文件中。"""
    if predicted_u is None or initial_coords is None:
        print("数据不完整，无法保存到Excel。")
        return

    print("\n正在准备将预测结果写入Excel文件...")
    
    # 创建一个DataFrame来结构化数据
    data_to_save = []
    for i in range(NODE_COUNT):
        node_label = i + 1
        start_index = i * 3
        
        u1 = predicted_u[start_index]
        u2 = predicted_u[start_index + 1]
        u3 = predicted_u[start_index + 2]
        
        x0, y0, z0 = initial_coords[i]
        
        data_to_save.append({
            'Node Label': node_label,
            'Initial_X': x0,
            'Initial_Y': y0,
            'Initial_Z': z0,
            'Predicted_U1': u1,
            'Predicted_U2': u2,
            'Predicted_U3': u3
        })
        
    df = pd.DataFrame(data_to_save)
    
    # 创建一个基于压力值的、唯一的文件名
    # 将文件名中的点替换为下划线，以避免问题
    pressure_str = str(pressure).replace('.', '_')
    excel_filename = f"prediction_at_{pressure_str}_Pa.xlsx"
    excel_filepath = os.path.join(save_dir, excel_filename)
    
    try:
        df.to_excel(excel_filepath, index=False, engine='openpyxl')
        print(f"成功！预测结果已按顺序保存到: {excel_filepath}")
    except PermissionError:
        print(f"错误：无法保存文件 {excel_filepath}。请检查文件是否被其他程序打开或是否有写入权限。")
    except Exception as e:
        print(f"保存到Excel时发生错误: {e}")

# --- 8. 计算和绘制变形后曲线的函数 ---
def calculate_and_plot_deformation(initial_coords, predicted_u, pressure_value):
    """计算变形坐标并使用matplotlib绘制3D图形。"""
    if initial_coords is None or predicted_u is None:
        print("无法计算变形坐标，无法绘图。")
        return

    print("正在生成 3D 可视化图形...")
    displacements_reshaped = predicted_u.reshape(initial_coords.shape[0], 3)
    deformed_coords = initial_coords + displacements_reshaped
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x_def, y_def, z_def = deformed_coords[:, 0], deformed_coords[:, 1], deformed_coords[:, 2]
    
    ax.scatter(x_def, y_def, z_def, color='red', s=20, label='Predicted Nodes', alpha=0.8)
    ax.plot(x_def, y_def, z_def, color='blue', linestyle='-', label='Connectivity (by order)')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title(f'Predicted Finger Deformation at P = {pressure_value} kPa')
    ax.legend()
    plt.grid(True)
    
    # 设置坐标轴比例一致
    max_range = np.array([x_def.max()-x_def.min(), y_def.max()-y_def.min(), z_def.max()-z_def.min()]).max() / 2.0
    mid_x = (x_def.max()+x_def.min()) * 0.5
    mid_y = (y_def.max()+y_def.min()) * 0.5
    mid_z = (z_def.max()+z_def.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()

# --- 9. 主执行程序 ---
if __name__ == '__main__':
    # 加载所有必要的组件
    model, scaler_X, scaler_y, device = load_prediction_components(
        MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH,
        INPUT_DIM, OUTPUT_DIM, HIDDEN_LAYER_1, HIDDEN_LAYER_2, HIDDEN_LAYER_3
    )
    initial_coords = load_initial_coordinates(INITIAL_COORDS_PATH, NODE_COUNT)

    if model is None or initial_coords is None:
        print("\n初始化失败，无法加载模型或初始坐标。程序退出。")
        exit()

    # 主循环，用于输入压力值
    while True:
        try:
            input_pressure_str = input("\n请输入要预测的气压 P 值 (kPa), 或输入 'quit' 退出: ")
            if input_pressure_str.lower() in ['quit', 'exit', 'q']:
                break
            input_pressure = float(input_pressure_str)

            # 1. 进行整体预测
            predicted_u = predict_displacements(model, scaler_X, scaler_y, device, input_pressure)
            if predicted_u is None:
                print("预测失败，请重试。")
                continue
            
            print(f"\n--- 压力 P = {input_pressure} kPa 的预测已完成 ---")

            # 2. *** 新增功能：自动将结果保存到Excel ***
            save_predictions_to_excel(predicted_u, initial_coords, input_pressure, MODEL_SAVE_DIR)

            # 3. (可选) 可视化整体变形
            show_plot = input("\n是否要显示整体3D变形图? (y/n, 默认 n): ").lower()
            if show_plot == 'y':
                calculate_and_plot_deformation(initial_coords, predicted_u, input_pressure)
            
        except ValueError:
            print("无效输入，请输入一个数字压力值或 'quit'。")
        except Exception as e_main:
            print(f"主程序发生错误: {e_main}")

    print("\n程序已退出。")
