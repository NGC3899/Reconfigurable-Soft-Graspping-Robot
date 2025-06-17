import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D # 通常在新版本 matplotlib 中不再需要显式导入
import sys

# --- 1. 用户需要确认/修改的参数 ---

# Excel 文件路径 (包含预测和仿真位移数据)
EXCEL_FILE_PATH = 'comparison.xlsx' # 假设文件在脚本同一目录下

# 包含数据的 Sheet 名称 (请根据你创建 Excel 文件时设置的名称修改)
# 在上一个脚本中，我们设置的是 "Displacement Comparison"
SHEET_NAME = 'Displacement Comparison'

# 定义列名 (请确保这些名称与 comparison.xlsx 中的表头完全一致)
COL_X0 = 'X0'           # 初始 X 坐标列
COL_Y0 = 'Y0'           # 初始 Y 坐标列
COL_Z0 = 'Z0'           # 初始 Z 坐标列
COL_U1_PRED = 'Predicted_U1' # 预测 U1 列 (F 列)
COL_U2_PRED = 'Predicted_U2' # 预测 U2 列 (G 列)
COL_U3_PRED = 'Predicted_U3' # 预测 U3 列 (H 列)
COL_U1_SIM = 'Simulated_U1'  # 仿真 U1 列 (假设用户添加到了 I 列，并命名为此)
COL_U2_SIM = 'Simulated_U2'  # 仿真 U2 列 (假设用户添加到了 J 列，并命名为此)
COL_U3_SIM = 'Simulated_U3'  # 仿真 U3 列 (假设用户添加到了 K 列，并命名为此)

# 可选：指定 NodeLabel 列名，用于错误检查和映射（如果 Excel 中有的话）
COL_NODE_LABEL = 'NodeLabel' # B 列

# --- 2. 加载和验证数据 ---

def load_comparison_data(file_path, sheet_name):
    """从 Excel 文件加载和验证对比数据"""
    print(f"正在读取 Excel 文件: '{file_path}', Sheet: '{sheet_name}'...")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        print(f"成功读取 Sheet '{sheet_name}'.")
        print("\n找到的列名:")
        print(df.columns.tolist())
    except FileNotFoundError:
        print(f"错误：找不到 Excel 文件 '{file_path}'")
        sys.exit()
    except ValueError as ve:
        if "Worksheet named" in str(ve) and sheet_name in str(ve):
            print(f"错误：在 Excel 文件中找不到名为 '{sheet_name}' 的 Sheet。请检查 Sheet 名称。")
        else:
            print(f"读取 Excel 时发生值错误: {ve}")
        sys.exit()
    except Exception as e:
        print(f"错误：读取 Excel 文件或 Sheet '{sheet_name}' 时出错: {e}")
        sys.exit()

    # 检查必需的列是否存在
    required_coords = [COL_X0, COL_Y0, COL_Z0]
    required_pred = [COL_U1_PRED, COL_U2_PRED, COL_U3_PRED]
    required_sim = [COL_U1_SIM, COL_U2_SIM, COL_U3_SIM]
    # NodeLabel 是可选的，但如果存在则使用
    optional_cols = [COL_NODE_LABEL] if COL_NODE_LABEL in df.columns else []

    all_required_cols = required_coords + required_pred + required_sim
    missing_cols = [col for col in all_required_cols if col not in df.columns]

    if missing_cols:
        print(f"\n错误：Excel Sheet '{sheet_name}' 中缺少以下必需的列: {missing_cols}")
        print("请确保 comparison.xlsx 文件包含正确的表头，并且仿真数据已添加到 I, J, K 列并正确命名。")
        sys.exit()

    # 检查数据是否有效（例如，是否都是数字，是否有 NaN）
    cols_to_check_numeric = all_required_cols
    nan_counts = df[cols_to_check_numeric].isnull().sum()
    non_numeric_issues = False

    for col in cols_to_check_numeric:
        # 尝试转换为数值类型，非数值会变成 NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().sum() > nan_counts[col]: # 如果转换后 NaN 增多，说明有非数值
            non_numeric_issues = True
            print(f"警告：列 '{col}' 中包含非数值数据，这些行将被忽略或导致错误。")

    # 检查仿真列是否有 NaN (用户可能未完全填充)
    nan_in_sim = df[required_sim].isnull().any(axis=1)
    if nan_in_sim.any():
        print(f"\n警告：仿真位移列 ({required_sim}) 中发现 {nan_in_sim.sum()} 行包含缺失值 (NaN)。")
        print("这些不完整的行将从绘图中排除。请确保所有节点的仿真数据都已填充。")
        df = df.dropna(subset=required_sim) # 删除仿真数据不完整的行

    if df.empty:
        print("\n错误：经过数据验证和清理后，没有有效数据可供绘图。")
        sys.exit()

    print(f"\n数据加载和验证完成。共有 {len(df)} 个有效节点的数据。")
    return df

# --- 3. 空间排序函数 (基于初始坐标) ---
# (改编自用户提供的 2.txt 文件中的逻辑 [cite: 6, 7, 8, 9, 10, 11])
def get_spatially_sorted_indices(df_nodes, x0_col, y0_col, z0_col, node_label_col=None):
    """
    使用最近邻算法根据初始坐标对节点进行空间排序。
    返回排序后的 DataFrame 行索引列表。
    """
    num_nodes = len(df_nodes)
    if num_nodes < 2:
        print("节点数少于 2，无需空间排序。")
        return df_nodes.index.tolist() # 返回原始索引顺序

    print(f"\n正在根据初始坐标对 {num_nodes} 个节点进行空间排序...")

    # 提取用于排序的坐标和原始索引
    coords = df_nodes[[x0_col, y0_col, z0_col]].values
    original_indices = df_nodes.index.tolist() # 获取 DataFrame 的行索引

    # 创建索引到坐标的映射
    index_coords_map = {idx: coords[i] for i, idx in enumerate(original_indices)}

    # 待访问的索引集合
    remaining_indices = set(original_indices)

    # 选择起始点 (简单选择第一个)
    current_index = original_indices[0]
    sorted_indices = [current_index]
    remaining_indices.remove(current_index)

    # 执行最近邻算法
    while remaining_indices:
        last_coord = index_coords_map[current_index]
        min_dist_sq = float('inf')
        nearest_index = -1 # 使用 DataFrame 索引，可以是任何类型

        for index in remaining_indices:
            coord = index_coords_map[index]
            dist_sq = np.sum((last_coord - coord)**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_index = index

        if nearest_index != -1: # 检查是否找到最近邻
            sorted_indices.append(nearest_index)
            remaining_indices.remove(nearest_index)
            current_index = nearest_index
        else:
            if remaining_indices: # 仍然有剩余点但找不到，可能数据有问题
                 print(f"警告：空间排序中无法找到下一个最近邻。剩余 {len(remaining_indices)} 个索引。排序可能不完整。")
            break # 退出循环

    if len(sorted_indices) != num_nodes:
        print(f"警告：排序后的索引数量 ({len(sorted_indices)}) 与原始节点数 ({num_nodes}) 不符！")

    print(f"空间排序完成。路径包含 {len(sorted_indices)} 个节点。")
    return sorted_indices # 返回排序后的 DataFrame 原始索引列表

# --- 4. 计算变形坐标 ---
def calculate_deformed_coords(df, x0, y0, z0, u1, u2, u3):
    """计算变形后的坐标"""
    x_def = df[x0] + df[u1]
    y_def = df[y0] + df[u2]
    z_def = df[z0] + df[u3]
    return x_def.values, y_def.values, z_def.values # 返回 NumPy 数组

# --- 5. 绘图函数 ---
def plot_comparison(x_pred, y_pred, z_pred, x_sim, y_sim, z_sim, x0, y0, z0):
    """绘制预测和仿真变形曲线的 3D 对比图"""
    if len(x_pred) < 2 or len(x_sim) < 2:
        print("错误：需要至少两个节点才能绘制曲线。")
        return

    print("\n正在生成 3D 对比图...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制预测曲线 (蓝色)
    ax.plot(x_pred, y_pred, z_pred, marker='.', markersize=4, linestyle='-', color='blue', label='Predicted Deformation')

    # 绘制仿真曲线 (红色)
    ax.plot(x_sim, y_sim, z_sim, marker='x', markersize=4, linestyle='--', color='red', label='Simulated Deformation')

    # (可选) 绘制初始形状 (灰色虚线)
    # ax.plot(x0, y0, z0, marker='', linestyle=':', color='gray', label='Initial Shape (Sorted)')

    # 设置坐标轴标签和标题
    ax.set_xlabel('Global X')
    ax.set_ylabel('Global Y')
    ax.set_zlabel('Global Z')
    ax.set_title('Predicted vs. Simulated Deformation Comparison')

    # 尝试设置坐标轴范围 (类似 2.txt [cite: 13, 14, 15, 19, 20])
    try:
        all_x = np.concatenate((x_pred, x_sim, x0))
        all_y = np.concatenate((y_pred, y_sim, y0))
        all_z = np.concatenate((z_pred, z_sim, z0))
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        z_min, z_max = all_z.min(), all_z.max()

        max_range = max(x_max-x_min, y_max-y_min, z_max-z_min) if len(all_x) > 0 else 1.0
        if max_range < 1e-6: max_range = 1.0 # 避免范围过小
        margin = max_range * 0.1

        mid_x = (x_max+x_min) * 0.5
        mid_y = (y_max+y_min) * 0.5
        mid_z = (z_max+z_min) * 0.5
        ax.set_xlim(mid_x - max_range/2 - margin, mid_x + max_range/2 + margin)
        ax.set_ylim(mid_y - max_range/2 - margin, mid_y + max_range/2 + margin)
        ax.set_zlim(mid_z - max_range/2 - margin, mid_z + max_range/2 + margin)
        print("已调整坐标轴范围。")
    except Exception as e_axis:
        print(f"注意：无法自动调整坐标轴范围 (错误: {e_axis})。使用默认范围。")

    ax.legend()
    plt.grid(True)
    plt.show()
    print("图形已显示。")


# --- 6. 主程序 ---
if __name__ == '__main__':
    # 加载数据
    df_data = load_comparison_data(EXCEL_FILE_PATH, SHEET_NAME)

    # 获取空间排序后的索引
    sorted_indices = get_spatially_sorted_indices(df_data, COL_X0, COL_Y0, COL_Z0)

    # 根据排序后的索引重新排列 DataFrame (重要!)
    df_sorted = df_data.loc[sorted_indices]

    # 计算变形坐标 (在排序后的数据上)
    print("\n正在计算变形坐标...")
    try:
        x_pred, y_pred, z_pred = calculate_deformed_coords(df_sorted, COL_X0, COL_Y0, COL_Z0,
                                                       COL_U1_PRED, COL_U2_PRED, COL_U3_PRED)

        x_sim, y_sim, z_sim = calculate_deformed_coords(df_sorted, COL_X0, COL_Y0, COL_Z0,
                                                    COL_U1_SIM, COL_U2_SIM, COL_U3_SIM)

        # 提取排序后的初始坐标用于绘图或对比
        x0_sorted = df_sorted[COL_X0].values
        y0_sorted = df_sorted[COL_Y0].values
        z0_sorted = df_sorted[COL_Z0].values
        print("变形坐标计算完成。")

        # 绘制对比图
        plot_comparison(x_pred, y_pred, z_pred, x_sim, y_sim, z_sim, x0_sorted, y0_sorted, z0_sorted)

    except KeyError as ke:
         print(f"\n错误：计算变形坐标时发生列名错误: {ke}。请检查脚本中的列名定义是否与 Excel 文件完全一致。")
    except Exception as e_main:
         print(f"\n处理过程中发生意外错误: {e_main}")

    print("\n脚本执行完毕。")