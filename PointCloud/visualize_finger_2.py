# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
# 较新版本的 matplotlib 可能不需要下面这行来启用 3D 绘图
# from mpl_toolkits.mplot3d import Axes3D
import sys
import numpy as np # 用于计算坐标轴范围 和 距离

# --- 用户需要确认/修改的参数 ---

# 1. Abaqus 脚本生成的 Excel 文件路径 (包含位移数据)
# excel_file_path = r'C:\Users\admin\Desktop\FEM\Displacement_Results.xlsx' # <<< 确认这个路径和文件名正确
excel_file_path = r'D:\FEM\Displacement_Results.xlsx'
# 2. 包含节点位移数据的 Sheet 名称
sheet_name = 'Nodal Displacements' # <<< 请修改为你的 Excel 中实际的 Sheet 名称

# 3. 包含节点标签列表的文本文件路径 (顺序可能不符合绘图要求)
node_list_file_path = r'C:\Users\admin\Desktop\txt file\num_list.txt' # <<< 你提供的文件名

# 4. 要绘制的压力迭代次数
iteration_to_plot = 1 # <<< 你指定绘制第一次迭代

# 5. Y 轴平移距离
y_translation = 10.0 # <<< 指定 Y 轴平移距离

# --- 参数修改结束 ---

# 读取节点标签列表 (从 num_list.txt)
print("Reading node list file: '{}'...".format(node_list_file_path))
node_labels_from_file = []
try:
    with open(node_list_file_path, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped: # 避免空行
                try:
                    node_label = int(line_stripped)
                    node_labels_from_file.append(node_label)
                except ValueError:
                    print("警告：文件 '{}' 中找到非整数行: '{}'，已忽略。".format(node_list_file_path, line_stripped)) # [cite: 2]
    if not node_labels_from_file:
        print("错误：未能从文件 '{}' 中读取到任何有效的节点标签。".format(node_list_file_path)) # [cite: 3]
        sys.exit()
    print("Read {} node labels from file.".format(len(node_labels_from_file))) # [cite: 3]
    print("First 5 labels (from file): {}".format(node_labels_from_file[:5])) # [cite: 3]
    print("Last 5 labels (from file): {}".format(node_labels_from_file[-5:])) # [cite: 3]

except FileNotFoundError:
    print("错误：找不到节点列表文件 '{}'".format(node_list_file_path)) # [cite: 3]
    sys.exit()
except Exception as e:
    print("错误：读取节点列表文件 '{}' 时出错: {}".format(node_list_file_path, e)) # [cite: 3]
    sys.exit()


print("\nReading Excel file: '{}', Sheet: '{}'".format(excel_file_path, sheet_name)) # [cite: 4]
try:
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name, engine='openpyxl') # [cite: 4]
    print("Excel sheet '{}' read successfully.".format(sheet_name)) # [cite: 4]
    print("\nColumns found in Excel sheet:") # [cite: 4]
    print(df.columns.tolist()) # [cite: 4]
except FileNotFoundError:
    print("错误：找不到 Excel 文件 '{}'".format(excel_file_path)) # [cite: 4]
    sys.exit()
except ValueError as ve:
    if "Worksheet named" in str(ve) and sheet_name in str(ve):
         print("错误：在 Excel 文件中找不到名为 '{}' 的 Sheet。请检查 Sheet 名称。".format(sheet_name)) # [cite: 4]
    else:
         print("读取 Excel 时发生值错误: {}".format(ve)) # [cite: 4]
    sys.exit()
except Exception as e:
    print("错误：读取 Excel 文件或 Sheet '{}' 时出错: {}".format(sheet_name, e)) # [cite: 5]
    sys.exit()

# 检查必需的列是否存在
required_columns = ['Iteration', 'Pressure', 'NodeLabel', 'X0', 'Y0', 'Z0', 'U1', 'U2', 'U3'] # [cite: 5]
missing_columns = [col for col in required_columns if col not in df.columns] # [cite: 5]
if missing_columns:
    print("\n错误：Excel Sheet '{}' 中缺少以下必需的列: {}".format(sheet_name, missing_columns)) # [cite: 5]
    sys.exit()

print("\nFiltering data for iteration {} and specified nodes...".format(iteration_to_plot)) # [cite: 5]
# 筛选特定迭代的数据
df_iter = df[df['Iteration'] == iteration_to_plot] # [cite: 5]

if df_iter.empty:
    print("\n错误：在 Excel 文件中找不到迭代次数 {} 的数据。".format(iteration_to_plot)) # [cite: 5]
    available_iterations = sorted(df['Iteration'].unique()) # [cite: 5]
    print("文件中可用的迭代次数: {}".format(available_iterations)) # [cite: 5]
    sys.exit()

# --- 从迭代数据中筛选出 num_list.txt 中存在的节点 ---
df_iter_indexed = df_iter.set_index('NodeLabel') # [cite: 5]
available_nodes_in_iter = set(df_iter_indexed.index) # [cite: 5]

# 找出在文件中且在当前迭代数据中都存在的节点标签
nodes_to_process = [label for label in node_labels_from_file if label in available_nodes_in_iter] # [cite: 6]

# 报告文件中但数据中缺失的节点
missing_nodes_in_data = [label for label in node_labels_from_file if label not in available_nodes_in_iter] # [cite: 6]
if missing_nodes_in_data:
    print("\n警告：来自 num_list.txt 的以下节点标签在迭代 {} 的数据中找不到: {}".format(iteration_to_plot, missing_nodes_in_data)) # [cite: 6]

if not nodes_to_process:
     print("\n错误：提供的节点标签列表中，没有任何一个标签存在于迭代 {} 的数据中。无法绘图。".format(iteration_to_plot)) # [cite: 6]
     sys.exit()

# 获取这些存在节点的数据 (此时顺序可能还是 num_list.txt 的顺序，或被 .loc 打乱)
df_relevant_nodes = df_iter_indexed.loc[nodes_to_process].reset_index() # [cite: 6]
print("Found data for {} nodes specified in the list.".format(len(df_relevant_nodes))) # [cite: 6]

# --- 新增：空间排序逻辑 (基于初始坐标 X0, Y0, Z0) ---
print("\nAttempting to spatially sort {} nodes based on initial coordinates (X0, Y0, Z0)...".format(len(df_relevant_nodes))) # [cite: 6]

if len(df_relevant_nodes) > 1:
    # 提取节点标签和初始坐标
    coords_df = df_relevant_nodes[['NodeLabel', 'X0', 'Y0', 'Z0']].copy() # [cite: 6]
    # 创建一个字典，方便通过 NodeLabel 查找坐标
    node_coords_map = {row['NodeLabel']: np.array([row['X0'], row['Y0'], row['Z0']])
                       for index, row in coords_df.iterrows()} # [cite: 7]

    # 待访问的节点标签集合
    remaining_labels = set(coords_df['NodeLabel']) # [cite: 7]

    # 选择一个起始节点 (可以用 num_list.txt 中的第一个有效节点)
    start_label = nodes_to_process[0] # [cite: 7]
    current_label = start_label # [cite: 7]

    # 存储排序后的节点标签列表
    spatially_sorted_labels = [current_label] # [cite: 7]
    remaining_labels.remove(current_label) # [cite: 7]

    # 执行最近邻算法
    while remaining_labels: # [cite: 7]
        last_coord = node_coords_map[current_label] # [cite: 8]
        min_dist_sq = float('inf') # [cite: 8]
        nearest_label = -1 # 初始化为无效值 # [cite: 8]

        # 遍历所有剩余未访问的节点
        for label in remaining_labels: # [cite: 8]
            coord = node_coords_map[label] # [cite: 8]
            # 计算当前节点与目标节点的平方欧氏距离 (避免开方，只比较大小)
            dist_sq = np.sum((last_coord - coord)**2) # [cite: 8]

            # 如果找到更近的节点
            if dist_sq < min_dist_sq: # [cite: 9]
                min_dist_sq = dist_sq # [cite: 9]
                nearest_label = label # [cite: 9]

        # 如果成功找到最近的节点
        if nearest_label != -1: # [cite: 9]
            spatially_sorted_labels.append(nearest_label) # [cite: 9]
            remaining_labels.remove(nearest_label) # [cite: 10]
            current_label = nearest_label # 更新当前节点为新找到的最近节点 # [cite: 10]
        else:
            # 如果 remaining_labels 不为空但找不到最近节点，说明出错了
            print("警告：在空间排序中无法找到下一个最近的节点。剩余 {} 个节点。排序可能不完整。".format(len(remaining_labels))) # [cite: 10]
            break # 退出循环 # [cite: 10]

    print("Spatial sorting complete. Total nodes in sorted path: {}".format(len(spatially_sorted_labels))) # [cite: 10]
    if len(spatially_sorted_labels) < len(nodes_to_process):
         print("警告：排序后的节点数少于开始时的节点数！") # [cite: 11]

    # 使用排序后的标签列表来重新排列 DataFrame
    # 设置 NodeLabel 为索引，然后用 .loc 按列表顺序选取，最后重置索引
    df_filtered_sorted = df_relevant_nodes.set_index('NodeLabel').loc[spatially_sorted_labels].reset_index() # [cite: 11]

    print("First 5 nodes in spatial order: {}".format(spatially_sorted_labels[:5])) # [cite: 11]
    print("Last 5 nodes in spatial order: {}".format(spatially_sorted_labels[-5:])) # [cite: 11]

else:
    # 如果只有一个节点或没有节点，无需排序
    df_filtered_sorted = df_relevant_nodes # [cite: 11]
    print("Less than 2 nodes found, no spatial sorting needed.") # [cite: 11]

# --- 排序结束 ---


# 确保筛选和排序后仍有数据
if df_filtered_sorted.empty:
     print("\n错误：按节点标签列表筛选并排序后没有数据剩余。")
     sys.exit() # [cite: 12]

print("\nCalculating deformed coordinates for spatially sorted nodes...") # [cite: 12]
# 计算变形后的坐标 (在排序后的 DataFrame 上操作)
try:
    # 使用 .loc 避免 SettingWithCopyWarning
    df_filtered_sorted.loc[:, 'X_def'] = df_filtered_sorted['X0'] + df_filtered_sorted['U1'] # [cite: 12]
    df_filtered_sorted.loc[:, 'Y_def'] = df_filtered_sorted['Y0'] + df_filtered_sorted['U2'] # [cite: 12]
    df_filtered_sorted.loc[:, 'Z_def'] = df_filtered_sorted['Z0'] + df_filtered_sorted['U3'] # [cite: 12]
except KeyError as ke:
    print("\n错误：计算变形坐标时缺少列: {}。请检查 Excel 文件。".format(ke)) # [cite: 12]
    sys.exit()
except Exception as e_calc:
     print("\n计算变形坐标时出错: {}".format(e_calc)) # [cite: 12]
     sys.exit()

# 提取坐标用于绘图 (现在这些坐标是按照空间顺序排列的)
x_coords = df_filtered_sorted['X_def'].values # [cite: 12]
y_coords = df_filtered_sorted['Y_def'].values # [cite: 12]
z_coords = df_filtered_sorted['Z_def'].values # [cite: 12]

# (可选) 提取初始位置坐标用于对比 (也按照空间顺序)
x0_coords = df_filtered_sorted['X0'].values # [cite: 12]
y0_coords = df_filtered_sorted['Y0'].values # [cite: 12]
z0_coords = df_filtered_sorted['Z0'].values # [cite: 12]

# --- 新增：创建平移后的曲线坐标 ---
print("\nCreating translated curve (Y + {})...".format(y_translation))
if len(x_coords) > 0:
    x_coords_t = x_coords.copy() # 复制 X 坐标
    y_coords_t = y_coords + y_translation # Y 坐标平移
    z_coords_t = z_coords.copy() # 复制 Z 坐标
    print("Translated coordinates generated.")
else:
    print("错误：没有有效的变形坐标可用于创建平移曲线。")
    sys.exit()
# --- 新增结束 ---

print("Plotting original and translated curves with connecting lines...")
# 创建 3D 图
fig = plt.figure(figsize=(12, 9)) # 稍微调大图像尺寸
ax = fig.add_subplot(111, projection='3d') # [cite: 13]

# --- 修改：绘制两条曲线和连接线 ---
if len(x_coords) > 0:
    # 1. 绘制原始 3D 曲线
    ax.plot(x_coords, y_coords, z_coords, marker='.', markersize=5, linestyle='-',color='blue', label='Original Deformed Edge (Iter {})'.format(iteration_to_plot)) # [cite: 13]

    # 2. 绘制 Y 轴平移后的 3D 曲线
    ax.plot(x_coords_t, y_coords_t, z_coords_t, marker='.', markersize=5, linestyle='-', color='blue', label='Translated Edge (Y + {})'.format(y_translation))

    # 3. 绘制连接起点的线段
    ax.plot([x_coords[0], x_coords_t[0]], [y_coords[0], y_coords_t[0]], [z_coords[0], z_coords_t[0]],
            linestyle='-', color='blue', label='Start Connector')

    # 4. 绘制连接终点的线段
    ax.plot([x_coords[-1], x_coords_t[-1]], [y_coords[-1], y_coords_t[-1]], [z_coords[-1], z_coords_t[-1]],
            linestyle='-', color='blue', label='End Connector')

# (可选) 绘制初始位置作为对比 (如果需要)
# ax.plot(x0_coords, y0_coords, z0_coords, marker='.', markersize=3, linestyle='--', color='gray', label='Initial Edge (Spatially Sorted)') # [cite: 13]

# --- 修改结束 ---

# 设置坐标轴标签和标题
ax.set_xlabel('Global X') # [cite: 13]
ax.set_ylabel('Global Y') # [cite: 13]
ax.set_zlabel('Global Z') # [cite: 13]
# 获取该迭代的压力值用于标题
pressure_value = df_filtered_sorted['Pressure'].iloc[0] if not df_filtered_sorted.empty else 'N/A' # [cite: 13]
title_str = 'Soft Finger Contact Surface Outline\nIteration: {}, Pressure: {:.0f} Pa'.format(
              iteration_to_plot, pressure_value) # [cite: 13]
ax.set_title(title_str) # [cite: 13]

# 尝试设置坐标轴比例接近相等（可能需要调整）
try:
    # 计算各轴范围 (基于原始、平移和初始点)
    all_x = np.concatenate((x_coords, x_coords_t, x0_coords)) # [cite: 13]
    all_y = np.concatenate((y_coords, y_coords_t, y0_coords)) # [cite: 13]
    all_z = np.concatenate((z_coords, z_coords_t, z0_coords)) # [cite: 13]
    x_min, x_max = all_x.min(), all_x.max() # [cite: 14]
    y_min, y_max = all_y.min(), all_y.max() # [cite: 14]
    z_min, z_max = all_z.min(), all_z.max() # [cite: 14]

    # 找到最大范围，并以此设置所有轴，中心对齐
    max_range = max(x_max-x_min, y_max-y_min, z_max-z_min) if len(all_x) > 0 else 1.0 # [cite: 14]
    # 添加一点边距
    if max_range < 1e-6: # 避免范围过小 # [cite: 14]
        max_range = 1.0 # [cite: 14]
    margin = max_range * 0.1 # [cite: 14]
    mid_x = (x_max+x_min) * 0.5 # [cite: 14]
    mid_y = (y_max+y_min) * 0.5 # [cite: 14]
    mid_z = (z_max+z_min) * 0.5 # [cite: 14]
    ax.set_xlim(mid_x - max_range/2 - margin, mid_x + max_range/2 + margin) # [cite: 15]
    ax.set_ylim(mid_y - max_range/2 - margin, mid_y + max_range/2 + margin) # [cite: 15]
    ax.set_zlim(mid_z - max_range/2 - margin, mid_z + max_range/2 + margin) # [cite: 15]
    print("Axis limits adjusted for better aspect ratio.") # [cite: 15]
except Exception as e_axis:
    print("Note: Could not automatically adjust axis limits (Error: {}). Using default limits.".format(e_axis)) # [cite: 15]

ax.legend() # [cite: 16]
plt.grid(True) # 添加网格线
plt.show() # [cite: 16]

print("Plot displayed.") # [cite: 16]