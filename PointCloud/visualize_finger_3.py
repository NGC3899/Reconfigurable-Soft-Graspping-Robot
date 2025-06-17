# -*- coding: utf-8 -*-
import pandas as pd
# import matplotlib.pyplot as plt # 3D 绘图不再需要
import sys
import numpy as np
import pyvista as pv # <<< 新增，用于网格操作和可视化

# --- 用户需要确认/修改的参数 ---

# 1. Excel 文件路径
# excel_file_path = r'C:\Users\admin\Desktop\FEM\Displacement_Results.xlsx' # <<< 确认路径
excel_file_path = r'D:\FEM\Displacement_Results.xlsx'
# 2. Sheet 名称
sheet_name = 'Nodal Displacements' # <<< 确认 Sheet 名

# 3. 节点列表文件路径
node_list_file_path = r'C:\Users\admin\Desktop\txt file\num_list.txt' # <<< 确认路径

# 4. 要绘制的压力迭代次数
iteration_to_plot = 1 #

# 5. Y 轴平移距离 (用于手指的第二条曲线)
y_translation = 10.0 #

# 6. 球体参数
sphere_radius = 5.0 # 球半径
sphere_center_initial = np.array([0, 0, 0]) # 球体变换前的初始中心
sphere_color = 'red' # 球体颜色
colliding_point_color = 'magenta' # 碰撞点/标记的颜色

# 7. 球体的位姿 (4x4 齐次变换矩阵)
#    示例：绕 Z 轴旋转 45 度，然后平移到 (15, 10, 20)
theta_z = np.radians(45)
cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)
rotation_matrix_z = np.array([
    [cos_z, -sin_z, 0],
    [sin_z,  cos_z, 0],
    [    0,      0, 1]
])
translation_vector = np.array([15, 5, 20]) # <<< 调整球体中心位置

# 构建 4x4 变换矩阵
sphere_transform_matrix = np.identity(4)
sphere_transform_matrix[:3, :3] = rotation_matrix_z # 应用旋转
sphere_transform_matrix[:3, 3] = translation_vector # 应用平移

# 8. PyVista 碰撞检测参数
contact_offset = 0.1 # 表面需要多近才算接触
# collision_method 和 generate_contacts 参数已被移除以兼容旧版本

# --- 参数修改结束 ---

# --- 辅助函数 (变换点 - 也可用于网格) ---

def transform_points(points, matrix):
    """将 4x4 齐次变换矩阵应用于 N x 3 的点集"""
    if points.shape[1] != 3:
        raise ValueError("输入点应为 N x 3 形状")
    if matrix.shape != (4, 4):
        raise ValueError("变换矩阵应为 4 x 4 形状")

    num_pts = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_pts, 1)))) # 转为齐次坐标
    transformed_homogeneous = homogeneous_points @ matrix.T # 应用变换
    # 除以 w 分量，转回笛卡尔坐标
    transformed_points = transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3, np.newaxis]
    return transformed_points

# --- 主代码逻辑 ---

# 读取节点列表
node_labels_from_file = []
try:
    with open(node_list_file_path, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped:
                try:
                    node_labels_from_file.append(int(line_stripped)) # 添加节点标签
                except ValueError: pass # 忽略非整数行
except Exception as e:
    print(f"读取节点列表文件时出错: {e}")
    sys.exit()

# 读取 Excel 数据
try:
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name, engine='openpyxl') # 使用 openpyxl 引擎
except Exception as e:
    print(f"读取 Excel 文件时出错: {e}")
    sys.exit()

# 检查列是否存在
required_columns = ['Iteration', 'Pressure', 'NodeLabel', 'X0', 'Y0', 'Z0', 'U1', 'U2', 'U3'] # 定义必需的列名
if not all(col in df.columns for col in required_columns):
    print("错误：Excel 文件中缺少必需的列。")
    sys.exit()

# 筛选指定迭代次数的数据
df_iter = df[df['Iteration'] == iteration_to_plot] # 根据迭代次数筛选数据
if df_iter.empty:
    print(f"错误：找不到迭代次数 {iteration_to_plot} 的数据。") # 如果筛选结果为空
    sys.exit()

# 筛选并排序节点 (空间排序)
df_iter_indexed = df_iter.set_index('NodeLabel') # 将 NodeLabel 设为索引
available_nodes_in_iter = set(df_iter_indexed.index) # 获取该迭代中所有可用的节点标签
nodes_to_process = [label for label in node_labels_from_file if label in available_nodes_in_iter] # 筛选出在文件列表和迭代数据中都存在的节点
if not nodes_to_process:
     print("错误：来自列表的节点在迭代数据中均未找到。")
     sys.exit()
df_relevant_nodes = df_iter_indexed.loc[nodes_to_process].reset_index() # 获取相关节点的数据

# --- 空间排序逻辑 (与之前相同) ---
spatially_sorted_labels = []
if len(df_relevant_nodes) > 1:
    coords_df = df_relevant_nodes[['NodeLabel', 'X0', 'Y0', 'Z0']].copy()
    node_coords_map = {row['NodeLabel']: np.array([row['X0'], row['Y0'], row['Z0']])
                       for _, row in coords_df.iterrows()} # 创建节点标签到坐标的映射
    remaining_labels = set(coords_df['NodeLabel']) # 待排序的节点标签集合
    start_label = nodes_to_process[0] # 从列表中的第一个节点开始
    current_label = start_label
    spatially_sorted_labels.append(current_label) # 添加到排序列表
    remaining_labels.remove(current_label) # 从待排序集合中移除

    while remaining_labels: # 当还有未排序节点时
        last_coord = node_coords_map[current_label] # 获取当前最后一个已排序节点的坐标
        min_dist_sq = float('inf') # 初始化最小距离平方为无穷大
        nearest_label = -1 # 初始化最近节点标签

        for label in remaining_labels: # 遍历剩余节点
            dist_sq = np.sum((last_coord - node_coords_map[label])**2) # 计算与当前节点距离的平方
            if dist_sq < min_dist_sq: # 如果找到更近的节点
                min_dist_sq = dist_sq
                nearest_label = label

        if nearest_label != -1: # 如果找到了最近的节点
            spatially_sorted_labels.append(nearest_label) # 添加到排序列表
            remaining_labels.remove(nearest_label) # 从待排序集合中移除
            current_label = nearest_label # 更新当前节点
        else:
            # 如果找不到下一个最近的节点（例如，图不连通），则中断排序
            break # 排序中断
    # 根据空间排序后的标签列表重新索引 DataFrame
    df_filtered_sorted = df_relevant_nodes.set_index('NodeLabel').loc[spatially_sorted_labels].reset_index() # 按排序后的标签获取数据
else:
    # 如果只有一个相关节点，无需排序
    df_filtered_sorted = df_relevant_nodes

if df_filtered_sorted.empty:
    print("错误：筛选和排序后没有剩余数据。")
    sys.exit()
# --- 结束空间排序 ---

# 计算变形后的坐标
try:
    # 将初始坐标 (X0, Y0, Z0) 与位移 (U1, U2, U3) 相加
    df_filtered_sorted['X_def'] = df_filtered_sorted['X0'] + df_filtered_sorted['U1'] # 计算 X 方向变形后坐标
    df_filtered_sorted['Y_def'] = df_filtered_sorted['Y0'] + df_filtered_sorted['U2'] # 计算 Y 方向变形后坐标
    df_filtered_sorted['Z_def'] = df_filtered_sorted['Z0'] + df_filtered_sorted['U3'] # 计算 Z 方向变形后坐标
except Exception as e_calc:
     print(f"计算变形坐标时出错: {e_calc}")
     sys.exit()

# 提取曲线坐标
curve1_points = df_filtered_sorted[['X_def', 'Y_def', 'Z_def']].values # 获取第一条曲线的点坐标
if len(curve1_points) < 2:
    print("错误：需要至少两个点来定义手指表面。")
    sys.exit()

# 创建平移后的曲线坐标
curve2_points = curve1_points.copy() # 复制第一条曲线的点
curve2_points[:, 1] += y_translation # 将 Y 坐标平移指定距离

# --- 使用 PyVista 创建网格 ---

print("正在创建手指表面网格...")
num_points_per_curve = len(curve1_points) # 每条曲线上的点数
# 将两条曲线的点合并成一个顶点数组
finger_vertices = np.vstack((curve1_points, curve2_points))

# 创建面片 (初始为四边形，PyVista 内部会处理或转换为三角形)
# 连接 curve1 上的点 i, i+1 和 curve2 上的点 i+1, i
finger_faces = []
for i in range(num_points_per_curve - 1):
    # 定义四边形顶点索引: (curve1_i, curve1_i+1, curve2_i+1, curve2_i)
    # 索引相对于 `finger_vertices` 数组
    p1_idx = i
    p2_idx = i + 1
    p3_idx = i + 1 + num_points_per_curve
    p4_idx = i + num_points_per_curve
    # PyVista 的面片格式：[顶点数, 顶点索引1, 顶点索引2, ...]
    finger_faces.append([4, p1_idx, p2_idx, p3_idx, p4_idx])

# 将所有面片数据连接成一个 NumPy 数组
if finger_faces:
    faces_np = np.hstack(finger_faces)
else:
    faces_np = np.array([], dtype=int) # 如果没有面片，则为空数组


# 创建手指表面的 PyVista PolyData 对象
finger_mesh = pv.PolyData(finger_vertices, faces=faces_np)
# 可选：清理网格 (移除未使用的点等)
finger_mesh.clean(inplace=True)
print(f"手指网格创建完成，包含 {finger_mesh.n_points} 个顶点和 {finger_mesh.n_cells} 个面片。")


print("正在创建并变换球体网格...")
# 使用 PyVista 在原点创建一个球体网格
sphere_mesh_local = pv.Sphere(radius=sphere_radius, center=sphere_center_initial, theta_resolution=30, phi_resolution=30)

# 对球体的顶点应用变换矩阵
sphere_points_local = sphere_mesh_local.points
sphere_points_global = transform_points(sphere_points_local, sphere_transform_matrix)

# 用变换后的顶点创建最终的球体网格 (拓扑结构/面片保持不变)
sphere_mesh_global = pv.PolyData(sphere_points_global, faces=sphere_mesh_local.faces)
print(f"球体网格已创建并变换。近似中心: {sphere_mesh_global.center}")


# --- 使用 PyVista 进行碰撞检测 ---
print("正在执行网格碰撞检测...")

# 移除了 method 和 generate_contacts 参数以兼容旧版本 PyVista
# 现在 collision 函数只返回两个值：是否碰撞 (布尔值) 和 交集信息 (可能是一个 PolyData 或 None)
try:
    collided, intersection_mesh = finger_mesh.collision(sphere_mesh_global, cell_tolerance=contact_offset) # <<< 最终修改在此行
except TypeError as e_coll:
    print(f"调用 collision 函数时发生 TypeError: {e_coll}")
    print("这可能仍然是由于 PyVista 版本不兼容。尝试最简单的调用：")
    try:
        # 尝试不带 cell_tolerance 的调用
        collided, intersection_mesh = finger_mesh.collision(sphere_mesh_global)
    except Exception as e_coll_basic:
        print(f"尝试最简单的 collision 调用也失败了: {e_coll_basic}")
        collided = False
        intersection_mesh = None
except Exception as e_generic:
    print(f"执行碰撞检测时发生未知错误: {e_generic}")
    collided = False
    intersection_mesh = None


contact_points = None
if collided and intersection_mesh is not None and isinstance(intersection_mesh, pv.PolyData) and intersection_mesh.n_points > 0:
    print(f"检测到手指和球体之间发生碰撞！")
    # 假设 intersection_mesh 包含了接触点
    contact_points = intersection_mesh.points
    print(f"从交集网格中提取 {len(contact_points)} 个接触点。")

elif collided:
     print(f"检测到手指和球体之间发生碰撞，但未生成明确的接触点信息 (交集网格为空或无效)。")
     print(f"Intersection_mesh type: {type(intersection_mesh)}")
     if intersection_mesh is not None and isinstance(intersection_mesh, pv.PolyData):
         print(f"Intersection_mesh n_points: {intersection_mesh.n_points}")


else:
    print("未检测到碰撞。")


# --- 使用 PyVista 进行可视化 ---
print("正在绘制网格...")

# 检查 PyVista 是否能够成功初始化绘图器
try:
    plotter = pv.Plotter(window_size=[1000, 800]) # 创建一个绘图器对象
except Exception as e_plotter:
    print(f"初始化 PyVista Plotter 时出错: {e_plotter}")
    print("无法进行可视化。请检查 PyVista 和 VTK 安装是否正确。")
    sys.exit()


# 添加手指网格
try:
    plotter.add_mesh(finger_mesh, color='lightblue', style='surface', smooth_shading=True, label='Finger Surface')
    plotter.add_mesh(finger_mesh, color='black', style='wireframe', line_width=1)
except Exception as e_add_finger:
    print(f"添加手指网格到绘图器时出错: {e_add_finger}")

# 添加球体网格
try:
    plotter.add_mesh(sphere_mesh_global, color=sphere_color, style='surface', opacity=0.7, smooth_shading=True, label='Sphere')
    plotter.add_mesh(sphere_mesh_global, color='gray', style='wireframe', line_width=1, opacity=0.5)
except Exception as e_add_sphere:
    print(f"添加球体网格到绘图器时出错: {e_add_sphere}")

# 添加接触点 (如果找到)
if contact_points is not None and len(contact_points) > 0:
    try:
        plotter.add_points(contact_points, color=colliding_point_color, point_size=10, render_points_as_spheres=True, label='Contact Points')
        print(f"正在可视化 {len(contact_points)} 个接触点。")
    except Exception as e_add_points:
        print(f"添加接触点到绘图器时出错: {e_add_points}")

# 添加标题和图例
try:
    plotter.add_text(f"手指-球体碰撞检测 (迭代次数: {iteration_to_plot})", position="upper_edge", font_size=12)
    plotter.add_legend() # 显示图例
    plotter.add_axes() # 显示坐标轴
    plotter.enable_zoom_scaling() # 允许缩放
except Exception as e_add_extras:
     print(f"添加文本/图例/坐标轴时出错: {e_add_extras}")

# 显示绘图窗口
print("正在显示交互式 3D 绘图...")
try:
    plotter.show()
except Exception as e_show:
    print(f"显示绘图窗口时出错: {e_show}")


print("绘图窗口已关闭或显示失败。")