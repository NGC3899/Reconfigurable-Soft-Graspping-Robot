# -*- coding: utf-8 -*-
import pandas as pd
import sys
import numpy as np
import pyvista as pv # 仍然需要用于手指网格和可视化

# --- 用户需要确认/修改的参数 ---

# 1. Excel 文件路径
excel_file_path = r'C:\Users\admin\Desktop\FEM\Displacement_Results.xlsx' # <<< 确认路径

# 2. Sheet 名称
sheet_name = 'Nodal Displacements' # <<< 确认 Sheet 名

# 3. 节点列表文件路径
node_list_file_path = r'C:\Users\admin\Desktop\num_list.txt' # <<< 确认路径

# 4. 要绘制的压力迭代次数
iteration_to_plot = 1 #

# 5. Y 轴平移距离 (用于手指的第二条曲线)
y_translation = 10.0 #

# 6. 对象点云参数 (仍使用球体生成作为示例)
sphere_radius = 5.0 # 球半径 (用于生成点云)
num_sphere_points = 200 # 球表面点数 (可调整)
object_point_color = 'blue' # 对象点云颜色 <<< 改为蓝色以便区分坐标轴
colliding_point_color = 'magenta' # 接触点颜色

# 7. 接触检测阈值 (点云点到指面距离)
collision_threshold = 0.5 # <<< 调整此值定义接触敏感度

# 8. 重叠/交叉检测阈值 (极小距离)
overlap_threshold = 1e-6 # 用于检测点是否几乎重合

# 9. 球体(对象)的初始位姿 (4x4 齐次变换矩阵) - 用于生成初始点云位置
theta_z = np.radians(45)
cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)
rotation_matrix_z = np.array([
    [cos_z, -sin_z, 0],
    [sin_z,  cos_z, 0],
    [    0,      0, 1]
])
# <<< 调整这个向量可以改变对象点云的全局位置 >>>
translation_vector = np.array([15, 10, 20]) # <<< 调整球体/对象初始中心位置

# 构建 4x4 变换矩阵
sphere_transform_matrix = np.identity(4)
sphere_transform_matrix[:3, :3] = rotation_matrix_z # 应用旋转
sphere_transform_matrix[:3, 3] = translation_vector # 应用平移

# 10. 可视化坐标轴的长度
axis_length = sphere_radius * 1.5 # 例如，设为半径的1.5倍

# --- 参数修改结束 ---

# --- 辅助函数 ---

def generate_sphere_points(radius, num_points):
    """使用斐波那契格点在球面上生成近似均匀的点云"""
    points = []
    phi = np.pi * (3. - np.sqrt(5.)) # 黄金角
    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        points.append((x, y, z))
    points_arr = np.array(points) * radius
    return points_arr

def transform_points(points, matrix):
    """将 4x4 齐次变换矩阵应用于 N x 3 的点集"""
    if points.shape[1] != 3:
        raise ValueError("输入点应为 N x 3 形状")
    if matrix.shape != (4, 4):
        raise ValueError("变换矩阵应为 4 x 4 形状")
    num_pts = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_pts, 1))))
    transformed_homogeneous = homogeneous_points @ matrix.T
    transformed_points = transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3, np.newaxis]
    return transformed_points

# --- 主代码逻辑 ---

# 读取节点列表 (代码同前)
node_labels_from_file = []
try:
    with open(node_list_file_path, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped:
                try:
                    node_labels_from_file.append(int(line_stripped))
                except ValueError: pass
except Exception as e:
    print(f"读取节点列表文件时出错: {e}")
    sys.exit()

# 读取 Excel 数据 (代码同前)
try:
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name, engine='openpyxl')
except Exception as e:
    print(f"读取 Excel 文件时出错: {e}")
    sys.exit()

# 检查列是否存在 (代码同前)
required_columns = ['Iteration', 'Pressure', 'NodeLabel', 'X0', 'Y0', 'Z0', 'U1', 'U2', 'U3']
if not all(col in df.columns for col in required_columns):
    print("错误：Excel 文件中缺少必需的列。")
    sys.exit()

# 筛选指定迭代次数的数据 (代码同前)
df_iter = df[df['Iteration'] == iteration_to_plot]
if df_iter.empty:
    print(f"错误：找不到迭代次数 {iteration_to_plot} 的数据。")
    sys.exit()

# 筛选并排序节点 (空间排序) (代码同前)
df_iter_indexed = df_iter.set_index('NodeLabel')
available_nodes_in_iter = set(df_iter_indexed.index)
nodes_to_process = [label for label in node_labels_from_file if label in available_nodes_in_iter]
if not nodes_to_process:
     print("错误：来自列表的节点在迭代数据中均未找到。")
     sys.exit()
df_relevant_nodes = df_iter_indexed.loc[nodes_to_process].reset_index()

# --- 空间排序逻辑 (代码同前) ---
spatially_sorted_labels = []
if len(df_relevant_nodes) > 1:
    coords_df = df_relevant_nodes[['NodeLabel', 'X0', 'Y0', 'Z0']].copy()
    node_coords_map = {row['NodeLabel']: np.array([row['X0'], row['Y0'], row['Z0']])
                       for _, row in coords_df.iterrows()}
    remaining_labels = set(coords_df['NodeLabel'])
    start_label = nodes_to_process[0]
    current_label = start_label
    spatially_sorted_labels.append(current_label)
    remaining_labels.remove(current_label)
    while remaining_labels:
        last_coord = node_coords_map[current_label]
        min_dist_sq = float('inf')
        nearest_label = -1
        for label in remaining_labels:
            dist_sq = np.sum((last_coord - node_coords_map[label])**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_label = label
        if nearest_label != -1:
            spatially_sorted_labels.append(nearest_label)
            remaining_labels.remove(nearest_label)
            current_label = nearest_label
        else:
            break
    df_filtered_sorted = df_relevant_nodes.set_index('NodeLabel').loc[spatially_sorted_labels].reset_index()
else:
    df_filtered_sorted = df_relevant_nodes

if df_filtered_sorted.empty:
    print("错误：筛选和排序后没有剩余数据。")
    sys.exit()
# --- 结束空间排序 ---

# 计算手指变形后的坐标 (代码同前)
try:
    df_filtered_sorted['X_def'] = df_filtered_sorted['X0'] + df_filtered_sorted['U1']
    df_filtered_sorted['Y_def'] = df_filtered_sorted['Y0'] + df_filtered_sorted['U2']
    df_filtered_sorted['Z_def'] = df_filtered_sorted['Z0'] + df_filtered_sorted['U3']
except Exception as e_calc:
     print(f"计算变形坐标时出错: {e_calc}")
     sys.exit()

# 提取手指曲线坐标 (代码同前)
curve1_points = df_filtered_sorted[['X_def', 'Y_def', 'Z_def']].values
if len(curve1_points) < 2:
    print("错误：需要至少两个点来定义手指表面。")
    sys.exit()
curve2_points = curve1_points.copy()
curve2_points[:, 1] += y_translation

# --- 创建手指表面网格 (使用 PyVista) ---
print("正在创建手指表面网格...")
num_points_per_curve = len(curve1_points)
finger_vertices = np.vstack((curve1_points, curve2_points))
finger_faces = []
for i in range(num_points_per_curve - 1):
    p1_idx, p2_idx = i, i + 1
    p3_idx, p4_idx = i + 1 + num_points_per_curve, i + num_points_per_curve
    finger_faces.append([4, p1_idx, p2_idx, p3_idx, p4_idx])

if finger_faces:
    faces_np = np.hstack(finger_faces)
else:
    faces_np = np.array([], dtype=int)

finger_mesh = pv.PolyData(finger_vertices, faces=faces_np)
finger_mesh.clean(inplace=True)
print(f"手指网格创建完成，包含 {finger_mesh.n_points} 个顶点和 {finger_mesh.n_cells} 个面片。")
print(f"手指网格位于全局坐标系。") # 明确手指位置

# --- 生成对象点云并进行处理 ---
print("正在生成对象点云...")
# 生成局部坐标下的点云 (仍以球体为例)
object_points_local = generate_sphere_points(sphere_radius, num_sphere_points)
print(f"本地坐标系生成了 {object_points_local.shape[0]} 个对象点。")

# 应用初始位姿变换得到全局坐标
print("正在将对象点云变换到全局坐标...")
object_points_global = transform_points(object_points_local, sphere_transform_matrix)
print(f"对象点已变换到全局坐标。")

# 计算点云质心 (在全局坐标系下)
centroid = np.mean(object_points_global, axis=0)
print(f"计算得到对象点云质心 (全局坐标): {centroid}")

# !!! 不再进行中心化 !!!
# object_points_centered = object_points_global - centroid
# print(f"对象点云已中心化，以质心为原点。")
# 现在直接使用 object_points_global

# --- 接触检测 (全局点云点 到 手指网格表面距离) ---
print("正在执行接触检测 (点到面，使用全局坐标)...")
contact_point_indices = [] # 存储接触点的索引 (在 object_points_global 中)
contact_distances = []     # 存储接触点的具体距离
min_distance_overall = float('inf') # 记录所有点中最小的距离
closest_point_index = -1 # 记录最近点的索引 (在对象点云中的索引)

overlap_detected = False # 标记是否检测到重叠

num_object_points = object_points_global.shape[0] # 使用全局点云
for i in range(num_object_points):
    point = object_points_global[i] # <<< 使用全局坐标点

    # 查找手指网格上离该点最近的点的 *索引*
    try:
        closest_point_idx_on_mesh = finger_mesh.find_closest_point(point)

        if closest_point_idx_on_mesh < 0 or closest_point_idx_on_mesh >= finger_mesh.n_points:
             print(f"警告：在点 {i} 查找到无效的最近点索引 ({closest_point_idx_on_mesh})。跳过此点。")
             continue

        closest_mesh_point_coords = finger_mesh.points[closest_point_idx_on_mesh]
        distance = np.linalg.norm(point - closest_mesh_point_coords)

    except TypeError as e_find_closest_type:
         print(f"警告：调用 find_closest_point 时发生 TypeError: {e_find_closest_type}。跳过点 {i}。")
         continue
    except Exception as e_find_closest:
        print(f"警告：在点 {i} 查找最近点或计算距离时出错: {e_find_closest}。跳过此点。")
        continue

    # 检查重叠/交叉
    if distance < overlap_threshold:
        print(f"错误：检测到交叉/重叠！对象点 {i} (全局坐标: {point}) 与手指表面距离过近 ({distance:.2e})。")
        overlap_detected = True
        # break

    # 检查接触
    if distance < collision_threshold:
        contact_point_indices.append(i)
        contact_distances.append(distance)

    # 更新全局最近点信息
    if distance < min_distance_overall:
        min_distance_overall = distance
        closest_point_index = i

print("接触检测完成。")

if overlap_detected:
    print("警告：检测到至少一处对象点云与手指表面交叉/重叠。")

if contact_point_indices:
    print(f"找到 {len(contact_point_indices)} 个接触点 (距离 < {collision_threshold})。")
    if contact_distances:
        min_contact_dist_index = np.argmin(contact_distances)
        closest_contact_point_cloud_index = contact_point_indices[min_contact_dist_index]
        # 使用全局坐标报告
        closest_contact_point_coords = object_points_global[closest_contact_point_cloud_index]
        print(f"距离手指表面最近的接触点索引(在点云中): {closest_contact_point_cloud_index}, 全局坐标: {closest_contact_point_coords}, 距离: {contact_distances[min_contact_dist_index]:.4f}")
    else:
        print("接触点列表为空，无法确定最近接触点。")

else:
    print("未找到接触点。")
    if closest_point_index != -1:
         # 使用全局坐标报告
         print(f"点云中离手指表面最近的点索引为 {closest_point_index}, 全局坐标为 {object_points_global[closest_point_index]}, 距离为 {min_distance_overall:.4f} (大于阈值)。")


# --- 使用 PyVista 进行可视化 ---
print("正在绘制网格和点云...")

try:
    plotter = pv.Plotter(window_size=[1000, 800])
except Exception as e_plotter:
    print(f"初始化 PyVista Plotter 时出错: {e_plotter}")
    sys.exit()

# 添加手指网格 (位置固定)
try:
    plotter.add_mesh(finger_mesh, color='lightblue', style='surface', smooth_shading=True, label='Finger Surface')
    plotter.add_mesh(finger_mesh, color='black', style='wireframe', line_width=1)
except Exception as e_add_finger:
    print(f"添加手指网格到绘图器时出错: {e_add_finger}")

# 添加对象点云 (使用全局坐标)
try:
    # 分离非接触点和接触点
    non_contact_mask = np.ones(num_object_points, dtype=bool)
    valid_indices = []
    if contact_point_indices:
        valid_indices = [idx for idx in contact_point_indices if idx < num_object_points]
        if valid_indices:
             non_contact_mask[valid_indices] = False

    non_contact_points = object_points_global[non_contact_mask] # <<< 使用全局坐标
    contact_points_coords = object_points_global[valid_indices] if valid_indices else np.empty((0, 3)) # <<< 使用全局坐标

    # 绘制非接触点
    if non_contact_points.shape[0] > 0:
        plotter.add_points(non_contact_points, color=object_point_color, point_size=5, render_points_as_spheres=True, label='Object Points (Non-Contact)')
    # 绘制接触点
    if contact_points_coords.shape[0] > 0:
        plotter.add_points(contact_points_coords, color=colliding_point_color, point_size=8, render_points_as_spheres=True, label=f'Object Points (Contact, Dist < {collision_threshold})')

except Exception as e_add_points:
    print(f"添加对象点云到绘图器时出错: {e_add_points}")

# 可视化质心和坐标轴
try:
    # 添加质心点
    plotter.add_points(centroid, color='green', point_size=15, render_points_as_spheres=True, label='Object Centroid')

    # 创建并添加坐标轴线段
    # X轴 (红色)
    x_axis_end = centroid + [axis_length, 0, 0]
    x_axis = pv.Line(centroid, x_axis_end)
    plotter.add_mesh(x_axis, color='red', line_width=5, label='Object X-axis')
    # Y轴 (绿色)
    y_axis_end = centroid + [0, axis_length, 0]
    y_axis = pv.Line(centroid, y_axis_end)
    plotter.add_mesh(y_axis, color='green', line_width=5, label='Object Y-axis')
    # Z轴 (蓝色)
    z_axis_end = centroid + [0, 0, axis_length]
    z_axis = pv.Line(centroid, z_axis_end)
    plotter.add_mesh(z_axis, color='blue', line_width=5, label='Object Z-axis')

except Exception as e_add_centroid_axes:
    print(f"添加质心或坐标轴时出错: {e_add_centroid_axes}")


# 添加标题和图例
try:
    plotter.add_text(f"手指-对象点云接触检测 (迭代: {iteration_to_plot})", position="upper_edge", font_size=12)
    plotter.add_legend()
    plotter.add_axes() # 添加全局坐标轴指示器
    # plotter.enable_zoom_scaling() # 已移除
except Exception as e_add_extras:
     print(f"添加文本/图例/坐标轴时出错: {e_add_extras}")

# 显示绘图窗口
print("正在显示交互式 3D 绘图...")
try:
    plotter.show()
except Exception as e_show:
    print(f"显示绘图窗口时出错: {e_show}")

print("绘图窗口已关闭或显示失败。")