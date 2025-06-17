# -*- coding: utf-8 -*-
import pandas as pd
import sys
import numpy as np
import pyvista as pv
# 新增导入用于线性代数计算
import numpy.linalg as LA

# --- 用户需要确认/修改的参数 ---
# 1. Excel 文件路径
# excel_file_path = r'C:\Users\admin\Desktop\ML_Training_Data\Displacement_Results_1.xlsx' # <<< 确认路径
excel_file_path = r'D:\FEM\Displacement_Results.xlsx'
# 2. Sheet 名称
sheet_name = 'Nodal Displacements' # <<< 确认 Sheet 名
# 3. 节点列表文件路径
node_list_file_path = r'C:\Users\admin\Desktop\txt file\num_list.txt' # <<< 确认路径
# 4. 要绘制的压力迭代次数
iteration_to_plot = 1 #
# 5. Y 轴平移距离 (用于手指的第二条曲线)
y_translation = 10.0 #
# 6. 对象点云参数 (仍使用球体生成作为示例)
sphere_radius = 5.0 # 球半径 (用于生成点云)
num_sphere_points = 4000 #
object_point_color = 'blue' #
colliding_point_color = 'magenta' # 接触点颜色
intersection_point_color = 'yellow' # 交叉状态下的"接触"点颜色
# 7. 接触检测阈值 (点云点到指面距离)
collision_threshold = 0.5 # <<< 调整此值定义接触敏感度
# 8. 重叠/交叉检测阈值 (极小距离)
overlap_threshold = 1e-6 # 用于检测点是否几乎重合
# 9. 球体(对象)的初始位姿 (4x4 齐次变换矩阵) - 用于生成初始点云位置
theta_z = np.radians(45) #
cos_z, sin_z = np.cos(theta_z), np.sin(theta_z) #
rotation_matrix_z = np.array([[cos_z, -sin_z, 0], [sin_z,  cos_z, 0], [0, 0, 1]]) #
translation_vector = np.array([15, 15, 25.5]) # <<< 调整球体/对象初始中心位置
sphere_transform_matrix = np.identity(4) #
sphere_transform_matrix[:3, :3] = rotation_matrix_z #
sphere_transform_matrix[:3, 3] = translation_vector #
# 10. 可视化坐标轴的长度
axis_length = sphere_radius * 1.5 #
# 11. 可视化法线和切向量的长度 (也用于平面大小参考)
vector_viz_length = sphere_radius * 0.5 #
# 12. 计算特征值时的小阈值 (用于判断非零)
eigenvalue_threshold = 1e-6 #
# 13. 接触点的摩擦系数 (静摩擦系数) - 虽然不再可视化锥体，但计算可能仍需
friction_coefficient = 0.5 # <<< 输入摩擦系数 mu

# --- 参数修改结束 ---

# --- 辅助函数 ---
def generate_sphere_points(radius, num_points):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))
    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2
        radius_at_y = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        points.append((x, y, z))
    return np.array(points) * radius

def transform_points(points, matrix):
    if points.shape[1] != 3:
        raise ValueError("Input points should be N x 3 shape")
    if matrix.shape != (4, 4):
        raise ValueError("Transformation matrix should be 4 x 4 shape")
    num_pts = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_pts, 1))))
    transformed_homogeneous = homogeneous_points @ matrix.T
    return transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3, np.newaxis]

def get_orthogonal_vectors(normal_vector):
    n = np.asarray(normal_vector).astype(float)
    norm_n = LA.norm(n)
    if norm_n < 1e-9:
        raise ValueError("Normal vector cannot be zero vector.")
    n /= norm_n # 确保是单位向量

    if np.abs(n[0]) > 0.9:
        v_arbitrary = np.array([0.0, 1.0, 0.0])
    else:
        v_arbitrary = np.array([1.0, 0.0, 0.0])

    t1 = np.cross(n, v_arbitrary)
    norm_t1 = LA.norm(t1)
    if norm_t1 < 1e-9: # 如果 n 平行于 v_arbitrary
        v_arbitrary = np.array([0.0, 0.0, 1.0]) # 换一个 Z 轴尝试
        t1 = np.cross(n, v_arbitrary)
        norm_t1 = LA.norm(t1)
        if norm_t1 < 1e-9: # 如果 n 也平行于 Z 轴
             if np.abs(n[0]) > 0.9:
                 t1 = np.array([0.,1.,0.])
                 t2_temp = np.array([0.,0.,1.])
             elif np.abs(n[1]) > 0.9:
                 t1 = np.array([1.,0.,0.])
                 t2_temp = np.array([0.,0.,1.])
             else:
                 t1 = np.array([1.,0.,0.])
                 t2_temp = np.array([0.,1.,0.])
             t1 /= LA.norm(t1)
             t2 = t2_temp / LA.norm(t2_temp)
             # 确保 t2 与 n, t1 正交并满足右手系
             t2 = np.cross(n, t1)
             t2 /= LA.norm(t2) # 再次归一化以防万一
             return t1, t2

    t1 /= norm_t1
    t2_temp = np.cross(n, t1)
    norm_t2 = LA.norm(t2_temp)
    if norm_t2 < 1e-9:
        raise ValueError("Could not compute valid second tangent vector.")
    t2 = t2_temp / norm_t2
    return t1, t2

# --- 主代码逻辑 ---

# 读取和处理节点/Excel数据...
node_labels_from_file = []
try:
    with open(node_list_file_path, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped:
                try:
                    node_labels_from_file.append(int(line_stripped))
                except ValueError:
                    pass
except Exception as e:
    print(f"读取节点列表文件时出错: {e}")
    sys.exit()

try:
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name, engine='openpyxl')
except Exception as e:
    print(f"读取 Excel 文件时出错: {e}")
    sys.exit()

required_columns = ['Iteration', 'Pressure', 'NodeLabel', 'X0', 'Y0', 'Z0', 'U1', 'U2', 'U3']
if not all(col in df.columns for col in required_columns):
    print("错误：Excel 文件中缺少必需的列。")
    sys.exit()

df_iter = df[df['Iteration'] == iteration_to_plot]
if df_iter.empty:
    print(f"错误：找不到迭代次数 {iteration_to_plot} 的数据。")
    sys.exit()

df_iter_indexed = df_iter.set_index('NodeLabel')
available_nodes_in_iter = set(df_iter_indexed.index)
nodes_to_process = [label for label in node_labels_from_file if label in available_nodes_in_iter]
if not nodes_to_process:
    print("错误：来自列表的节点在迭代数据中均未找到。")
    sys.exit()

df_relevant_nodes = df_iter_indexed.loc[nodes_to_process].reset_index()
spatially_sorted_labels = []
if len(df_relevant_nodes) > 1:
    coords_df = df_relevant_nodes[['NodeLabel', 'X0', 'Y0', 'Z0']].copy()
    node_coords_map = {row['NodeLabel']: np.array([row['X0'], row['Y0'], row['Z0']]) for _, row in coords_df.iterrows()}
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

try:
    df_filtered_sorted['X_def'] = df_filtered_sorted['X0'] + df_filtered_sorted['U1']
    df_filtered_sorted['Y_def'] = df_filtered_sorted['Y0'] + df_filtered_sorted['U2']
    df_filtered_sorted['Z_def'] = df_filtered_sorted['Z0'] + df_filtered_sorted['U3']
except Exception as e_calc:
    print(f"计算变形坐标时出错: {e_calc}")
    sys.exit()

curve1_points = df_filtered_sorted[['X_def', 'Y_def', 'Z_def']].values
if len(curve1_points) < 2:
    print("错误：需要至少两个点来定义手指表面。")
    sys.exit()

curve2_points = curve1_points.copy()
curve2_points[:, 1] += y_translation

# --- 创建手指表面网格并计算法线 ---
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
try:
    print("计算单元法线...")
    finger_mesh.compute_normals(point_normals=False, cell_normals=True, inplace=True, auto_orient_normals=True, flip_normals=False)
    if 'Normals' not in finger_mesh.cell_data:
        raise RuntimeError("无法计算手指网格的单元法线。")
    using_cell_normals = True
    print("成功计算单元法线。")
except Exception as e_normals:
    print(f"计算手指网格单元法线时出错: {e_normals}")
    sys.exit()

print(f"手指网格创建完成并计算单元法线，包含 {finger_mesh.n_points} 个顶点和 {finger_mesh.n_cells} 个面片。")

# --- 生成对象点云并进行处理 ---
print("正在生成对象点云...")
object_points_local = generate_sphere_points(sphere_radius, num_sphere_points)
print(f"本地坐标系生成了 {object_points_local.shape[0]} 个对象点。")
print("正在将对象点云变换到全局坐标...")
object_points_global = transform_points(object_points_local, sphere_transform_matrix)
print(f"对象点已变换到全局坐标。")
centroid = np.mean(object_points_global, axis=0)
print(f"计算得到对象点云质心 (全局坐标): {centroid}")

# --- 接触检测 ---
print("正在执行接触检测 (点到面，使用单元法线判断内外侧)...")
contact_points_data = []
min_distance_overall = float('inf')
closest_point_index_overall = -1
overlap_detected = False
found_positive_side = False
found_negative_side = False
num_object_points = object_points_global.shape[0]

for i in range(num_object_points):
    point = object_points_global[i]
    try:
        closest_cell_id, closest_point_coords_on_mesh = finger_mesh.find_closest_cell(point, return_closest_point=True)
        if closest_cell_id < 0:
            continue
        normal = finger_mesh.cell_normals[closest_cell_id].copy()
        distance = LA.norm(point - closest_point_coords_on_mesh)
        vector_to_point = point - closest_point_coords_on_mesh

        if LA.norm(vector_to_point) > 1e-9:
            side_sign = np.sign(np.dot(vector_to_point, normal))
        else:
            side_sign = 0 # 点重合或极近

    except IndexError:
        print(f"警告：访问单元法线时索引错误 (Cell ID: {closest_cell_id})。跳过点 {i}。")
        continue
    except Exception as e_find_closest:
        print(f"警告：在点 {i} 查找最近单元/法线或计算距离时出错: {e_find_closest}。跳过此点。")
        continue

    if distance < overlap_threshold:
        print(f"错误：检测到交叉/重叠！对象点 {i} (全局坐标: {point}) 与手指表面距离过近 ({distance:.2e})。")
        overlap_detected = True
    elif distance < collision_threshold:
        contact_points_data.append((distance, i, side_sign, closest_cell_id, closest_point_coords_on_mesh))
        if side_sign > 0:
            found_positive_side = True
        elif side_sign < 0:
            found_negative_side = True

    if distance < min_distance_overall:
        min_distance_overall = distance
        closest_point_index_overall = i

print("接触检测完成。")

# --- 判断最终状态 ---
final_state = "No Contact"
closest_contact_local_coords = None
contact_point_on_finger = None
normal_at_contact = None
tangent1_at_contact = None
tangent2_at_contact = None
grasp_matrix_G = None
grasp_isotropy_index = None
contact_indices_to_visualize = []

if overlap_detected:
    final_state = "Overlap Detected"
    print("最终状态：重叠/交叉 (Overlap Detected)")
    contact_indices_to_visualize = [data[1] for data in contact_points_data] if contact_points_data else []
elif found_positive_side and found_negative_side and contact_points_data:
    final_state = "Intersection"
    print("最终状态：交叉穿过 (Intersection)")
    contact_indices_to_visualize = [data[1] for data in contact_points_data]
elif contact_points_data: # 仅在一侧有接触点
    final_state = "Contact"
    print("最终状态：接触 (Contact)")
    contact_points_data.sort(key=lambda x: x[0])
    closest_distance, closest_contact_index, closest_side_sign, closest_contact_cell_id, contact_point_on_finger = contact_points_data[0]

    closest_contact_global_coords = object_points_global[closest_contact_index]
    closest_contact_local_coords = closest_contact_global_coords - centroid
    print(f"最近接触点(对象)索引: {closest_contact_index}, 局部坐标: {closest_contact_local_coords}, 距离: {closest_distance:.4f}, 相对侧: {'外侧' if closest_side_sign > 0 else '内侧' if closest_side_sign < 0 else '表面'}")
    print(f"接触点(手指表面)全局坐标: {contact_point_on_finger}")

    # === G 和 GII 计算部分（包含摩擦）===
    try:
        normal_at_contact = finger_mesh.cell_normals[closest_contact_cell_id].copy()
        norm_mag = LA.norm(normal_at_contact)

        if norm_mag > 1e-6:
             normal_at_contact /= norm_mag # 归一化
             print(f"接触点(手指表面)单位法向量 n: {normal_at_contact}")

             tangent1_at_contact, tangent2_at_contact = get_orthogonal_vectors(normal_at_contact)
             print(f"接触点(手指表面)单位切向量 t1: {tangent1_at_contact}")
             print(f"接触点(手指表面)单位切向量 t2: {tangent2_at_contact}")

             # ... [G 和 GII 计算代码保持不变] ...
             mu = friction_coefficient
             r = contact_point_on_finger - centroid
             n = normal_at_contact
             t1 = tangent1_at_contact
             t2 = tangent2_at_contact
             d1 = n + mu * t1
             d2 = n - mu * t1
             d3 = n + mu * t2
             d4 = n - mu * t2
             r_vec = np.asarray(r)
             d1_vec = np.asarray(d1)
             d2_vec = np.asarray(d2)
             d3_vec = np.asarray(d3)
             d4_vec = np.asarray(d4)
             wrench_d1 = np.concatenate((d1_vec, np.cross(r_vec, d1_vec)))
             wrench_d2 = np.concatenate((d2_vec, np.cross(r_vec, d2_vec)))
             wrench_d3 = np.concatenate((d3_vec, np.cross(r_vec, d3_vec)))
             wrench_d4 = np.concatenate((d4_vec, np.cross(r_vec, d4_vec)))
             grasp_matrix_G = np.column_stack((wrench_d1, wrench_d2, wrench_d3, wrench_d4))
             print(f"\n抓取矩阵 G (6x4, 基于摩擦系数 mu={mu}):")
             with np.printoptions(precision=4, suppress=True): print(grasp_matrix_G)
             J = grasp_matrix_G @ grasp_matrix_G.T
             try:
                 eigenvalues = LA.eigvalsh(J)
                 non_zero_eigenvalues = eigenvalues[eigenvalues > eigenvalue_threshold]
                 if len(non_zero_eigenvalues) > 0:
                     lambda_min = np.min(non_zero_eigenvalues)
                     lambda_max = np.max(non_zero_eigenvalues)
                     print(f"\n考虑摩擦(mu={mu})后:")
                     print(f"  非零最小特征值 (lambda_min): {lambda_min:.4f}")
                     print(f"  非零最大特征值 (lambda_max): {lambda_max:.4f}")
                     if lambda_max < eigenvalue_threshold:
                         print("  最大非零特征值过小，无法计算有意义的各向同性指数。")
                         grasp_isotropy_index = 0.0
                     elif lambda_min < 0:
                          print(f"  警告：计算出的最小非零特征值为负数 ({lambda_min:.2e})，无法计算各向同性指数。")
                          grasp_isotropy_index = None
                     else:
                          grasp_isotropy_index = np.sqrt(lambda_min / lambda_max)
                          print(f"  抓取各向同性指数 (sqrt(lambda_min/lambda_max)): {grasp_isotropy_index:.4f}")
                 else:
                     print("  未找到显著大于零的特征值，无法计算各向同性指数。")
                     grasp_isotropy_index = 0.0
             except LA.LinAlgError as e_eig:
                 print(f"  计算特征值时出错: {e_eig}")
                 grasp_isotropy_index = None
        else: # 法向量为零
             print("警告：接触单元的法向量接近零，无法计算 G 和各向同性指数。")
             normal_at_contact = None
    except IndexError:
         print(f"警告：访问单元法线时索引错误 (Cell ID: {closest_contact_cell_id})，无法获取接触点法向量。")
         normal_at_contact = None
    except Exception as e_vec:
         print(f"计算法向量、切向量或G/GII时出错: {e_vec}")
         normal_at_contact = None
    # === 结束 G 和 GII 计算部分 ===

    contact_indices_to_visualize = [data[1] for data in contact_points_data]
else: # 无接触
    print("最终状态：无接触 (No Contact)")
    if closest_point_index_overall != -1:
        print(f"点云中离手指表面最近的点索引为 {closest_point_index_overall}, 全局坐标为 {object_points_global[closest_point_index_overall]}, 距离为 {min_distance_overall:.4f} (大于阈值)。")


# --- 使用 PyVista 进行可视化 ---
print("正在绘制网格和点云...")
try:
    plotter = pv.Plotter(window_size=[1000, 800])
except Exception as e_plotter:
    print(f"初始化 PyVista Plotter 时出错: {e_plotter}")
    sys.exit()

# 添加手指网格
try:
    plotter.add_mesh(finger_mesh, color='lightblue', style='surface', opacity=0.7, smooth_shading=True, label='Finger Surface')
except Exception as e_add_finger:
    print(f"添加手指网格到绘图器时出错: {e_add_finger}")

# 添加对象点云（分开绘制逻辑）
try:
    num_object_points_viz = object_points_global.shape[0]
    non_contact_mask = np.ones(num_object_points_viz, dtype=bool)
    highlight_points_coords = np.empty((0, 3))
    valid_indices = []
    if contact_indices_to_visualize:
        valid_indices = [idx for idx in contact_indices_to_visualize if idx < num_object_points_viz]
    if valid_indices:
        non_contact_mask[valid_indices] = False
        highlight_points_coords = object_points_global[valid_indices]

    non_contact_points_coords = object_points_global[non_contact_mask]
    if non_contact_points_coords.shape[0] > 0:
        plotter.add_points(non_contact_points_coords, color=object_point_color, point_size=5, render_points_as_spheres=True, label='Object Points (Non-Contact)')
    if highlight_points_coords.shape[0] > 0:
        highlight_color = 'gray'
        highlight_label = 'Highlight Points (Unknown State)'
        if final_state == "Intersection":
            highlight_color = intersection_point_color
            highlight_label = 'Object Points (Intersection)'
        elif final_state == "Contact":
            highlight_color = colliding_point_color
            highlight_label = f'Object Points (Contact, Dist < {collision_threshold})'
        elif final_state == "Overlap Detected":
            highlight_color = 'orange'
            highlight_label = 'Object Points (Near Overlap)'
        plotter.add_points(highlight_points_coords, color=highlight_color, point_size=8, render_points_as_spheres=True, label=highlight_label)
except Exception as e_add_points:
    print(f"添加对象点云到绘图器时出错: {e_add_points}")

# 可视化质心和坐标轴
try:
    plotter.add_points(centroid, color='green', point_size=15, render_points_as_spheres=True, label='Object Centroid')
    x_axis_end = centroid + [axis_length, 0, 0]
    x_axis = pv.Line(centroid, x_axis_end)
    plotter.add_mesh(x_axis, color='red', line_width=5, label='Object X-axis')
    y_axis_end = centroid + [0, axis_length, 0]
    y_axis = pv.Line(centroid, y_axis_end)
    plotter.add_mesh(y_axis, color='green', line_width=5, label='Object Y-axis')
    z_axis_end = centroid + [0, 0, axis_length]
    z_axis = pv.Line(centroid, z_axis_end)
    plotter.add_mesh(z_axis, color='blue', line_width=5, label='Object Z-axis')
except Exception as e_add_centroid_axes:
    print(f"添加质心或坐标轴时出错: {e_add_centroid_axes}")

# 可视化接触点法向量、切向量和切平面
if final_state == "Contact" and contact_point_on_finger is not None and normal_at_contact is not None:
    try:
        # 接触点
        plotter.add_points(contact_point_on_finger, color='black', point_size=12, render_points_as_spheres=True, label='Contact Point (Finger)')

        # 法向量
        plotter.add_arrows(cent=contact_point_on_finger, direction=normal_at_contact, mag=vector_viz_length, color='black', label='Normal Vector')

        # 切向量
        if tangent1_at_contact is not None:
            plotter.add_arrows(cent=contact_point_on_finger, direction=tangent1_at_contact, mag=vector_viz_length * 0.8, color='purple', label='Tangent 1')
        if tangent2_at_contact is not None:
            plotter.add_arrows(cent=contact_point_on_finger, direction=tangent2_at_contact, mag=vector_viz_length * 0.8, color='teal', label='Tangent 2')

        # --- 新增：可视化切平面 ---
        plane_size = vector_viz_length * 3 # 调整平面可视化大小
        tangent_plane = pv.Plane(center=contact_point_on_finger, # 平面中心在接触点
                                 direction=normal_at_contact, # 平面法向与接触点法向一致
                                 i_size=plane_size, j_size=plane_size, # 设置平面大小
                                 i_resolution=1, j_resolution=1) # 分辨率设为1即可

        plotter.add_mesh(tangent_plane, color='gray', opacity=0.5, # 半透明灰色
                         style='surface', label='Tangent Plane')
        # --- 结束：可视化切平面 ---

        # --- 移除摩擦锥可视化代码 ---
        # (相关代码已删除)
        # --- 结束移除 ---

    except Exception as e_add_contact_items: # 重命名了异常变量
        print(f"可视化接触点、向量或平面时出错: {e_add_contact_items}")


# 添加标题和图例
try:
    title_text = f"手指-对象点云状态: {final_state} (迭代: {iteration_to_plot})"
    # 移除了 GII 显示，因为摩擦锥相关的计算不再是主要焦点，如果需要可以加回来
    # if final_state == "Contact" and grasp_isotropy_index is not None:
    #     title_text += f"\nGrasp Isotropy Index (mu={friction_coefficient}): {grasp_isotropy_index:.4f}"
    plotter.add_text(title_text, position="upper_edge", font_size=10)
    plotter.add_legend()
    plotter.add_axes()
except Exception as e_add_extras:
    print(f"添加文本/图例/坐标轴时出错: {e_add_extras}")

# 显示绘图窗口
print("正在显示交互式 3D 绘图...")
try:
    plotter.show()
except Exception as e_show:
    print(f"显示绘图窗口时出错: {e_show}")

print("绘图窗口已关闭或显示失败。")