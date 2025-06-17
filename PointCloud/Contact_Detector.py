# -*- coding: utf-8 -*-
import open3d as o3d
import numpy as np
import trimesh # 用于精确的网格处理和接触检测
import copy
import time # 用于计时和暂停观察
import matplotlib.pyplot as plt # 用于生成颜色

print(f"Open3D version: {o3d.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Trimesh version: {trimesh.__version__}")

# --- 1. 参数定义 ---

# 球体参数
sphere_radius = 0.5
sphere_center = np.array([2.0, 0.0, 0.0])
sphere_resolution = 20

# 连杆机器人参数 (3个旋转关节 - RRR)
link_lengths = [0.0, 0.8, 0.6] # L0, L1, L2
link_radius = 0.05
link_colors = [[0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.8]] # 红、绿、蓝

# 机器人基座相对于世界坐标系的齐次变换矩阵
T_world_base = np.array([
    [0.0, 1.0, 0.0, 1.5],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

# 接触检测参数
contact_distance_threshold = 10000

# *** 关节角度循环参数 ***
angle_step_deg = 10
theta1_range_deg = np.arange(-40, 40 + angle_step_deg, angle_step_deg)
theta2_range_deg = np.arange(0, 90 + angle_step_deg, angle_step_deg)
theta3_range_deg = np.arange(-60, 60 + angle_step_deg, angle_step_deg)

# *** 可视化更新频率 ***
visualization_update_skips = 20 # 调小一点，刷新更频繁

# 用于存储第一个找到的接触状态
first_contact_found = False
first_contact_angles = None
first_contact_point = None
# first_contact_robot_mesh_o3d = None # 不再需要单独存储接触时的网格

# --- Helper Functions --- (保持不变)
def create_transformation_matrix(theta, length):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    length_float = float(length)
    return np.array([
        [cos_t, -sin_t, 0.0, length_float],
        [sin_t,  cos_t, 0.0, 0.0],
        [0.0,    0.0,   1.0, 0.0],
        [0.0,    0.0,   0.0, 1.0]
    ], dtype=np.float64)

def create_link_base_mesh(length, radius):
    if length <= 1e-6: return None
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    R_z_to_x = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
    cylinder.rotate(R_z_to_x, center=(0, 0, 0))
    cylinder.translate((length / 2.0, 0, 0))
    cylinder.compute_vertex_normals()
    return cylinder

# --- 2. 生成球体网格 --- (保持不变)
print("生成球体网格...")
sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=sphere_resolution)
sphere_mesh.translate(sphere_center)
sphere_mesh.paint_uniform_color([0.1, 0.7, 0.1]) # Green
sphere_mesh.compute_vertex_normals()
try:
    tm_sphere = trimesh.Trimesh(vertices=np.asarray(sphere_mesh.vertices),
                                faces=np.asarray(sphere_mesh.triangles),
                                vertex_normals=np.asarray(sphere_mesh.vertex_normals),
                                process=False)
except Exception as e: print(f"错误: 球体转换为 Trimesh 失败 - {e}"); exit()
print("球体网格生成并转换为 Trimesh 对象完毕。")

# --- 3. 生成基础连杆网格 --- (保持不变)
print("生成基础连杆网格...")
link1_base_mesh = create_link_base_mesh(link_lengths[1], link_radius)
link2_base_mesh = create_link_base_mesh(link_lengths[2], link_radius)
link_meshes_exist = link1_base_mesh is not None or link2_base_mesh is not None
if not link_meshes_exist: print("错误：未能生成任何基础连杆网格。"); exit()
print("基础连杆网格生成完毕。")

# --- 3.5 设置初始姿态的机器人网格 (用于可视化起点) ---
print("\n计算初始姿态机器人网格...")
joint_angles_initial_deg = np.array([theta1_range_deg[0], theta2_range_deg[0], theta3_range_deg[0]])
joint_angles_initial_rad = np.radians(joint_angles_initial_deg)
T_0_1_init = create_transformation_matrix(joint_angles_initial_rad[0], link_lengths[0])
T_1_2_init = create_transformation_matrix(joint_angles_initial_rad[1], link_lengths[1])
T_0_j1_init = T_0_1_init
T_0_j2_init = T_0_j1_init @ T_1_2_init

initial_link_meshes_list = []
robot_mesh_for_vis = o3d.geometry.TriangleMesh() # *** 这个对象将在循环中被更新 ***

if link1_base_mesh:
    link1_initial = copy.deepcopy(link1_base_mesh)
    link1_initial.transform(T_0_j1_init)
    link1_initial.paint_uniform_color(link_colors[0])
    initial_link_meshes_list.append(link1_initial)
if link2_base_mesh:
    link2_initial = copy.deepcopy(link2_base_mesh)
    link2_initial.transform(T_0_j2_init)
    link2_initial.paint_uniform_color(link_colors[1])
    initial_link_meshes_list.append(link2_initial)

if initial_link_meshes_list:
    for mesh in initial_link_meshes_list:
        robot_mesh_for_vis += mesh
else: print("警告：未能生成初始连杆用于可视化。")

# 应用基座变换
robot_mesh_for_vis.transform(T_world_base)
robot_mesh_for_vis.compute_vertex_normals()


# --- 3.6 初始化并设置非阻塞式可视化窗口 ---
print("初始化可视化窗口...")
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='机器人角度迭代过程 (按 Q 或关闭窗口结束)')

# 添加静态物体：球体和坐标系
vis.add_geometry(sphere_mesh)
world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
robot_base_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
robot_base_frame_vis.transform(T_world_base)
vis.add_geometry(world_frame)
vis.add_geometry(robot_base_frame_vis)

# 添加机器人网格 (这个对象会被更新)
geometry_added = False
if robot_mesh_for_vis.has_triangles():
    vis.add_geometry(robot_mesh_for_vis)
    geometry_added = True
    print("初始机器人姿态已添加到可视化。")
else:
    print("初始机器人网格无效，未添加到可视化。")

# 设置初始视角等 (可选)
# ctr = vis.get_view_control()
# ctr.set_zoom(0.8)

print("可视化窗口已创建，开始循环...")
time.sleep(1) # 短暂停顿，让窗口有时间完全显示

# --- 4. 循环遍历关节角度并检测接触 ---
total_iterations = len(theta1_range_deg) * len(theta2_range_deg) * len(theta3_range_deg)
print(f"总共需要检查 {total_iterations} 种角度组合。按 Q 或关闭窗口可提前中止。")
iteration_count = 0
loop_start_time = time.time()
window_open = True # 标志窗口是否仍然打开
contact_marker_added = False # 标志是否已添加接触点

for theta1_deg in theta1_range_deg:
    if first_contact_found or not window_open: break
    for theta2_deg in theta2_range_deg:
        if first_contact_found or not window_open: break
        for theta3_deg in theta3_range_deg:
            if first_contact_found or not window_open: break

            iteration_count += 1
            joint_angles_rad = np.radians([theta1_deg, theta2_deg, theta3_deg])

            # --- 更新机器人网格 ---
            T_0_1 = create_transformation_matrix(joint_angles_rad[0], link_lengths[0])
            T_1_2 = create_transformation_matrix(joint_angles_rad[1], link_lengths[1])
            # T_2_3 = create_transformation_matrix(joint_angles_rad[2], link_lengths[2])
            T_0_j1 = T_0_1
            T_0_j2 = T_0_j1 @ T_1_2

            current_link_meshes = []
            if link1_base_mesh:
                link1_transformed = copy.deepcopy(link1_base_mesh)
                link1_transformed.transform(T_0_j1)
                link1_transformed.paint_uniform_color(link_colors[0])
                current_link_meshes.append(link1_transformed)
            if link2_base_mesh:
                link2_transformed = copy.deepcopy(link2_base_mesh)
                link2_transformed.transform(T_0_j2)
                link2_transformed.paint_uniform_color(link_colors[1])
                current_link_meshes.append(link2_transformed)

            robot_mesh_base_frame = o3d.geometry.TriangleMesh()
            if current_link_meshes:
                for mesh in current_link_meshes: robot_mesh_base_frame += mesh
            else: continue

            robot_mesh_world_frame = copy.deepcopy(robot_mesh_base_frame)
            robot_mesh_world_frame.transform(T_world_base)
            robot_mesh_world_frame.compute_vertex_normals()

            # --- 更新可视化窗口中的机器人 ---
            if geometry_added and robot_mesh_world_frame.has_triangles():
                robot_mesh_for_vis.vertices = robot_mesh_world_frame.vertices
                robot_mesh_for_vis.triangles = robot_mesh_world_frame.triangles
                robot_mesh_for_vis.vertex_normals = robot_mesh_world_frame.vertex_normals
                robot_mesh_for_vis.vertex_colors = robot_mesh_world_frame.vertex_colors
                vis.update_geometry(robot_mesh_for_vis)
            elif not geometry_added and robot_mesh_world_frame.has_triangles():
                 robot_mesh_for_vis = robot_mesh_world_frame
                 vis.add_geometry(robot_mesh_for_vis)
                 geometry_added = True

            # --- 控制可视化刷新频率和窗口事件 ---
            if iteration_count % visualization_update_skips == 0 or iteration_count == 1:
                if not vis.poll_events(): window_open = False; break # 处理事件, 如果关闭则退出
                vis.update_renderer() # 更新渲染
                # 打印进度
                current_time = time.time()
                elapsed = current_time - loop_start_time
                est_total_time = (elapsed / iteration_count) * total_iterations if iteration_count > 0 else 0
                print(f"\r进度: {iteration_count}/{total_iterations} | "
                      f"角度(deg): ({theta1_deg}, {theta2_deg}, {theta3_deg}) | "
                      f"已用时: {elapsed:.1f}s | 预计总时间: {est_total_time:.1f}s", end="")

            # --- 4.3 进行接触检测 ---
            if not robot_mesh_world_frame.has_triangles(): continue
            try:
                vertices = np.asarray(robot_mesh_world_frame.vertices, dtype=np.float64)
                faces = np.asarray(robot_mesh_world_frame.triangles, dtype=np.int32)
                if np.any(faces >= len(vertices)) or np.any(faces < 0): continue
                tm_robot = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                if not tm_robot.is_volume or len(tm_robot.faces) == 0 : continue

                proximity_query = trimesh.proximity.ProximityQuery(tm_robot)
                distances_sphere_to_robot, points_on_robot = proximity_query.closest_point(tm_sphere.vertices)
                if distances_sphere_to_robot.size == 0: continue

                min_dist_idx = np.argmin(distances_sphere_to_robot)
                min_dist = distances_sphere_to_robot[min_dist_idx]

                if min_dist < contact_distance_threshold:
                    contact_point_on_sphere_approx = tm_sphere.vertices[min_dist_idx]
                    contact_point_on_robot = points_on_robot[min_dist_idx]
                    contact_marker_pos = (contact_point_on_sphere_approx + contact_point_on_robot) / 2.0

                    print(f"\n*** 接触发现于角度 (deg): ({theta1_deg}, {theta2_deg}, {theta3_deg}) ***")
                    print(f"  最小距离: {min_dist:.6f}")
                    print(f"  接触点标记位置: {contact_marker_pos}")

                    first_contact_found = True
                    first_contact_angles = np.array([theta1_deg, theta2_deg, theta3_deg])
                    first_contact_point = contact_marker_pos

                    # *** 在当前窗口添加接触点标记 ***
                    if not contact_marker_added: # 确保只添加一次
                        print("  在可视化窗口中添加接触点标记...")
                        contact_marker_radius = link_radius * 0.6
                        contact_marker_color = [1.0, 0.0, 0.0] # 红色
                        marker = o3d.geometry.TriangleMesh.create_sphere(radius=contact_marker_radius)
                        marker.translate(first_contact_point)
                        marker.paint_uniform_color(contact_marker_color)
                        vis.add_geometry(marker)
                        contact_marker_added = True
                        # 强制刷新一次显示标记
                        if not vis.poll_events(): window_open = False; break
                        vis.update_renderer()

                    # break # 跳出最内层循环 (已在循环开始处检查 first_contact_found)

            except ValueError as ve: continue
            except Exception as e: continue
        # --- End of theta3 loop ---
    # --- End of theta2 loop ---
# --- End of theta1 loop ---
print() # 换行

loop_end_time = time.time()
print(f"\n角度遍历和接触检测完成。总耗时: {loop_end_time - loop_start_time:.2f} 秒。")

# --- 5. 保持窗口打开直到用户关闭 ---
if window_open:
    if first_contact_found:
        print(f"\n找到接触！最终状态显示在窗口中。角度(deg): {first_contact_angles}")
        print("请手动关闭可视化窗口以结束脚本。")
    else:
        print("\n在指定范围内未检测到接触。最终状态显示在窗口中。")
        print("请手动关闭可视化窗口以结束脚本。")

    # 循环以保持窗口打开和响应
    while window_open:
        if not vis.poll_events(): # 处理事件，如果用户关闭窗口则返回 False
            window_open = False
        vis.update_renderer() # 保持渲染
        time.sleep(0.01) # 短暂休眠，降低CPU占用

# --- 6. 清理 ---
print("正在关闭可视化窗口...")
vis.destroy_window() # 销毁窗口资源
print("\n脚本执行完毕。")