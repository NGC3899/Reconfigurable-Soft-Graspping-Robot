# -*- coding: utf-8 -*-
import numpy as np
import pyvista as pv
import numpy.linalg as LA
import torch
import torch.nn as nn
import joblib
import sys
from scipy.spatial.distance import cdist # 用于近邻搜索
import traceback # 用于打印详细错误
import time # 用于可视化暂停

# --- 1. ML 模型定义 ---
class MLPRegression(nn.Module):
    """一个简单的多层感知机用于回归"""
    def __init__(self, input_dim, output_dim, h1, h2, h3):
        super(MLPRegression, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, h3), nn.Tanh(),
            nn.Linear(h3, output_dim) )
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
sphere_translation = np.array([0, 0, sphere_radius + 35.0]) # <<< 球体位置
show_axes = True; finger_color = 'lightcoral'; tray_color = 'tan'; sphere_color = 'blue'
colliding_point_color = 'magenta'; intersection_point_color = 'yellow'; overlap_point_color = 'orange'
collision_threshold = 1.0; overlap_threshold = 1e-4
friction_coefficient = 0.5; eigenvalue_threshold = 1e-6
max_pressure = 40000.0
pressure_step = 100.0 # <<< 调整压力增加步长
visualization_update_interval = 0.05 # 秒，控制可视化更新频率
contact_marker_radius = 1.0
contact_normal_length = 5.0
contact_plane_size = 4.0

# --- 5. 辅助函数 ---
def create_rotation_matrix(axis, angle_rad):
    axis = np.asarray(axis).astype(float); axis /= LA.norm(axis)
    a = np.cos(angle_rad / 2.0); b, c, d = -axis * np.sin(angle_rad / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d; bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)], [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)], [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
def create_rotation_matrix_x(a): return create_rotation_matrix([1,0,0], a)
def create_rotation_matrix_y(a): return create_rotation_matrix([0,1,0], a)
def create_rotation_matrix_z(a): return create_rotation_matrix([0,0,1], a)
def create_translation_matrix(t): matrix=np.identity(4); matrix[:3, 3]=t; return matrix
def create_transformation_matrix(r, t): matrix=np.identity(4); matrix[:3, :3]=r if r is not None else np.identity(3); matrix[:3, 3]=t if t is not None else np.zeros(3); return matrix
def generate_sphere_points(radius, num_points):
    points = []; phi = np.pi * (3. - np.sqrt(5.))
    for i in range(num_points):
        z = 1 - (i / float(num_points - 1)) * 2; radius_at_z = np.sqrt(1 - z * z); theta = phi * i
        x = np.cos(theta) * radius_at_z; y = np.sin(theta) * radius_at_z; points.append((x, y, z));
    return np.array(points) * radius
def transform_points(points, matrix):
    if points.shape[1] != 3: raise ValueError("Points shape");
    if matrix.shape != (4, 4): raise ValueError("Matrix shape");
    num_pts = points.shape[0]; h_points = np.hstack((points, np.ones((num_pts, 1))))
    t_h = h_points @ matrix.T; return t_h[:, :3] / t_h[:, 3, np.newaxis]
def load_initial_coordinates(file_path, expected_nodes):
    try: coords = np.loadtxt(file_path, dtype=np.float32, usecols=(0, 1, 2))
    except FileNotFoundError: print(f"错误：找不到文件 {file_path}"); return None
    except Exception as e: print(f"加载初始坐标时出错: {e}"); return None
    if coords.shape == (expected_nodes, 3): print(f"成功加载 {coords.shape[0]} 个初始节点坐标。"); return coords
    else: print(f"错误：坐标形状 {coords.shape} 与预期 ({expected_nodes}, 3) 不符。"); return None
def create_faces_array(num_nodes_per_curve):
    faces = []; num_quads = num_nodes_per_curve - 1
    if num_quads <= 0: return np.array([], dtype=int)
    for i in range(num_quads): p1, p2=i, i+1; p3, p4=(i+1)+num_nodes_per_curve, i+num_nodes_per_curve; faces.append([4,p1,p2,p3,p4]);
    return np.hstack(faces)
def sort_points_spatially(points):
    if points is None or points.shape[0] < 2: return points
    num_points = points.shape[0]; sorted_indices = []; remaining_indices = list(range(num_points))
    start_node_index = np.argmin(points[:, 0]); current_index = start_node_index
    sorted_indices.append(current_index); remaining_indices.pop(remaining_indices.index(current_index))
    while remaining_indices:
        last_point = points[current_index, np.newaxis]; remaining_points = points[remaining_indices]
        distances = cdist(last_point, remaining_points)[0]
        nearest_neighbor_relative_index = np.argmin(distances)
        nearest_neighbor_absolute_index = remaining_indices[nearest_neighbor_relative_index]
        sorted_indices.append(nearest_neighbor_absolute_index); current_index = nearest_neighbor_absolute_index
        remaining_indices.pop(nearest_neighbor_relative_index)
    return points[sorted_indices]
def get_orthogonal_vectors(normal_vector):
    n = np.asarray(normal_vector).astype(float); norm_n = LA.norm(n)
    if norm_n < 1e-9: raise ValueError("Normal vector zero.")
    n /= norm_n
    if np.abs(n[0]) > 0.9: v_arbitrary = np.array([0., 1., 0.])
    else: v_arbitrary = np.array([1., 0., 0.])
    t1 = np.cross(n, v_arbitrary); norm_t1 = LA.norm(t1)
    if norm_t1 < 1e-9:
        v_arbitrary = np.array([0., 0., 1.]); t1 = np.cross(n, v_arbitrary); norm_t1 = LA.norm(t1)
        if norm_t1 < 1e-9:
             if np.abs(n[0]) > 0.9: t1 = np.array([0.,1.,0.]); t2_temp = np.array([0.,0.,1.])
             elif np.abs(n[1]) > 0.9: t1 = np.array([1.,0.,0.]); t2_temp = np.array([0.,0.,1.])
             else: t1 = np.array([1.,0.,0.]); t2_temp = np.array([0.,1.,0.])
             t1 /= LA.norm(t1); t2 = t2_temp / LA.norm(t2_temp); t2 = np.cross(n, t1); t2 /= LA.norm(t2); return t1, t2
    t1 /= norm_t1; t2_temp = np.cross(n, t1); norm_t2 = LA.norm(t2_temp)
    if norm_t2 < 1e-9: raise ValueError("Cannot compute tangent 2.")
    t2 = t2_temp / norm_t2; return t1, t2
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
def calculate_gii_multi_contact(contacts_info, object_centroid, mu, eigenvalue_threshold):
    if not contacts_info or len(contacts_info) == 0: print("无接触点信息，无法计算 GII。"); return None
    all_wrenches = []
    for contact in contacts_info:
        if isinstance(contact, dict): pt_on_mesh = contact.get('pt_on_mesh'); normal_finger = contact.get('normal')
        elif isinstance(contact, (tuple, list)) and len(contact) >= 6: pt_on_mesh = contact[4]; normal_finger = contact[5]
        else: print("警告：接触点信息格式不兼容，跳过此点。"); continue
        if pt_on_mesh is None or normal_finger is None: print("警告：接触点坐标或法向量缺失，跳过此点。"); continue
        norm_mag = LA.norm(normal_finger)
        if norm_mag < 1e-6: print("警告：接触点法向量接近零，跳过此点。"); continue
        n_contact = - (normal_finger / norm_mag)
        try: t1, t2 = get_orthogonal_vectors(n_contact)
        except ValueError as e: print(f"警告：计算切向量失败: {e}，跳过此接触点。"); continue
        r_contact = pt_on_mesh - object_centroid
        d1 = n_contact + mu * t1; d2 = n_contact - mu * t1; d3 = n_contact + mu * t2; d4 = n_contact - mu * t2
        w1 = np.concatenate((d1, np.cross(r_contact, d1))); w2 = np.concatenate((d2, np.cross(r_contact, d2)))
        w3 = np.concatenate((d3, np.cross(r_contact, d3))); w4 = np.concatenate((d4, np.cross(r_contact, d4)))
        all_wrenches.extend([w1, w2, w3, w4])
    if not all_wrenches: print("未能成功计算任何接触点的 Wrench。"); return None
    grasp_matrix_G = np.column_stack(all_wrenches); print(f"构建总抓取矩阵 G，形状: {grasp_matrix_G.shape}")
    J = grasp_matrix_G @ grasp_matrix_G.T
    try:
        eigenvalues = LA.eigvalsh(J); non_zero_eigenvalues = eigenvalues[eigenvalues > eigenvalue_threshold]
        if len(non_zero_eigenvalues) > 0:
            lambda_min = np.min(non_zero_eigenvalues); lambda_max = np.max(non_zero_eigenvalues)
            if lambda_max < eigenvalue_threshold: return 0.0
            if lambda_min < -eigenvalue_threshold: print(f"警告: lambda_min < 0 ({lambda_min:.2e})"); return None
            elif lambda_min < 0: lambda_min = 0.0
            return np.sqrt(lambda_min / lambda_max) if lambda_max > 0 else 0.0
        else: return 0.0
    except LA.LinAlgError as e_eig: print(f"计算特征值时出错: {e_eig}"); return None

# --- 主脚本 ---
if __name__ == '__main__':
    # --- 初始化 ---
    initial_coords_ref = load_initial_coordinates(INITIAL_COORDS_PATH, NODE_COUNT)
    if initial_coords_ref is None: sys.exit()
    model, scaler_X, scaler_y, device = load_prediction_components( MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH, INPUT_DIM, OUTPUT_DIM, HIDDEN_LAYER_1, HIDDEN_LAYER_2, HIDDEN_LAYER_3 )
    if model is None: sys.exit()

    width_translation_vector = np.array([0, finger_width, 0]);
    bottom_node_index_ref = np.argmin(initial_coords_ref[:, 0])
    ref_bottom_midpoint = initial_coords_ref[bottom_node_index_ref] + width_translation_vector / 2.0
    T1_translate = create_translation_matrix(-ref_bottom_midpoint)
    rotation_ref_to_local = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    T2_rotate = create_transformation_matrix(rotation_ref_to_local, None)
    faces_np = create_faces_array(NODE_COUNT)
    if faces_np.size == 0: print("错误：无法生成面片。"); sys.exit()

    tray = pv.Cylinder(center=tray_center, radius=tray_radius, height=tray_height, direction=(0, 0, 1), resolution=100)
    object_points_local = generate_sphere_points(sphere_radius, num_sphere_points)
    object_transform = create_transformation_matrix(create_rotation_matrix_z(sphere_rotation_angle_z), sphere_translation)
    object_points_global = transform_points(object_points_local, object_transform)
    object_centroid = np.mean(object_points_global, axis=0)
    num_object_points = object_points_global.shape[0]
    print(f"对象点云质心 (全局): {object_centroid.round(3)}")

    color_map = { 'Non-Contact': list(pv.Color(sphere_color).int_rgb) + [255], 'Contact': list(pv.Color(colliding_point_color).int_rgb) + [255], 'Overlap': list(pv.Color(overlap_point_color).int_rgb) + [255], 'Intersection': list(pv.Color(intersection_point_color).int_rgb) + [255], 'Gray': list(pv.Color('grey').int_rgb) + [255] }

    # --- 初始化自动调节状态 ---
    current_pressures = np.array([0.0, 0.0, 0.0])
    finger_contact_achieved = [False, False, False]
    final_state_overall = "Initializing"
    grasp_isotropy_index = None
    iteration_count = 0
    all_contacts_this_step = []
    closest_contact_per_finger = [None] * 3

    # --- 初始化绘图器 ---
    plotter = pv.Plotter(window_size=[1000, 800]); plotter.camera_position = 'xy'
    plotter.add_mesh(tray, color=tray_color, opacity=0.5, name='tray')
    plotter.add_mesh(pv.PointSet(object_centroid), color='green', point_size=15, render_points_as_spheres=True, name='centroid')
    if show_axes: plotter.add_axes_at_origin(labels_off=False)
    finger_actors = [None] * 3 # 初始化为 None
    point_cloud_polydata = pv.PolyData(object_points_global)
    initial_rgba_colors = np.array([color_map['Non-Contact']] * num_object_points, dtype=np.uint8)
    point_cloud_polydata.point_data['colors'] = initial_rgba_colors
    point_cloud_actor = plotter.add_mesh(point_cloud_polydata, scalars='colors', rgba=True, style='points', point_size=7, render_points_as_spheres=True, name='sphere_points')
    # status_text_actor 的引用不再需要

    print("\n开始自动调节压力...")

    # --- 自动调节循环 ---
    try:
        while True:
            iteration_count += 1
            print(f"\n--- Iteration {iteration_count} ---")
            # --- 1. 增加压力 ---
            pressure_changed = False
            for i in range(3):
                if not finger_contact_achieved[i] and current_pressures[i] < max_pressure:
                    current_pressures[i] += pressure_step
                    current_pressures[i] = min(current_pressures[i], max_pressure)
                    pressure_changed = True
            print(f"当前压力: P1={current_pressures[0]:.0f}, P2={current_pressures[1]:.0f}, P3={current_pressures[2]:.0f}")

            # --- 2. 计算变形 ---
            all_predictions_valid = True
            deformed_finger_meshes_world_this_iter = []
            for i in range(3):
                displacements_matrix = predict_displacements_for_pressure(model, scaler_X, scaler_y, device, current_pressures[i])
                if displacements_matrix is None: print(f"手指 {i+1} 预测失败。"); all_predictions_valid = False; break
                deformed_curve1_ref_unordered = initial_coords_ref + displacements_matrix
                curve2_ref = initial_coords_ref + width_translation_vector
                deformed_curve2_ref_unordered = curve2_ref + displacements_matrix
                sorted_deformed_curve1_ref = sort_points_spatially(deformed_curve1_ref_unordered)
                sorted_deformed_curve2_ref = sort_points_spatially(deformed_curve2_ref_unordered)
                sorted_deformed_vertices_ref = np.vstack((sorted_deformed_curve1_ref, sorted_deformed_curve2_ref))
                deformed_mesh_ref = pv.PolyData(sorted_deformed_vertices_ref, faces=faces_np)

                angle_rad = np.radians([0, 120, 240][i])
                rot_angle_z_placing = angle_rad + np.pi / 2.0
                rot_z_placing = create_rotation_matrix_z(rot_angle_z_placing)
                target_pos_on_circle = np.array([ finger_placement_radius * np.cos(angle_rad), finger_placement_radius * np.sin(angle_rad), 0.0 ])
                T3_place = create_transformation_matrix(rot_z_placing, target_pos_on_circle)
                T_final = T3_place @ T2_rotate @ T1_translate
                final_transformed_mesh = deformed_mesh_ref.transform(T_final, inplace=False)
                final_transformed_mesh.clean(inplace=True)
                final_transformed_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True, auto_orient_normals=True)
                deformed_finger_meshes_world_this_iter.append(final_transformed_mesh)

            if not all_predictions_valid: break

            # --- 3. 碰撞检测 ---
            all_contacts_this_step = [] # 重置
            object_point_status = ['Non-Contact'] * num_object_points # 重置
            contacting_fingers_this_step = set()
            min_dist_per_finger = [float('inf')] * 3 # 重置每指最近距离
            closest_contact_per_finger = [None] * 3 # 重置每指最近接触信息

            for obj_pt_idx, obj_point in enumerate(object_points_global):
                closest_dist_for_this_pt = float('inf'); info_for_this_pt = None
                finger_idx_for_this_pt = -1

                for finger_idx, finger_mesh in enumerate(deformed_finger_meshes_world_this_iter):
                    if finger_mesh is None or finger_mesh.n_cells == 0 or 'Normals' not in finger_mesh.cell_data: continue
                    try:
                        closest_cell_id, pt_on_mesh = finger_mesh.find_closest_cell(obj_point, return_closest_point=True)
                        if closest_cell_id < 0 or closest_cell_id >= finger_mesh.n_cells: continue
                        dist = LA.norm(obj_point - pt_on_mesh)

                        # 更新该对象点的全局最近信息
                        if dist < closest_dist_for_this_pt:
                             closest_dist_for_this_pt = dist
                             finger_idx_for_this_pt = finger_idx
                             if closest_cell_id < len(finger_mesh.cell_data['Normals']):
                                normal = finger_mesh.cell_normals[closest_cell_id]
                                info_for_this_pt = (dist, finger_idx, closest_cell_id, pt_on_mesh, normal)
                             else: info_for_this_pt = None

                        # 更新该手指的最近接触信息
                        if dist < collision_threshold and dist < min_dist_per_finger[finger_idx]:
                             min_dist_per_finger[finger_idx] = dist
                             if closest_cell_id < len(finger_mesh.cell_data['Normals']):
                                normal_f = finger_mesh.cell_normals[closest_cell_id]
                                closest_contact_per_finger[finger_idx] = (dist, obj_pt_idx, finger_idx, closest_cell_id, pt_on_mesh, normal_f)
                             else: closest_contact_per_finger[finger_idx] = None

                    except RuntimeError: pass
                    except IndexError: pass
                    except Exception as e_coll: print(f"点 {obj_pt_idx} 与手指 {finger_idx+1} 检测出错: {e_coll}"); info_for_this_pt=None; continue

                # 根据全局最近信息判断点状态
                if info_for_this_pt:
                    dist, finger_idx, cell_id, pt_on_mesh, normal = info_for_this_pt # 使用最近手指的信息
                    if dist < overlap_threshold:
                        object_point_status[obj_pt_idx] = 'Overlap'; all_contacts_this_step.append({'type': 'Overlap', 'obj_pt_idx': obj_pt_idx})
                    elif dist < collision_threshold:
                        object_point_status[obj_pt_idx] = 'Contact'
                        contacting_fingers_this_step.add(finger_idx) # 记录接触的手指
                        contact_data = {'type': 'Contact', 'obj_pt_idx': obj_pt_idx, 'dist': dist, 'finger_idx': finger_idx, 'cell_id': cell_id, 'pt_on_mesh': pt_on_mesh, 'normal': normal}
                        all_contacts_this_step.append(contact_data)
            print("碰撞检测完成。")

            # --- 4. 更新手指接触状态 ---
            for i in range(3):
                if i in contacting_fingers_this_step:
                    finger_contact_achieved[i] = True
            print(f"接触状态: Finger1={finger_contact_achieved[0]}, Finger2={finger_contact_achieved[1]}, Finger3={finger_contact_achieved[2]}")

            # --- 5. 更新可视化 ---
            for i in range(3):
                mesh_to_update = deformed_finger_meshes_world_this_iter[i]
                if mesh_to_update is not None:
                    if finger_actors[i] is None: # 首次添加
                        finger_actors[i] = plotter.add_mesh(mesh_to_update, color=finger_color, style='surface', edge_color='grey', opacity=0.85, smooth_shading=True, name=f'finger_{i}')
                    else: # 后续更新
                        # --- 已修正：使用 mapper.SetInputData 更新 Actor ---
                        finger_actors[i].mapper.SetInputData(mesh_to_update)

            # 更新点云颜色
            rgba_colors_array = np.array([color_map.get(status, color_map['Gray']) for status in object_point_status], dtype=np.uint8)
            if rgba_colors_array.shape == (num_object_points, 4):
                 point_cloud_polydata.point_data['colors'] = rgba_colors_array
            else: print("颜色数组形状错误")

            # 更新状态文本
            status_text = f"Iter: {iteration_count} P=[{current_pressures[0]:.0f}, {current_pressures[1]:.0f}, {current_pressures[2]:.0f}] Contacts={list(finger_contact_achieved)}"
            plotter.add_text(status_text, position="upper_edge", font_size=10, name='status_text') # 用 name 更新

            plotter.render()
            time.sleep(visualization_update_interval)

            # --- 6. 检查终止条件 ---
            all_contacted = all(finger_contact_achieved)
            should_stop = True
            for i in range(3):
                if not finger_contact_achieved[i] and current_pressures[i] < max_pressure:
                    should_stop = False; break
            if should_stop:
                 if all_contacted: final_state_overall = "All Contacted"
                 else: final_state_overall = "Max Pressure Reached"
                 print(f"达到终止条件: {final_state_overall}")
                 break

    except KeyboardInterrupt: print("\n用户手动中断。"); final_state_overall = "Interrupted"
    except Exception as e: print(f"\n自动调节循环中发生错误: {e}"); traceback.print_exc(); final_state_overall = "Error"

    # --- 循环结束后 ---
    print(f"\n自动调节结束。最终状态: {final_state_overall}")
    print(f"最终压力: P1={current_pressures[0]:.0f}, P2={current_pressures[1]:.0f}, P3={current_pressures[2]:.0f}")

    # --- 计算最终 GII ---
    grasp_isotropy_index = None
    final_contacts_for_gii = [info for info in closest_contact_per_finger if info is not None]

    final_has_overlap = any(s == 'Overlap' for s in object_point_status)

    if final_has_overlap: print("最终存在重叠，不计算GII。")
    elif len(final_contacts_for_gii) == 3: # 必须有三个接触点才计算
        print(f"找到 3 个接触点 (每个手指最近点)，计算 GII...")
        try:
            grasp_isotropy_index = calculate_gii_multi_contact(
                final_contacts_for_gii, object_centroid, friction_coefficient, eigenvalue_threshold)
            print(f"最终 GII (mu={friction_coefficient}): {grasp_isotropy_index if grasp_isotropy_index is not None else '无法计算'}")
        except Exception as e_gii: print(f"最终 GII 计算出错: {e_gii}"); traceback.print_exc()
    else: print(f"最终有效接触点数量 ({len(final_contacts_for_gii)}) 不足 3 个，不计算 GII。")

    # --- 可视化最终接触点、法线和切平面 ---
    if len(final_contacts_for_gii) > 0:
        print("正在添加最终接触点可视化元素...")
        # --- 修改：定义接触点颜色列表 ---
        contact_marker_colors = ['red', 'green', 'cyan'] # 为 3 个接触点定义不同颜色
        # --- 结束修改 ---
        for k, contact_info in enumerate(final_contacts_for_gii):
             if contact_info is None: continue
             # (dist, obj_pt_idx, finger_idx, cell_id, pt_on_mesh, normal_finger)
             pt_on_mesh = contact_info[4]
             normal_finger = contact_info[5]

             # --- 修改：使用预定义的颜色 ---
             plotter.add_mesh(pv.Sphere(center=pt_on_mesh, radius=contact_marker_radius),
                              color=contact_marker_colors[k % len(contact_marker_colors)], # 循环使用颜色
                              name=f'contact_marker_{k}')
             # --- 结束修改 ---

             if normal_finger is not None and LA.norm(normal_finger) > 1e-6:
                 norm_viz = normal_finger / LA.norm(normal_finger)
                 plotter.add_arrows(cent=pt_on_mesh, direction=norm_viz, mag=contact_normal_length,
                                    color='black', name=f'contact_normal_{k}')
                 try:
                     tangent_plane = pv.Plane(center=pt_on_mesh, direction=norm_viz,
                                             i_size=contact_plane_size, j_size=contact_plane_size,
                                             i_resolution=1, j_resolution=1)
                     plotter.add_mesh(tangent_plane, color='gray', opacity=0.4,
                                      style='surface', name=f'tangent_plane_{k}')
                 except Exception as e_plane: print(f"绘制接触点 {k+1} 的切平面时出错: {e_plane}")

    # --- 更新最终状态文本 ---
    gii_text = f"Final GII: {grasp_isotropy_index:.4f}" if grasp_isotropy_index is not None else "Final GII: N/A"
    final_status_text = f"Final State: {final_state_overall}\nP=[{current_pressures[0]:.0f}, {current_pressures[1]:.0f}, {current_pressures[2]:.0f}]\n{gii_text}"
    plotter.add_text(final_status_text, position="upper_edge", font_size=10, name='status_text') # 使用 name 更新

    print("\n按 Q 键退出可视化窗口。")
    plotter.render()
    plotter.show()

    print("\n程序结束。")