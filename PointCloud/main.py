import open3d as o3d
import numpy as np
import os
import time
import copy # To avoid modifying original point clouds during visualization prep
import matplotlib.pyplot as plt # 用于生成颜色

print(f"Open3D version: {o3d.__version__}")

# --- 1. 参数设置 ---

# 文件路径
ply_file_path = r'C:\Users\admin\Desktop\merged_pointcloud.ply' # 请确保路径正确

# 去噪参数 (SOR) - 可选
enable_denoising = True
sor_nb_neighbors = 30
sor_std_ratio = 1.0

# 平面移除参数
max_plane_find_iterations = 5
plane_distance_threshold = 0.02 # RANSAC 内点阈值 (重要!)
plane_ransac_n = 3
plane_num_iterations = 1000
min_plane_points_ratio = 0.05

# *** 新增：DBSCAN 聚类参数 ***
enable_clustering = True # 是否启用聚类步骤
# eps: DBSCAN 的邻域距离阈值。需要仔细调整！
#   - 如果点云稀疏或物体部分分离，需要增大 eps
#   - 如果不同物体靠得很近，需要减小 eps
#   - 可以尝试设置为法线估计半径的几倍，或基于点云平均点间距调整
dbscan_eps = 0.05 # !!! 关键参数，需要根据点云尺度和密度调整 !!!
dbscan_min_points = 10 # 形成一个簇所需要的最小点数 (包括自身)

# 法线估计参数
normal_radius_ratio = 0.05
normal_max_nn = 30

# 重建参数 (泊松)
poisson_depth = 9
poisson_density_quantile = 0.01
min_points_for_reconstruction = 100

# --- 2. 加载点云 ---
# (与之前代码相同)
print(f"加载点云: {ply_file_path}")
if not os.path.exists(ply_file_path): print(f"错误: 文件未找到 - {ply_file_path}"); exit()
try: pcd_orig = o3d.io.read_point_cloud(ply_file_path)
except Exception as e: print(f"错误: 加载点云失败 - {e}"); exit()
if not pcd_orig.has_points(): print("错误: 点云为空."); exit()
print(f"原始点云点数: {len(pcd_orig.points)}")

# --- 3. 去噪 (可选) ---
# (与之前代码相同)
if enable_denoising:
    print(f"\n开始去噪 (SOR)...")
    start_time = time.time()
    try:
        pcd_processed, _ = pcd_orig.remove_statistical_outlier(
                                                 nb_neighbors=sor_nb_neighbors,
                                                 std_ratio=sor_std_ratio)
        print(f"去噪完成，耗时: {time.time() - start_time:.2f} 秒")
        print(f"去噪后点数: {len(pcd_processed.points)}")
    except Exception as e:
        print(f"去噪失败: {e}. 使用原始点云。")
        pcd_processed = pcd_orig
else:
    print("\n跳过去噪步骤.")
    pcd_processed = pcd_orig

# --- 4. 迭代移除平面 ---
# (与之前代码基本相同)
print("\n开始迭代移除平面...")
pcd_remaining = copy.deepcopy(pcd_processed) # 从(去噪后)点云开始
removed_planes_info = []
min_plane_points = int(len(pcd_remaining.points) * min_plane_points_ratio)
print(f"有效平面所需最小点数: {min_plane_points}")

iteration = 1
start_time_planes = time.time()
while iteration <= max_plane_find_iterations and len(pcd_remaining.points) >= min_plane_points:
    # (循环体与之前相同，不断更新 pcd_remaining)
    print(f"\n--- 平面移除迭代 {iteration} ---")
    print(f"当前剩余点数: {len(pcd_remaining.points)}")
    try:
        plane_model, inliers = pcd_remaining.segment_plane(distance_threshold=plane_distance_threshold,
                                                           ransac_n=plane_ransac_n,
                                                           num_iterations=plane_num_iterations)
    except Exception as e: print(f"错误: segment_plane 执行失败 - {e}. 停止移除."); break
    num_inliers = len(inliers)
    print(f"找到潜在平面，内点数: {num_inliers}")
    if num_inliers < min_plane_points: print(f"平面点数不足，停止移除。"); break
    [a, b, c, d] = plane_model
    print(f"移除平面 {iteration}: {a:.3f}x+{b:.3f}y+{c:.3f}z+{d:.3f}=0 ({num_inliers}点)")
    removed_planes_info.append({'model': plane_model, 'inliers_count': num_inliers})
    pcd_remaining = pcd_remaining.select_by_index(inliers, invert=True) # 保留外点
    iteration += 1
print(f"\n平面移除完成，耗时: {time.time() - start_time_planes:.2f} 秒")
print(f"总共移除了 {len(removed_planes_info)} 个平面。")
print(f"移除平面后剩余点数: {len(pcd_remaining.points)}")


# --- 4.5 *** 新增：聚类并保留最大簇 *** ---
if enable_clustering and pcd_remaining.has_points():
    print(f"\n开始聚类分析 (DBSCAN)...")
    print(f"DBSCAN 参数: eps={dbscan_eps}, min_points={dbscan_min_points}")
    start_time = time.time()

    try:
        # 执行 DBSCAN 聚类
        # labels 是一个列表，labels[i] 是第 i 个点的簇标签，-1 表示噪声
        labels = np.array(pcd_remaining.cluster_dbscan(eps=dbscan_eps,
                                                       min_points=dbscan_min_points,
                                                       print_progress=True)) # 显示聚类进度
    except Exception as e:
        print(f"错误: DBSCAN 聚类失败 - {e}")
        labels = np.array([]) # 出错则设置空标签

    if labels.size > 0:
        # 找出最大的簇标签 (忽略噪声标签 -1)
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True) # 只统计非噪声簇

        if unique_labels.size > 0:
            largest_cluster_label = unique_labels[np.argmax(counts)]
            largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
            num_noise_points = np.sum(labels == -1)

            print(f"聚类完成，耗时: {time.time() - start_time:.2f} 秒")
            print(f"  找到 {unique_labels.size} 个簇。")
            print(f"  最大簇的标签: {largest_cluster_label}, 包含 {len(largest_cluster_indices)} 个点。")
            print(f"  识别出 {num_noise_points} 个噪声点。")

            # 只保留最大簇的点云
            pcd_largest_cluster = pcd_remaining.select_by_index(largest_cluster_indices)
            print(f"保留最大簇后，点云数量变为: {len(pcd_largest_cluster.points)}")

            # 更新 pcd_remaining 用于后续步骤
            pcd_remaining = pcd_largest_cluster

            # 可选：可视化聚类结果（最大簇 vs 其他）
            # print("显示聚类结果（最大簇 vs 噪声/其他簇）...")
            # pcd_largest_cluster.paint_uniform_color([0, 0, 1]) # 最大簇为蓝色
            # pcd_noise_others = pcd_remaining.select_by_index(largest_cluster_indices, invert=True)
            # pcd_noise_others.paint_uniform_color([0.5, 0.5, 0.5]) # 其他为灰色
            # o3d.visualization.draw_geometries([pcd_largest_cluster, pcd_noise_others], window_name="聚类结果")

        else:
            print("聚类未能找到任何有效的簇 (所有点可能都被标记为噪声)。跳过保留最大簇步骤。")
    else:
        print("聚类标签为空，跳过保留最大簇步骤。")

else:
     if not pcd_remaining.has_points():
         print("\n点云在聚类前已为空，跳过聚类步骤。")
     else:
         print("\n跳过聚类步骤。")


# --- 5. 对剩余点云进行三维重建 ---
# (与之前代码相同，但现在操作的是聚类筛选后的 pcd_remaining)
if len(pcd_remaining.points) < min_points_for_reconstruction:
    print(f"\n错误: 最终剩余点数 ({len(pcd_remaining.points)}) 过少，无法进行重建。")
    exit()

print(f"\n开始对最终剩余的 {len(pcd_remaining.points)} 个点进行三维重建...")

# 5.1 估计法线
# (与之前代码相同)
print("估计法线...")
start_time = time.time()
bbox = pcd_remaining.get_axis_aligned_bounding_box()
diagonal_length = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
radius_normal = diagonal_length * normal_radius_ratio
print(f"  使用半径: {radius_normal:.4f}")
pcd_remaining.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=normal_max_nn))
print(f"法线估计完成，耗时: {time.time() - start_time:.2f} 秒")

# 5.2 执行泊松表面重建
# (与之前代码相同)
print(f"\n执行泊松表面重建... Depth={poisson_depth}")
start_time = time.time()
reconstructed_mesh = None
try:
    mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                                        pcd_remaining, depth=poisson_depth, linear_fit=False)
    print(f"泊松重建初步完成，耗时: {time.time() - start_time:.2f} 秒")
    # 清理
    print("清理泊松重建结果...")
    density_threshold_value = np.quantile(densities, poisson_density_quantile)
    vertices_to_remove = densities < density_threshold_value
    mesh_poisson.remove_vertices_by_mask(vertices_to_remove)
    mesh_poisson.remove_degenerate_triangles()
    mesh_poisson.remove_duplicated_vertices()
    mesh_poisson.remove_duplicated_triangles()
    mesh_poisson.remove_non_manifold_edges()
    mesh_poisson.compute_vertex_normals()
    reconstructed_mesh = mesh_poisson
    print("泊松重建与清理完成。")
except Exception as e: print(f"泊松重建失败: {e}")

# --- 6. 可视化最终重建结果 ---
# (与之前代码相同)
if reconstructed_mesh and reconstructed_mesh.has_triangles():
    print("\n显示最终重建的三维网格模型...")
    reconstructed_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    # 可选同时显示点云
    vis_geom = [reconstructed_mesh]
    # if pcd_remaining.has_points():
    #     pcd_remaining.paint_uniform_color([0,0,1])
    #     vis_geom.append(pcd_remaining)
    o3d.visualization.draw_geometries(vis_geom, window_name="最终重建网格")
    # 保存网格 (可选)
    # output_mesh_file = "reconstructed_object_clustered.ply"
    # o3d.io.write_triangle_mesh(output_mesh_file, reconstructed_mesh)
    # print(f"重建的网格已保存到: {output_mesh_file}")
else:
    print("\n未能成功生成可显示的重建网格。")

# --- 结束 ---
print("\n处理流程结束!")