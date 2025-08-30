import open3d as o3d
import numpy as np
import os

def stl_to_ply(stl_file_path, ply_file_path, target_points=3500, convert_mm_to_m=True, clipping_height=None):
    """
    将 STL 文件转换为均匀抽稀的蓝色 PLY 点云文件，并可选择沿Z轴截取。

    参数:
    stl_file_path (str): 输入的 STL 文件路径。
    ply_file_path (str): 输出的 PLY 文件路径。
    target_points (int): 目标点云数量。
    convert_mm_to_m (bool): 如果为 True，则将坐标单位从毫米转换为米。
    clipping_height (float or None): 沿Z轴截取平面的高度。只保留Z坐标大于此值的点。如果为 None，则不截取。
    """
    # 1. 检查文件是否存在
    if not os.path.exists(stl_file_path):
        print(f"错误：找不到文件 '{stl_file_path}'")
        return

    # 2. 加载 STL 文件
    try:
        mesh = o3d.io.read_triangle_mesh(stl_file_path)
    except Exception as e:
        print(f"读取 STL 文件时出错: {e}")
        return

    if mesh.is_empty():
        print("错误：STL 文件为空或无法解析。")
        return

    # 3. 从网格表面采样点云
    pcd = mesh.sample_points_uniformly(number_of_points=target_points)

    if not pcd.has_points():
        print("错误：无法从 STL 文件中采样点云。")
        return
        
    # 4. 单位转换 (从 mm 到 m)
    if convert_mm_to_m:
        points_np = np.asarray(pcd.points)
        points_np /= 1000.0
        pcd.points = o3d.utility.Vector3dVector(points_np)
        print("单位转换：已将点云坐标从毫米 (mm) 转换为米 (m)。")

    # 5. 新增：沿Z轴截取点云
    if clipping_height is not None:
        points_np = np.asarray(pcd.points)
        
        # 创建一个布尔掩码，选择Z坐标大于截取高度的点
        mask = points_np[:, 2] > clipping_height
        
        # 使用掩码过滤点
        points_cropped = points_np[mask]
        
        if points_cropped.shape[0] > 0:
            # 用截取后的点更新点云对象
            pcd.points = o3d.utility.Vector3dVector(points_cropped)
            print(f"点云截取：已在 Z={clipping_height} 处截取，保留了上半部分 {len(pcd.points)} 个点。")
        else:
            # 如果截取后没有点剩下，则发出警告并使用原始点云
            print(f"警告：在 Z={clipping_height} 处截取后没有剩余点。将保存截取前的点云。")

    # 6. 定义颜色
    # 您之前的版本使用了肉色，这里保留。如果需要蓝色，可以改为 [0.0, 0.0, 1.0]
    custom_color = [0.5, 0.5, 0.5]  # RGB (肉色)
    pcd.paint_uniform_color(custom_color)

    # 7. 保存为 PLY 文件
    try:
        o3d.io.write_point_cloud(ply_file_path, pcd, write_ascii=True)
        print(f"成功将点云保存至 '{ply_file_path}'")
        print(f"最终点云数量: {len(pcd.points)}")
    except Exception as e:
        print(f"保存 PLY 文件时出错: {e}")

if __name__ == '__main__':
    # --- 请在这里配置您的文件路径和参数 ---
    
    # 输入的 STL 文件路径
    input_stl_file = 'C:\\Users\\admin\\Desktop\\3D_13.stl'
    
    # 输出的 PLY 文件路径
    output_ply_file = 'C:\\Users\\admin\\Desktop\\3D_13.ply' # 建议改名以作区分

    # --- 新增：在这里设置截取平面的Z轴高度 ---
    # 单位与转换后的点云一致 (如果 convert_mm_to_m=True, 则单位是米)
    # 将此值设置为 None 即可禁用截取功能
    clipping_z_height = None  # 示例：在 Z=0.005米 (5毫米) 的高度截取

    # 执行转换
    stl_to_ply(
        input_stl_file, 
        output_ply_file, 
        target_points=3500, 
        convert_mm_to_m=True,
        clipping_height=clipping_z_height # 传递截取高度
    )
