import open3d as o3d
import numpy as np
import os

def visualize_bunny_for_publication(file_path):
    """
    加载斯坦福兔子点云，以学术出版物级别的高质量标准进行渲染，并打开交互式窗口以便手动截图。

    参数:
    file_path (str): .ply 文件的完整路径。
    """
    # --- 1. 检查并加载点云文件 ---
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 '{file_path}'")
        return
    print(f"正在从 '{file_path}' 加载点云...")
    try:
        pcd = o3d.io.read_point_cloud(file_path)
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return
    if not pcd.has_points():
        print("错误：加载的点云中不包含任何点。")
        return
    print("点云加载成功！")

    # --- 2. 预处理点云 ---
    # 将点云中心移动到坐标原点
    center_point = pcd.get_center()
    pcd.translate(-center_point)
    
    # 【已修改】使用更小的体素尺寸进行下采样，以保留更多细节和轮廓
    voxel_size = 0.00035
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # --- 3. 【已修改】扩大虚拟相机拍摄范围以显示完整轮廓 ---
    # 我们将使用一个裁剪框来模拟一个更宽的视野，以捕捉完整的轮廓
    # 这种方法比 hidden_point_removal 保留的点更多，更能看清物体形状
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb_min = aabb.get_min_bound()
    aabb_max = aabb.get_max_bound()

    # 定义一个从Z轴正方向看的裁剪框
    # Z轴的最小边界从0开始，这样可以切掉兔子的后半部分
    # X和Y轴保持完整，以显示整个轮廓
    min_bound = np.array([aabb_min[0], aabb_min[1], 0])
    max_bound = np.array([aabb_max[0], aabb_max[1], aabb_max[2]])
    
    cropping_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    pcd_cropped = pcd.crop(cropping_box)
    
    # --- 4. 【关键优化】计算并定向法线以实现逼真光照 ---
    # 定义相机位置用于法线定向
    camera_location = np.array([0, 0, aabb_max[2] * 2])
    # 为每个点估算法线
    pcd_cropped.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    # 将法线的方向统一朝向相机位置，确保光照效果正确
    pcd_cropped.orient_normals_towards_camera_location(camera_location=camera_location)

    # --- 5. 【已修改】设置专业的颜色和材质 ---
    num_points = len(pcd_cropped.points)
    if num_points == 0:
        print("警告：裁剪后没有剩余的点。")
        return
    print(f"最终用于显示的点数: {num_points}")
    
    # 定义鲜亮的橙色
    bright_orange_color = np.array([1.0, 0.65, 0.0])
    colors = np.tile(bright_orange_color, (num_points, 1))
    pcd_cropped.colors = o3d.utility.Vector3dVector(colors)

    # --- 6. 设置渲染器并进行交互式可视化 ---
    # 创建一个可视化窗口，以便手动调整和截图
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Publication-Quality Bunny", width=1920, height=1080, visible=True)
    
    vis.add_geometry(pcd_cropped)

    # 获取渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0]) # 设置纯白背景
    opt.point_size = 2.5 # 增大点的大小
    opt.light_on = True # 开启光照

    # 设置一个固定的、效果好的初始相机视角
    view_control = vis.get_view_control()
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    extrinsic = np.array([
        [0.99, -0.04, 0.08, -0.005],
        [0.05, 0.99, -0.07, -0.01],
        [-0.08, 0.07, 0.99, 0.12],
        [0.0, 0.0, 0.0, 1.0]
    ])
    cam_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(cam_params)

    print("正在打开交互式窗口...")
    print("您可以自由旋转、缩放视角，然后手动截图。")
    print("完成后，请直接关闭窗口或按 'q' 键退出。")
    
    # 运行可视化窗口，程序会在此暂停直到窗口关闭
    vis.run()
    
    # 清理
    vis.destroy_window()


if __name__ == "__main__":
    bunny_file_path = r"C:\Users\admin\Desktop\Figure\stanford bunny\stanford-bunny\source\bunny_colored.ply"
    visualize_bunny_for_publication(bunny_file_path)
