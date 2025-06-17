import cv2
import numpy as np
import open3d as o3d
import os

# --- 配置参数 ---
# 图像文件名 (确保这些文件与脚本在同一目录，或提供完整路径)
COLOR_IMAGE_FILE = r"C:\Users\admin\Desktop\auto_aligned_color.png"
DEPTH_IMAGE_FILE = r"C:\Users\admin\Desktop\auto_aligned_depth_raw.png"
MASK_IMAGE_FILE = r"C:\Users\admin\Desktop\mask.png"

# 输出点云文件名
OUTPUT_PLY_FILE = r"C:\Users\admin\Desktop\roi_colored_point_cloud.ply"

# --- 相机内参和深度缩放因子 ---
# !!! 警告: 您必须将这些值替换为您的RealSense相机捕获图像时的真实参数 !!!
# 这些值通常在捕获图像时从相机SDK (如pyrealsense2的intrinsics对象) 获取
# 以下为示例值 (例如来自D435, 640x480分辨率，但您必须用自己的！)
INTRINSICS = {
    "width": 640,  # 图像宽度
    "height": 480, # 图像高度
    "fx": 606.8118,   # X轴焦距 (示例值)
    "fy": 606.6480,   # Y轴焦距 (示例值)
    "cx": 320.2263,   # X轴主点 (示例值, 通常为width/2)
    "cy": 256.1117    # Y轴主点 (示例值, 通常为height/2)
}
# 深度缩放因子：将深度图像中的原始值转换为米。
# RealSense相机通常以毫米为单位存储深度，所以scale通常是0.001
DEPTH_SCALE = 0.001 # 示例值 (例如0.001表示深度图中1000代表1米)


def create_roi_point_cloud(color_path, depth_path, mask_path, intrinsics, depth_scale):
    """
    根据对齐的RGB图、深度图和掩膜创建ROI区域的彩色点云。

    参数:
    color_path (str): 彩色图像路径。
    depth_path (str): 16位深度图像路径。
    mask_path (str): ROI掩膜图像路径 (白色为ROI)。
    intrinsics (dict): 相机内参 (fx, fy, cx, cy, width, height)。
    depth_scale (float): 深度图像的缩放因子，用于将深度值转换为米。

    返回:
    open3d.geometry.PointCloud: 生成的彩色点云对象，如果失败则返回None。
    """
    print("正在加载图像...")
    # 1. 加载图像
    color_image = cv2.imread(color_path, cv2.IMREAD_COLOR) # BGR格式
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) # 读取原始16位深度数据
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # 读取为灰度图

    if color_image is None:
        print(f"错误: 无法加载彩色图像 {color_path}")
        return None
    if depth_image is None:
        print(f"错误: 无法加载深度图像 {depth_path}")
        return None
    if mask_image is None:
        print(f"错误: 无法加载掩膜图像 {mask_path}")
        return None

    # 2. 检查图像尺寸是否一致
    if not (color_image.shape[:2] == depth_image.shape[:2] == mask_image.shape[:2]):
        print("错误: 彩色图、深度图和掩膜的尺寸不一致！")
        print(f"  彩色图尺寸: {color_image.shape[:2]}")
        print(f"  深度图尺寸: {depth_image.shape[:2]}")
        print(f"  掩膜图尺寸: {mask_image.shape[:2]}")
        return None
    
    height, width = color_image.shape[:2]
    if width != intrinsics["width"] or height != intrinsics["height"]:
        print(f"警告: 图像尺寸 ({width}x{height}) 与内参指定的尺寸 ({intrinsics['width']}x{intrinsics['height']}) 不符。")
        print("点云结果可能不准确。请确保内参与捕获这些图像时的配置一致。")
        # 你可以在这里选择停止，或者继续（但结果可能不正确）

    print("图像加载成功。")

    # 3. 处理掩膜
    # 确保掩膜是二值的 (ROI为True/1, 背景为False/0)
    # 假设掩膜中白色(255)代表ROI
    roi_mask = mask_image > 128 # 创建一个布尔掩膜

    # 4. 准备点云数据容器
    points = []
    colors = []

    print("正在生成点云数据...")
    # 5. 遍历像素，仅处理ROI区域
    # 获取ROI区域的像素坐标
    roi_pixels_v, roi_pixels_u = np.where(roi_mask) # v是行索引(y), u是列索引(x)

    for v, u in zip(roi_pixels_v, roi_pixels_u):
        # 获取深度值
        d = depth_image[v, u]

        # 如果深度值为0 (通常表示无效深度)，则跳过该点
        if d == 0:
            continue

        # 将原始深度值转换为米
        z_metric = d * depth_scale

        # 根据相机内参和深度值计算3D坐标 (相机坐标系)
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = Z
        x_metric = (u - intrinsics["cx"]) * z_metric / intrinsics["fx"]
        y_metric = (v - intrinsics["cy"]) * z_metric / intrinsics["fy"]
        
        points.append([x_metric, y_metric, z_metric])

        # 获取对应的颜色值 (OpenCV是BGR, Open3D需要RGB)
        # 并将颜色值从 [0, 255] 归一化到 [0, 1]
        b, g, r = color_image[v, u]
        colors.append([r / 255.0, g / 255.0, b / 255.0])

    if not points:
        print("错误: ROI区域没有有效的深度点来创建点云。")
        return None

    print(f"从ROI中提取了 {len(points)} 个点。")

    # 6. 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    print("点云对象创建成功。")
    return pcd

if __name__ == "__main__":
    # 检查图像文件是否存在
    for f_path in [COLOR_IMAGE_FILE, DEPTH_IMAGE_FILE, MASK_IMAGE_FILE]:
        if not os.path.exists(f_path):
            print(f"错误: 必需的图像文件未找到: {f_path}")
            print("请确保图像文件与脚本在同一目录，或已提供正确路径。")
            exit()

    # 创建点云
    roi_point_cloud = create_roi_point_cloud(
        COLOR_IMAGE_FILE,
        DEPTH_IMAGE_FILE,
        MASK_IMAGE_FILE,
        INTRINSICS,
        DEPTH_SCALE
    )

    if roi_point_cloud:
        print(f"\n点云包含 {len(roi_point_cloud.points)} 个点。")
        
        # 7. 可选：对点云进行下采样，如果点数过多
        # num_points_before = len(roi_point_cloud.points)
        # if num_points_before > 200000: # 例如，如果点数超过20万
        #     voxel_size = 0.005 # 体素大小 (米)，根据场景调整
        #     roi_point_cloud = roi_point_cloud.voxel_down_sample(voxel_size)
        #     print(f"点云已通过体素下采样至 {len(roi_point_cloud.points)} 个点 (体素大小: {voxel_size}m)。")

        # 8. 可选：移除离群点
        # cl, ind = roi_point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # roi_point_cloud = roi_point_cloud.select_by_index(ind)
        # print(f"移除离群点后，点云剩余 {len(roi_point_cloud.points)} 个点。")
        
        # 9. 保存点云到PLY文件
        try:
            o3d.io.write_point_cloud(OUTPUT_PLY_FILE, roi_point_cloud, write_ascii=False) # write_ascii=False更快，文件更小
            print(f"ROI彩色点云已保存到: {OUTPUT_PLY_FILE}")
        except Exception as e:
            print(f"保存点云到PLY文件失败: {e}")

        # 10. 可视化点云
        print("正在显示点云，按 'q' 关闭窗口...")
        o3d.visualization.draw_geometries(
            [roi_point_cloud],
            window_name="ROI Colored Point Cloud",
            width=800,
            height=600
        )
    else:
        print("未能生成点云。")