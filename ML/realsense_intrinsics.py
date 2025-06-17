import pyrealsense2 as rs
import numpy as np # 虽然此脚本不直接用numpy处理图像，但pyrealsense2常与numpy配合

def get_realsense_intrinsics_and_depth_scale():
    """
    连接到RealSense相机，获取并打印彩色相机的内参和深度缩放因子。
    """
    # 创建一个Pipeline对象，这是与RealSense设备交互的主要高级API
    pipeline = rs.pipeline()
    # 创建一个Config对象，用于指定所需的流和参数
    config = rs.config()

    # --- 配置要启用的流 ---
    # 为了获取特定配置下的内参，最好启用流。
    # 我们将启用颜色流和深度流。
    # 您可以根据您后续处理图像时使用的分辨率和帧率来配置。
    # 如果不指定，SDK会选择默认值。
    # 例如，使用640x480分辨率，30fps
    color_width = 640
    color_height = 480
    depth_width = 640 # 通常与彩色流分辨率一致或兼容
    depth_height = 480
    fps = 30

    print(f"尝试配置颜色流: {color_width}x{color_height} @ {fps}fps")
    config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
    print(f"尝试配置深度流: {depth_width}x{depth_height} @ {fps}fps")
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)

    intrinsics_dict = None
    depth_scale_val = None

    try:
        # 启动管线
        print("正在启动RealSense管线...")
        profile = pipeline.start(config)
        print("RealSense管线已启动。")

        # --- 获取彩色相机内参 ---
        # 我们通常将深度图对齐到彩色图，因此彩色相机的内参对点云生成很重要。
        color_stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        if color_stream_profile:
            intrinsics = color_stream_profile.get_intrinsics()
            intrinsics_dict = {
                "width": intrinsics.width,
                "height": intrinsics.height,
                "fx": intrinsics.fx,    # X轴焦距
                "fy": intrinsics.fy,    # Y轴焦距
                "cx": intrinsics.ppx,   # X轴主点 (principal point x)
                "cy": intrinsics.ppy,   # Y轴主点 (principal point y)
                "model": intrinsics.model.name, # 失真模型 (e.g., "BrownConrady", "None")
                "coeffs": intrinsics.coeffs     # 失真系数 [k1, k2, p1, p2, k3]
            }
            print("\n--- 彩色相机内参 ---")
            print(f"  图像宽度 (width): {intrinsics_dict['width']}")
            print(f"  图像高度 (height): {intrinsics_dict['height']}")
            print(f"  X轴焦距 (fx): {intrinsics_dict['fx']:.4f}")
            print(f"  Y轴焦距 (fy): {intrinsics_dict['fy']:.4f}")
            print(f"  X轴主点 (cx/ppx): {intrinsics_dict['cx']:.4f}")
            print(f"  Y轴主点 (cy/ppy): {intrinsics_dict['cy']:.4f}")
            print(f"  失真模型 (model): {intrinsics_dict['model']}")
            print(f"  失真系数 (coeffs): {intrinsics_dict['coeffs']}")
        else:
            print("错误: 未能获取彩色流配置信息。")

        # --- 获取深度缩放因子 ---
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor:
            depth_scale_val = depth_sensor.get_depth_scale()
            print("\n--- 深度缩放因子 ---")
            print(f"  Depth Scale: {depth_scale_val:.6f} (米/深度单位)")
            print(f"  (这意味着深度图中像素值乘以该因子后得到以米为单位的距离)")
        else:
            print("错误: 未能获取深度传感器。")

        # （可选）短暂地抓取几帧以确保相机完全初始化并应用了配置
        print("\n抓取几帧以确保配置生效...")
        for _ in range(10): # 抓取10帧
            pipeline.wait_for_frames()
        print("帧抓取完成。")


    except RuntimeError as e:
        print(f"RealSense运行时错误: {e}")
        print("请检查以下事项：")
        print("  1. RealSense相机是否已正确连接到USB 3.0端口？")
        print("  2. Intel RealSense SDK 2.0是否已正确安装？")
        print("  3. 是否有其他程序正在使用该相机？")
        print("  4. 相机是否支持所请求的分辨率和帧率？尝试降低分辨率/帧率。")
    except Exception as e:
        print(f"发生未知错误: {e}")
    finally:
        # 停止管线
        if 'pipeline' in locals() : # 检查pipeline是否已定义
            print("正在停止RealSense管线...")
            pipeline.stop()
            print("RealSense管线已停止。")
    
    return intrinsics_dict, depth_scale_val

if __name__ == "__main__":
    print("开始自动读取RealSense相机参数...")
    
    # 调用函数获取参数
    cam_intrinsics, cam_depth_scale = get_realsense_intrinsics_and_depth_scale()

    if cam_intrinsics and cam_depth_scale is not None:
        print("\n--- 参数获取成功 ---")
        print("请将以下参数用于您的点云重建脚本：")
        print("INTRINSICS = {")
        print(f"    \"width\": {cam_intrinsics['width']},")
        print(f"    \"height\": {cam_intrinsics['height']},")
        print(f"    \"fx\": {cam_intrinsics['fx']:.4f},") # 保留适当小数位数
        print(f"    \"fy\": {cam_intrinsics['fy']:.4f},")
        print(f"    \"cx\": {cam_intrinsics['cx']:.4f},")
        print(f"    \"cy\": {cam_intrinsics['cy']:.4f}")
        print("}")
        print(f"DEPTH_SCALE = {cam_depth_scale:.6f}")
        print("\n注意: 如果您的相机有显著的镜头失真，您可能还需要考虑使用失真系数(coeffs)来校正图像点，")
        print("或者在Open3D等库中创建PinholeCameraIntrinsic对象时提供这些系数。")
        print("对于许多RealSense应用，如果主要关注点云的几何形状而非精确的纹理映射，")
        print("有时可以忽略轻微的失真，或者SDK的对齐过程已部分处理。")
    else:
        print("\n未能成功获取所有相机参数。请检查之前的错误信息。")