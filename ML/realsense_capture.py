import pyrealsense2 as rs
import numpy as np
import cv2
import os   # 用于路径操作
import time # 用于计时和生成带时间戳的文件名

# --- 配置参数 ---
COLOR_WIDTH = 640
COLOR_HEIGHT = 480
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
FPS = 30

# --- 窗口名称 ---
WINDOW_COLOR = 'RealSense Color Stream'
WINDOW_DEPTH_ALIGNED_VIS = 'Aligned Depth Stream (Colormapped for Visualization)'

# --- 保存路径 ---
try:
    desktop_path_variant1 = os.path.join(os.path.expanduser("~"), "Desktop")
    desktop_path_variant2 = os.path.join(os.path.expanduser("~"), "桌面")

    if os.path.exists(desktop_path_variant1):
        SAVE_PATH = desktop_path_variant1
    elif os.path.exists(desktop_path_variant2):
        SAVE_PATH = desktop_path_variant2
    else:
        SAVE_PATH = "." 
        print(f"警告: 未找到桌面路径，图像将保存到当前目录: {os.getcwd()}")
    
    if not os.path.exists(SAVE_PATH) and SAVE_PATH != ".":
        os.makedirs(SAVE_PATH)
        print(f"创建保存路径: {SAVE_PATH}")
    elif SAVE_PATH != ".":
         print(f"图像将保存到: {SAVE_PATH}")

except Exception as e:
    SAVE_PATH = "."
    print(f"获取桌面路径时出错: {e}。图像将保存到当前目录: {os.getcwd()}")


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)

    print("正在尝试启动RealSense管线...")
    try:
        profile = pipeline.start(config)
        print("RealSense管线已启动。")

        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor:
            depth_scale = depth_sensor.get_depth_scale()
            print(f"深度比例因子: {depth_scale:.4f} (米/深度单位)")
        else:
            depth_scale = 0.001

        align_to = rs.stream.color
        align = rs.align(align_to)

        cv2.namedWindow(WINDOW_COLOR, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(WINDOW_DEPTH_ALIGNED_VIS, cv2.WINDOW_AUTOSIZE)

        print(f"开始读取视频流... 将在约5秒后自动保存一组图像到 '{SAVE_PATH}'。")
        print("按 'q' 或 'ESC' 键退出。按 's' 键可手动保存。")
        
        start_time = time.time() # 记录视频流开始的时间
        auto_save_done = False   # 自动保存操作是否已完成的标志
        frame_counter = 0        # 用于手动保存时区分文件名

        while True:
            frameset = pipeline.wait_for_frames()
            if not frameset:
                continue

            aligned_frameset = align.process(frameset)
            if not aligned_frameset:
                print("未能对齐帧组，跳过此帧。")
                continue

            aligned_depth_frame = aligned_frameset.get_depth_frame()
            color_frame = aligned_frameset.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                print("未能获取到对齐后的深度帧或颜色帧，跳过此帧。")
                continue
            
            aligned_depth_image_raw = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # --- 可视化部分 ---
            depth_colormap_for_display = cv2.applyColorMap(
                cv2.convertScaleAbs(aligned_depth_image_raw, alpha=0.03),
                cv2.COLORMAP_JET
            )
            cv2.imshow(WINDOW_COLOR, color_image)
            cv2.imshow(WINDOW_DEPTH_ALIGNED_VIS, depth_colormap_for_display)

            # --- 自动保存逻辑 ---
            current_time = time.time()
            if not auto_save_done and (current_time - start_time) >= 5.0:
                print("\n--- 达到5秒，执行自动保存 ---")
                ts_filename_part = time.strftime("%Y%m%d_%H%M%S")
                
                auto_color_filename = os.path.join(SAVE_PATH, f"auto_aligned_color_{ts_filename_part}.png")
                auto_depth_filename = os.path.join(SAVE_PATH, f"auto_aligned_depth_raw_{ts_filename_part}.png")

                cv2.imwrite(auto_color_filename, color_image)
                print(f"  已自动保存对齐的彩色图像到: {auto_color_filename}")

                cv2.imwrite(auto_depth_filename, aligned_depth_image_raw)
                print(f"  已自动保存对齐的16位深度图像到: {auto_depth_filename}")
                print(f"  深度图像数据类型: {aligned_depth_image_raw.dtype}, 范围: [{np.min(aligned_depth_image_raw)}-{np.max(aligned_depth_image_raw)}]")
                
                auto_save_done = True # 标记自动保存已完成，不再重复执行
                print("自动保存完成！将继续显示视频流。")
                # 如果希望在自动保存后立即退出，可以取消下面的注释：
                # print("自动保存后退出程序。")
                # break 

            # --- 按键处理 ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # q 或 ESC
                print("检测到退出键，正在关闭...")
                break
            elif key == ord('s'): # 手动保存
                frame_counter += 1
                ts_filename_part_manual = time.strftime("%Y%m%d_%H%M%S")
                
                manual_color_filename = os.path.join(SAVE_PATH, f"manual_aligned_color_{ts_filename_part_manual}_{frame_counter}.png")
                manual_depth_filename = os.path.join(SAVE_PATH, f"manual_aligned_depth_raw_{ts_filename_part_manual}_{frame_counter}.png")
                
                print(f"\n--- 手动保存图像 (第 {frame_counter} 组) ---")
                cv2.imwrite(manual_color_filename, color_image)
                print(f"  已手动保存对齐的彩色图像到: {manual_color_filename}")

                cv2.imwrite(manual_depth_filename, aligned_depth_image_raw)
                print(f"  已手动保存对齐的16位深度图像到: {manual_depth_filename}")
                print("手动保存完成！")

    except Exception as e:
        print(f"发生错误: {e}")

    finally:
        print("正在停止RealSense管线...")
        if 'pipeline' in locals() and pipeline:
            try:
                pipeline.stop()
                print("RealSense管线已停止。")
            except RuntimeError as e:
                print(f"停止管线时出错: {e}")
        cv2.destroyAllWindows()
        print("OpenCV窗口已关闭。")

if __name__ == '__main__':
    main()