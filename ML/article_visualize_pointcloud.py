import open3d as o3d
import os
import numpy as np

# --- 参数设置 (Parameters) ---

# 请将此路径修改为您电脑上 'bunny.ply' 文件的实际路径
# 使用 'r' 前缀可以防止反斜杠在路径中被错误地解析
ply_file_path = r"C:\Users\admin\Desktop\stanford bunny\stanford-bunny\source\bunny.ply"

# 点云颜色选择 (Point Cloud Color Selection)
# 推荐使用专业且清晰的单色。这里提供几个选项：
# 1. 专业蓝 (Professional Blue) - 默认选项，非常适合学术插图
point_color = [0.12, 0.47, 0.71] # RGB格式, 数值范围在 0 到 1 之间
# 2. 冷灰色 (Cool Gray) - 中性、不抢眼
# point_color = [0.5, 0.5, 0.6]
# 3. 科技感青色 (Technical Cyan)
# point_color = [0.0, 0.6, 0.6]

# 点的大小 (Point Size)
# 这个值可能需要根据您的屏幕分辨率和想要的视觉效果进行微调
point_size = 2.5

# 背景颜色 (Background Color)
# 白色 (White)
background_color = [1.0, 1.0, 1.0]

# --- 脚本主逻辑 (Main Script Logic) ---

def visualize_bunny_for_publication(filepath):
    """
    加载并可视化Stanford Bunny点云，用于生成出版级精美插图。
    """
    # 1. 检查文件是否存在
    if not os.path.exists(filepath):
        print(f"错误：文件未找到，请检查路径是否正确: {filepath}")
        return

    # 2. 加载点云
    print(f"正在从 {filepath} 加载点云...")
    try:
        pcd = o3d.io.read_point_cloud(filepath)
    except Exception as e:
        print(f"加载点云时出错: {e}")
        return

    if not pcd.has_points():
        print("错误：加载的点云中不包含任何点。")
        return

    # 3. 点云预处理，为高质量渲染做准备
    # 计算法线向量，这对于光照效果至关重要
    if not pcd.has_normals():
        print("点云没有法线，正在计算法线以获得更好的光照效果...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    # 将点云居中，使其位于坐标系原点，便于观察
    pcd.translate(-pcd.get_center())

    # 4. 设置点云颜色
    print(f"为点云设置颜色: {point_color}")
    pcd.paint_uniform_color(point_color)

    # --- 可选的颜色方案：基于法线的着色（可以更好地展示细节）---
    # 如果您想尝试，可以取消下面这行代码的注释。它会根据点的法线方向进行着色。
    # pcd.colors = o3d.utility.Vector3dVector(0.5 * (pcd.normals + 1.0))

    # 5. 可视化
    print("正在启动可视化窗口...")
    print("--- 操作指南 ---")
    print("  - 鼠标左键 + 拖动: 旋转视角")
    print("  - 鼠标滚轮: 缩放视图")
    print("  - 鼠标右键 + 拖动: 平移视图")
    print("  - 调整到您满意的角度后，可以直接使用系统截图工具（如QQ/微信截图，或Win+Shift+S）。")
    print("  - 或者，在窗口获得焦点时，按下 'S' 键，程序会自动保存一张名为 'bunny_screenshot.png' 的高分辨率截图。")
    print("  - 按下 'Q' 键或关闭窗口以退出程序。")

    # 使用 Visualizer 对象以获得更多控制权
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Stanford Bunny - Publication Figure", width=1920, height=1080)
    vis.add_geometry(pcd)

    # 获取渲染选项并进行精美化设置
    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color)
    opt.point_size = point_size
    opt.light_on = True
    opt.point_color_option = o3d.visualization.PointColorOption.Color
    opt.show_coordinate_frame = False # 不显示坐标系

    # 定义按键回调函数用于截图
    def capture_screenshot(visualizer):
        image_path = "bunny_screenshot.png"
        visualizer.capture_screen_image(image_path, do_render=True)
        print(f"截图已保存至: {image_path}")
        return False

    # 注册按键回调：按下'S'键时调用截图函数
    vis.register_key_callback(ord("S"), capture_screenshot)

    # 运行可视化
    vis.run()
    vis.destroy_window()
    print("可视化窗口已关闭。")

if __name__ == "__main__":
    visualize_bunny_for_publication(ply_file_path)
