import pyvista as pv
import numpy as np

def visualize_metallic_disk():
    """
    创建一个PyVista场景来可视化一个具有金属光泽的圆盘。
    """
    
    # 1. 定义圆盘的几何参数
    # 直径为60mm，所以半径为30mm
    disk_radius = 60.0
    # 厚度为2mm
    disk_height = 0.5
    
    # 2. 创建圆盘（圆柱体）网格
    # 我们将圆盘的中心放在坐标原点 (0, 0, 0)
    # PyVista的Cylinder默认沿z轴创建，所以我们设置方向为 (0, 0, 1)
    disk_mesh = pv.Cylinder(
        center=(0, 0, 0),
        direction=(0, 0, 1),
        radius=disk_radius,
        height=disk_height,
        resolution=100  # 使用较高的分辨率使圆周更平滑
    )
    
    # 3. 设置可视化窗口 (Plotter)
    # --- 修正：移除了 'lighting' 参数以兼容旧版PyVista ---
    # 物理渲染效果由 add_mesh 中的 pbr=True 启用
    plotter = pv.Plotter(window_size=[800, 800])
    plotter.set_background('#EAEAEA') # 设置一个浅灰色背景
    
    # 4. 添加网格到场景并设置材质属性
    # 要实现金属光泽，关键是设置 pbr=True，并调整 metallic 和 roughness
    plotter.add_mesh(
        disk_mesh,
        color='#C0C0C0',  # 银色
        pbr=True,          # 启用物理基础渲染 (Physically-Based Rendering)
        metallic=0.3,      # 金属度 (0到1)，越高越像金属
        roughness=0.05,     # 粗糙度 (0到1)，越低反射越清晰
        show_edges=False   # 不显示网格线，使表面更光滑
    )
    
    # 5. 设置灯光和相机
    plotter.enable_lightkit() # 使用一组预设的专业灯光
    plotter.camera_position = 'iso' # 设置为等距视角
    plotter.camera.zoom(1.5) # 稍微放大相机
    
    # 6. 添加标题并显示窗口
    plotter.add_text(
        "Reconstruction Base Disk (60mm x 2mm)",
        position="upper_left",
        font_size=12,
        color="black"
    )
    
    print("正在显示可视化窗口...请手动关闭窗口以结束程序。")
    plotter.show()

if __name__ == '__main__':
    visualize_metallic_disk()
