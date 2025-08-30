# -*- coding: utf-8 -*-

import os
import math  # 导入math库用于向上取整
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors

def generate_grid_pdf(output_path, grid_size_mm=10, line_width=0.5, line_color=colors.lightgrey):
    """
    生成一个铺满A4纸的灰色细线网格PDF文件。

    参数:
    output_path (str): 生成的PDF文件的保存路径，例如 'grid_A4.pdf'。
    grid_size_mm (int/float): 网格中每个方格的边长，单位为毫米。
                                默认值为10mm，这是一个常用的参考尺寸。
    line_width (float): 网格线的粗细，单位为磅(point)。默认0.1，非常细。
    line_color (Color): 网格线的颜色。默认是浅灰色(lightgrey)。
    """
    # --- 1. 初始化参数 ---
    # 获取A4纸的宽度和高度，单位为磅(point)
    width, height = A4
    
    # 检查输出路径的目录是否存在，如果不存在则创建
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    # --- 2. 创建PDF文档 ---
    # 使用A4页面尺寸创建一个新的PDF画布
    c = canvas.Canvas(output_path, pagesize=A4)
    print(f"开始生成PDF文件: {output_path}")
    print(f"页面尺寸 (A4): {width/mm:.2f} mm x {height/mm:.2f} mm")
    print(f"网格方格大小: {grid_size_mm} mm x {grid_size_mm} mm")

    # --- 3. 设置网格线样式 ---
    c.setStrokeColor(line_color)  # 设置线条颜色为浅灰色
    c.setLineWidth(line_width)      # 设置线条粗细

    # --- 4. 绘制网格线 ---
    # 使用向上取整来计算需要多少个格子才能完全覆盖页面
    num_cols = math.ceil(width / (grid_size_mm * mm))
    num_rows = math.ceil(height / (grid_size_mm * mm))

    # 准备x和y坐标列表，确保坐标范围能覆盖整个页面
    # range(num_cols + 1) 会生成从 0 到 num_cols 的所有整数，共 num_cols + 1 条垂直线
    x_list = [i * grid_size_mm * mm for i in range(num_cols + 1)]
    y_list = [i * grid_size_mm * mm for i in range(num_rows + 1)]
    
    # 使用grid()方法高效绘制所有网格线
    c.grid(x_list, y_list)
    print(f"已绘制 {len(x_list)} 条垂直线和 {len(y_list)} 条水平线以确保完全覆盖。")

    # --- 5. 保存PDF文件 ---
    c.save()
    print("-" * 30)
    print(f"成功！高清网格背景PDF已保存至: {os.path.abspath(output_path)}")


if __name__ == '__main__':
    # --- 用户配置 ---
    # 你可以在这里修改生成的PDF文件名和路径
    file_path = r"C:\Users\admin\Desktop\A4_Grid_10mm_full.pdf" # 更新文件名以作区分
    
    # 你可以在这里修改网格方格的大小（单位：毫米）
    # 常用尺寸有 5, 10, 15, 20 mm
    square_size = 10
    
    # 调用函数生成PDF
    generate_grid_pdf(output_path=file_path, grid_size_mm=square_size)
