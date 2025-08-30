# -*- coding: utf-8 -*-

import os
import math
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors

def generate_marker_sheet_pdf(
    output_path, 
    outer_diameter_mm=6, 
    border_thickness_mm=1.5, 
    spacing_mm=12
):
    """
    生成一张A4纸，上面布满黑边白心的环形光学标记点。

    参数:
    output_path (str): 生成的PDF文件的保存路径。
    outer_diameter_mm (float): 标记点的外径，单位毫米。
    border_thickness_mm (float): 标记点黑色边框的粗细，单位毫米。
    spacing_mm (float): 标记点中心之间的距离（间距），单位毫米。
    """
    # --- 1. 初始化参数 ---
    width, height = A4
    
    # 检查并创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    # --- 2. 创建PDF文档 ---
    c = canvas.Canvas(output_path, pagesize=A4)
    print(f"开始生成PDF文件: {output_path}")
    print(f"标记点外径: {outer_diameter_mm} mm, 边框粗细: {border_thickness_mm} mm, 间距: {spacing_mm} mm")

    # --- 3. 计算布局 ---
    # 定义页面边距，确保打印机不会切掉边缘的标记点
    margin_mm = 10
    
    # 计算可绘制区域
    drawable_width = width - (2 * margin_mm * mm)
    drawable_height = height - (2 * margin_mm * mm)

    # 计算可以容纳多少行和多少列的标记点
    num_cols = math.floor(drawable_width / (spacing_mm * mm))
    num_rows = math.floor(drawable_height / (spacing_mm * mm))
    
    print(f"将在页面上生成 {num_cols} 列 x {num_rows} 行 的标记点。")

    # 计算半径
    outer_radius_mm = outer_diameter_mm / 2.0
    inner_radius_mm = outer_radius_mm - border_thickness_mm
    
    if inner_radius_mm <= 0:
        print("错误：边框粗细必须小于外径的一半！")
        return

    # --- 4. 绘制所有标记点 ---
    # 使用嵌套循环遍历每个标记点的位置
    for row in range(num_rows):
        for col in range(num_cols):
            # 计算当前标记点的中心坐标 (x, y)
            # 我们从左上角开始绘制，所以y坐标需要从页面顶部开始计算
            x_center = (margin_mm * mm) + (col * spacing_mm * mm) + (spacing_mm * mm / 2)
            y_center = height - ((margin_mm * mm) + (row * spacing_mm * mm) + (spacing_mm * mm / 2))

            # 绘制环形：先画一个大的黑实心圆，再在上面覆盖一个小的白实心圆
            
            # a. 绘制外层的大黑圆
            c.setFillColor(colors.black)
            c.circle(x_center, y_center, outer_radius_mm * mm, stroke=0, fill=1)
            
            # b. 绘制内层的小白圆，形成环形效果
            c.setFillColor(colors.white)
            c.circle(x_center, y_center, inner_radius_mm * mm, stroke=0, fill=1)

    # --- 5. 保存PDF文件 ---
    c.save()
    print("-" * 30)
    print(f"成功！光学标记点PDF已保存至: {os.path.abspath(output_path)}")


if __name__ == '__main__':
    # --- 用户配置 ---
    # 在这里修改生成的PDF文件名和路径
    file_path = r"C:\Users\admin\Desktop\A4_Optical_Markers.pdf"
    
    # --- 在这里修改标记点的尺寸和布局 ---
    
    # 标记点的外圈直径（单位：毫米）
    marker_diameter = 6
    
    # 标记点黑色边框的厚度（单位：毫米）
    marker_border = 1.5
    
    # 标记点之间的中心距离，这个值要大于外径，以留出裁剪空间
    marker_spacing = 12
    
    # 调用函数生成PDF
    generate_marker_sheet_pdf(
        output_path=file_path,
        outer_diameter_mm=marker_diameter,
        border_thickness_mm=marker_border,
        spacing_mm=marker_spacing
    )
