# -*- coding: utf-8 -*-

import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import black, white

def generate_checkerboard_pdf(output_path, square_size_mm=15):
    """
    生成一个铺满A4纸的黑白棋盘格PDF文件。

    参数:
    output_path (str): 生成的PDF文件的保存路径，例如 'checkerboard_A4.pdf'。
    square_size_mm (int/float): 棋盘格中每个方块的边长，单位为毫米。
                                默认值为15mm，这是一个在A4纸上视觉效果和
                                实用性都很好的尺寸。
    """
    # --- 1. 初始化参数 ---
    # 获取A4纸的宽度和高度，单位从磅(point)转换为毫米(mm)
    width_mm, height_mm = A4[0]/mm, A4[1]/mm
    
    # 检查输出路径的目录是否存在，如果不存在则创建
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    # --- 2. 创建PDF文档 ---
    # 使用A4页面尺寸创建一个新的PDF画布
    c = canvas.Canvas(output_path, pagesize=A4)
    print(f"开始生成PDF文件: {output_path}")
    print(f"页面尺寸 (A4): {width_mm:.2f} mm x {height_mm:.2f} mm")
    print(f"棋盘格方块大小: {square_size_mm} mm x {square_size_mm} mm")

    # --- 3. 计算棋盘格的行数和列数 ---
    # 计算需要多少个方块才能完全覆盖A4纸的宽度和高度
    # 使用 int(x) + 1 来向上取整，确保覆盖整个页面
    num_cols = int(width_mm / square_size_mm) + 1
    num_rows = int(height_mm / square_size_mm) + 1
    print(f"计算出的棋盘格尺寸: {num_cols} 列 x {num_rows} 行")

    # --- 4. 绘制棋盘格 ---
    # 使用嵌套循环遍历每一个方块的位置
    for row in range(num_rows):
        for col in range(num_cols):
            # 计算当前方块的左下角坐标 (x, y)
            x = col * square_size_mm * mm
            y = row * square_size_mm * mm

            # 判断方块颜色，(行+列)为偶数时为一种颜色，奇数时为另一种
            # 这里我们让 (0,0) 位置为白色
            if (row + col) % 2 == 0:
                color = white
            else:
                color = black
            
            # 设置填充颜色
            c.setFillColor(color)

            # 绘制一个矩形方块
            # 参数: x, y, width, height, stroke=0 (无边框), fill=1 (填充)
            c.rect(x, y, square_size_mm * mm, square_size_mm * mm, stroke=0, fill=1)

    # --- 5. 保存PDF文件 ---
    c.save()
    print("-" * 30)
    print(f"成功！高清棋盘格PDF已保存至: {os.path.abspath(output_path)}")


if __name__ == '__main__':
    # --- 用户配置 ---
    # 你可以在这里修改生成的PDF文件名和路径
    file_path = r"C:\Users\admin\Desktop\A4_Checkerboard_15mm.pdf"
    
    # 你可以在这里修改棋盘格方块的大小（单位：毫米）
    # 常用尺寸有 10, 15, 20, 25 mm
    square_size = 15
    
    # 调用函数生成PDF
    generate_checkerboard_pdf(output_path=file_path, square_size_mm=square_size)

