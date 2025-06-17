# -*- coding: utf-8 -*-

import pandas as pd
import os # 用于检查文件是否存在

# --- 配置区域 ---
# !!! 请将下面的文件名替换为您实际的文件名 !!!
INPUT_FILE_1 = r'C:\Users\admin\Desktop\ML_Training_Data\Displacement_Results_1.xlsx'    # 第一个输入 Excel 文件名
INPUT_FILE_2 = r'C:\Users\admin\Desktop\ML_Training_Data\Displacement_Results_2.xlsx'    # 第二个输入 Excel 文件名
OUTPUT_FILE = r'C:\Users\admin\Desktop\ML_Training_Data\summary_data.xlsx' # 输出的汇总 Excel 文件名
# ----------------

def merge_excel_files(file1, file2, output_file):
    """
    合并两个 Excel 文件。
    将 file1 的全部内容复制到 output_file。
    将 file2 的内容（除第一行表头外）追加到 output_file 后面。

    Args:
        file1 (str): 第一个输入 Excel 文件的路径。
        file2 (str): 第二个输入 Excel 文件的路径。
        output_file (str): 输出汇总 Excel 文件的路径。
    """
    print(f"开始合并 Excel 文件...")
    print(f"文件1 (包含表头): {file1}")
    print(f"文件2 (追加数据): {file2}")
    print(f"输出文件: {output_file}")

    # 1. 检查输入文件是否存在
    if not os.path.exists(file1):
        print(f"错误: 输入文件 '{file1}' 未找到。请检查文件名和路径。")
        return
    if not os.path.exists(file2):
        print(f"错误: 输入文件 '{file2}' 未找到。请检查文件名和路径。")
        return

    try:
        # 2. 读取第一个 Excel 文件 (pandas 默认将第一行作为表头)
        print(f"正在读取 '{file1}'...")
        df1 = pd.read_excel(file1)
        print(f"读取 '{file1}' 完成，包含 {len(df1)} 行数据。")

        # 3. 读取第二个 Excel 文件 (同样默认第一行为表头)
        print(f"正在读取 '{file2}'...")
        df2 = pd.read_excel(file2)
        print(f"读取 '{file2}' 完成，包含 {len(df2)} 行数据。")
        # 注意：此时 df2 这个 DataFrame 变量中存储的是 Excel 文件从第二行开始的数据，
        # 因为 pandas 已经自动将第一行识别为列标题 (存储在 df2.columns 中)。
        # 所以我们直接使用 df2 进行后续拼接即可，它本身就不包含文件中的第一行（标题行）作为数据。

        # 4. 合并两个 DataFrame
        print("正在合并数据...")
        # pd.concat 用于按行 (axis=0) 拼接 DataFrame
        # ignore_index=True 会重新生成一个从 0 开始的连续索引，而不是保留原来的索引
        df_summary = pd.concat([df1, df2], ignore_index=True, sort=False)
        # sort=False 保持原始列顺序（如果列名不完全匹配）
        print(f"数据合并完成，总计 {len(df_summary)} 行数据。")

        # 5. 将合并后的 DataFrame 写入新的 Excel 文件
        print(f"正在将结果写入 '{output_file}'...")
        # index=False 参数确保 DataFrame 的索引不会被写入到 Excel 文件中作为单独的一列
        df_summary.to_excel(output_file, index=False)
        print(f"成功！合并后的文件已保存为 '{output_file}'。")

    except FileNotFoundError:
        # 理论上不会进入这里，因为前面检查了，但作为保险
        print("错误：读取文件时发生错误，请再次确认文件路径。")
    except ImportError:
        # 如果缺少必要的库
        print("错误：需要安装 pandas 和 openpyxl (用于读写 .xlsx)。")
        print("请在终端运行: pip install pandas openpyxl")
    except Exception as e:
        # 捕获其他可能的错误
        print(f"处理过程中发生未知错误: {e}")

# --- 主程序入口 ---
if __name__ == "__main__":
    # 调用合并函数
    merge_excel_files(INPUT_FILE_1, INPUT_FILE_2, OUTPUT_FILE)