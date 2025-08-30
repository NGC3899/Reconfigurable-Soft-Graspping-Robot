import numpy as np
import pandas as pd
import os

# --- 1. 参数设置 (您可以根据需要修改) ---

# 压力的最小值 (单位: Pa)
MIN_PRESSURE_PA = 0.0

# 压力的最大值 (单位: Pa)
MAX_PRESSURE_PA = 400000.0

# 需要采样的点的数量
NUM_SAMPLES = 20

# --- 新增：定义输出Excel文件的路径 ---
# 文件将保存在您之前提到的存放数据集的文件夹中
OUTPUT_DIR = r"D:\MLP_Training_Dataset"
OUTPUT_FILENAME = "sampled_pressures_for_testing.xlsx"
OUTPUT_EXCEL_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)


# --- 2. 生成压力值 ---

def generate_pressure_samples(min_p, max_p, num):
    """
    在指定的范围内生成均匀间隔的压力值。

    参数:
    - min_p (float): 压力最小值 (Pa)
    - max_p (float): 压力最大值 (Pa)
    - num (int): 采样的数量

    返回:
    - numpy.ndarray: 包含采样压力值的数组 (单位: Pa)
    """
    # np.linspace 会在指定的起始值和结束值之间，生成指定数量的、等间隔的数值。
    pressure_samples_pa = np.linspace(min_p, max_p, num)
    return pressure_samples_pa

# --- 3. 主执行程序 ---
if __name__ == "__main__":
    # 调用函数生成压力值
    sampled_pressures_pa = generate_pressure_samples(MIN_PRESSURE_PA, MAX_PRESSURE_PA, NUM_SAMPLES)

    # --- 4. 打印结果 ---
    print("--- 均匀采样的压力值 ---")
    
    # 打印帕斯卡 (Pa) 单位的结果
    print(f"\n单位: 帕斯卡 (Pa)，共 {len(sampled_pressures_pa)} 个点:")
    # 设置打印选项，使其更易读
    np.set_printoptions(precision=2, suppress=True)
    print(sampled_pressures_pa)

    # 将Pa转换为kPa并打印 (1 kPa = 1000 Pa)
    sampled_pressures_kpa = sampled_pressures_pa
    print(f"\n单位: 帕 (Pa)，共 {len(sampled_pressures_kpa)} 个点:")
    print(sampled_pressures_kpa)
    
    # --- 5. 新增：将结果保存到Excel文件 ---
    print("\n--- 正在将采样点写入Excel文件 ---")
    
    # 创建一个Pandas DataFrame，列名为 "Pressure (kPa)"
    df_to_save = pd.DataFrame({'Pressure (Pa)': sampled_pressures_kpa})

    # 确保输出目录存在，如果不存在则创建
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"输出目录不存在，已创建: {OUTPUT_DIR}")

    try:
        # 将DataFrame写入指定的Excel文件路径
        # index=False 表示不将DataFrame的索引写入文件
        df_to_save.to_excel(OUTPUT_EXCEL_PATH, index=False, engine='openpyxl')
        print(f"\n成功！采样点已写入到文件: {OUTPUT_EXCEL_PATH}")
    except PermissionError:
        print(f"\n错误：无法保存文件 {OUTPUT_EXCEL_PATH}。请检查文件是否被其他程序打开或您是否有写入权限。")
    except Exception as e:
        print(f"\n保存到Excel时发生未知错误: {e}")

