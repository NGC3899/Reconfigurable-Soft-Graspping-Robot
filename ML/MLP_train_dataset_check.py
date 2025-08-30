import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- 配置参数 (与您的训练脚本保持完全一致) ---
EXCEL_FILE_PATH = r'D:\MLP_Training_Dataset\combined_fem_data.xlsx'
NODE_COUNT = 63
OUTPUT_DIM = NODE_COUNT * 3
INPUT_DIM = 1
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42 # !!! 关键：这个值必须和训练脚本中的一样 !!!

# --- 加载和处理数据的简化流程 ---
try:
    df = pd.read_excel(EXCEL_FILE_PATH)
    print(f"成功加载总数据集: {EXCEL_FILE_PATH}")
except Exception as e:
    print(f"加载文件时出错: {e}")
    exit()

# 按压力值分组，提取输入(X)和输出(y)
grouped = df.groupby('Pressure')
inputs_list = []
outputs_list = []

for pressure, group in grouped:
    if len(group) == NODE_COUNT:
        inputs_list.append(pressure)
        # (我们这里只关心输入X，所以可以简化y的处理)
        outputs_list.append(np.zeros(OUTPUT_DIM)) 

X = np.array(inputs_list).reshape(-1, INPUT_DIM)
y = np.array(outputs_list)

# --- 使用与训练时完全相同的参数进行数据划分 ---
X_train, X_val, _, _ = train_test_split(
    X, y, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, shuffle=True
)

# --- 打印结果 ---
print("\n" + "="*50)
print("以下压力值 (Pa) 被划分到了【训练集】中：")
print("="*50)
# 将二维数组展平为一维并排序，方便查看
training_pressures = sorted(X_train.flatten())
print(training_pressures)
print(f"\n总计 {len(training_pressures)} 个训练样本。")


print("\n" + "="*50)
print("以下压力值 (Pa) 被划分到了【测试集】中：")
print("="*50)
validation_pressures = sorted(X_val.flatten())
print(validation_pressures)
print(f"\n总计 {len(validation_pressures)} 个测试样本。")
print("="*50)