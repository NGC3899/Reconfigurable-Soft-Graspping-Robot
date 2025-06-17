# source: 1
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # <--- 新增：用于数据标准化
import matplotlib.pyplot as plt
import joblib # <--- 新增：用于保存 Scaler

# --- 1. 配置参数 ---
# source: 1
EXCEL_FILE_PATH = r'C:\Users\admin\Desktop\ML_Training_Data\summary_data.xlsx' # !!! 重要：替换为您的 Excel 文件路径 !!!
# source: 2
NODE_COUNT = 63                    # 您提到的节点数
OUTPUT_DIM = NODE_COUNT * 3        # 输出维度 = 63 个节点 * 3 个自由度 = 189
INPUT_DIM = 1                      # 输入维度 = 1 (气压 P)

# --- 超参数 (需要根据实际情况调整) ---
# source: 2, 3
HIDDEN_LAYER_1 = 128              # 第一个隐藏层神经元数量 (调整建议)
HIDDEN_LAYER_2 = 256              # 第二个隐藏层神经元数量 (调整建议)
HIDDEN_LAYER_3 = 128              # 第三个隐藏层神经元数量 (调整建议)
# source: 3
LEARNING_RATE = 0.0005            # 学习率 (调整建议, 可以尝试 0.001, 0.0001 等)
BATCH_SIZE = 16                   # 批处理大小 (调整建议, 如果总样本少可以更大, 如 32 或 64)
NUM_EPOCHS = 500                  # 训练轮数 (调整建议, 观察损失曲线决定)
# source: 4
VALIDATION_SPLIT = 0.2            # 划分多少数据作为验证集
RANDOM_SEED = 42                  # 随机种子，保证数据划分可复现

# --- 文件保存路径 ---
MODEL_SAVE_PATH = r'C:\Users\admin\Desktop\best_mlp_model.pth'
X_SCALER_SAVE_PATH = r'C:\Users\admin\Desktop\x_scaler.joblib'
Y_SCALER_SAVE_PATH = r'C:\Users\admin\Desktop\y_scaler.joblib'

# 自动选择设备 (GPU优先)
# source: 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"将使用的设备: {device}")


# --- 2. 数据加载与预处理 (增加标准化) ---
# source: 5
def load_and_preprocess_data(file_path, node_count, output_dim, val_split, random_seed, x_scaler_path, y_scaler_path):
    """
    从 Excel 加载数据, 按压力分组, 排序节点, 展平位移, 标准化数据, 划分训练/验证集, 转换为张量。

    Args:
        file_path (str): Excel 文件路径。
        node_count (int): 每个压力值对应的节点数。
        output_dim (int): 总输出维度。
        val_split (float): 验证集比例。
        random_seed (int): 随机种子。
        x_scaler_path (str): 输入数据标准化器的保存路径。
        y_scaler_path (str): 输出数据标准化器的保存路径。

    Returns:
        tuple: 包含训练集和验证集的 PyTorch 张量 (X_train, y_train, X_val, y_val)
               以及输入和输出的标准化器 (scaler_X, scaler_y)。
               如果出错则返回 (None, None, None, None, None, None)。
    """
    try:
        # source: 6
        df = pd.read_excel(file_path)
        print(f"成功加载 Excel 文件: {file_path}")
        print(f"文件包含的列: {df.columns.tolist()}")

        # --- 检查必需的列 ---
        required_cols = ['Pressure', 'NodeLabel', 'U1', 'U2', 'U3']
        # source: 6, 7, 8, 9
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            if 'NodeLabel' in missing_cols and 'Pressure' in df.columns:
                 print("警告: Excel 文件中缺少 'NodeLabel' 列。将尝试基于压力分组内的顺序自动生成节点 ID (1 到 N)。")
                 df['NodeLabel'] = df.groupby('Pressure').cumcount() + 1
                 required_cols = ['Pressure', 'NodeLabel', 'U1', 'U2', 'U3']
                 if not all(col in df.columns for col in required_cols):
                     missing_cols = [col for col in required_cols if col not in df.columns]
                     raise ValueError(f"Excel 文件仍缺少必需的列: {missing_cols}")
            else:
                raise ValueError(f"Excel 文件缺少必需的列: {missing_cols}")

    # source: 9, 10
    except FileNotFoundError:
        print(f"错误: Excel 文件未找到于 {file_path}")
        return None, None, None, None, None, None
    except ValueError as ve:
        print(f"数据列错误: {ve}")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"读取或处理 Excel 文件时出错: {e}")
        return None, None, None, None, None, None

    # source: 10
    grouped = df.groupby('Pressure')
    inputs_list = []
    outputs_list = []

    print("开始按压力值处理数据...")
    # source: 10, 11, 12, 13
    for pressure, group in grouped:
        if len(group) != node_count:
            print(f"警告: 压力值 {pressure} 对应的节点数量为 {len(group)}，预期为 {node_count}。将跳过此压力值。")
            continue
        group_sorted = group.sort_values(by='NodeLabel')
        displacements = group_sorted[['U1', 'U2', 'U3']].values.flatten()
        if displacements.shape[0] != output_dim:
             print(f"错误处理压力 {pressure}: 展平后的位移向量长度为 {displacements.shape[0]}, 预期为 {output_dim}。")
             continue
        inputs_list.append(pressure)
        outputs_list.append(displacements)

    # source: 13, 14
    if not inputs_list:
        print("错误: 没有成功提取任何有效的数据。请检查 Excel 文件格式或内容。")
        return None, None, None, None, None, None

    X = np.array(inputs_list).reshape(-1, INPUT_DIM)
    y = np.array(outputs_list)
    print(f"数据预处理完成。输入数据 X 形状: {X.shape}, 输出数据 y 形状: {y.shape}")

    # --- !!! 新增：数据标准化 !!! ---
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # 使用训练数据拟合 Scaler 并转换训练数据
    # 注意：我们先划分再标准化，只用训练集拟合Scaler，防止验证集信息泄露
    # source: 14
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=random_seed, shuffle=True
    )
    print(f"已划分为训练集 ({X_train.shape[0]} 个样本) 和验证集 ({X_val.shape[0]} 个样本)")

    # 标准化 X (输入)
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val) # 使用训练集的 scaler 转换验证集

    # 标准化 y (输出)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val) # 使用训练集的 scaler 转换验证集

    print("输入和输出数据已完成标准化。")

    # --- 保存 Scaler ---
    joblib.dump(scaler_X, x_scaler_path)
    joblib.dump(scaler_y, y_scaler_path)
    print(f"输入 Scaler 已保存至: {x_scaler_path}")
    print(f"输出 Scaler 已保存至: {y_scaler_path}")
    # ------------------------------------

    # 转换为 PyTorch 张量
    # source: 14, 15
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, scaler_X, scaler_y

# --- 3. 定义 PyTorch Dataset ---
# source: 15, 16
class FingerDisplacementDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- 4. 定义 MLP 模型 (可调整激活函数) ---
# source: 16, 17, 18
class MLPRegression(nn.Module):
    """一个简单的多层感知机用于回归"""
    def __init__(self, input_dim, output_dim, h1, h2, h3):
        super(MLPRegression, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.Tanh(),                # <--- 调整建议：尝试 Tanh 激活函数
            nn.Linear(h1, h2),
            nn.Tanh(),                # <--- 调整建议：尝试 Tanh 激活函数
            nn.Linear(h2, h3),
            nn.Tanh(),                # <--- 调整建议：尝试 Tanh 激活函数
            nn.Linear(h3, output_dim) # 输出层无激活函数
        )

    def forward(self, x):
        return self.network(x)

# --- 5. 训练模型的函数 ---
# source: 18, 19, 20, 21, 22, 23, 24
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path):
    """训练并验证模型"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print("\n开始训练...")
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'训练损失 (标准化的): {epoch_train_loss:.6f}, ' # 提示损失是基于标准化数据的
              f'验证损失 (标准化的): {epoch_val_loss:.6f}')   # 提示损失是基于标准化数据的

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'---> 验证损失降低，最佳模型已保存至 {model_save_path} (验证损失: {best_val_loss:.6f})')

    print('训练完成。')
    return train_losses, val_losses

# --- 6. 主执行流程 ---
if __name__ == '__main__':
    # (1) 加载、预处理和标准化数据
    # source: 24
    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, scaler_X, scaler_y = \
        load_and_preprocess_data(EXCEL_FILE_PATH, NODE_COUNT, OUTPUT_DIM,
                                 VALIDATION_SPLIT, RANDOM_SEED,
                                 X_SCALER_SAVE_PATH, Y_SCALER_SAVE_PATH)

    # source: 25
    if X_train_tensor is None:
        print("数据加载或预处理失败，程序退出。")
        exit()

    # (2) 创建 Dataset 和 DataLoader
    # source: 25
    train_dataset = FingerDisplacementDataset(X_train_tensor, y_train_tensor)
    val_dataset = FingerDisplacementDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"\n已创建 DataLoader。训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")
    # 打印数据集大小，帮助判断训练速度快的原因
    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

    # (3) 初始化 MLP 模型
    # source: 25, 26
    model = MLPRegression(INPUT_DIM, OUTPUT_DIM, HIDDEN_LAYER_1, HIDDEN_LAYER_2, HIDDEN_LAYER_3).to(device)
    print("\n模型结构:")
    print(model)

    # (4) 定义损失函数和优化器
    # source: 26
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # 可以考虑增加 weight_decay 正则化

    # (5) 训练模型
    # source: 26
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, MODEL_SAVE_PATH
    )

    # (6) (可选) 绘制损失曲线图
    # source: 26, 27
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失 (Training Loss - Scaled)')
    plt.plot(val_losses, label='验证损失 (Validation Loss - Scaled)')
    plt.title('模型训练过程中的损失变化 (基于标准化数据)')
    plt.xlabel('训练轮数 (Epochs)')
    plt.ylabel('均方误差损失 (MSE Loss - Scaled)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log') # <--- 新增：使用对数刻度观察早期损失下降
    plt.savefig('loss_curve.png')
    print("\n损失曲线图已保存为 loss_curve.png")

    # source: 27
    print(f"\n训练结束。最佳模型已保存在 '{MODEL_SAVE_PATH}'。")
    print(f"用于标准化的 Scaler 已保存为 '{X_SCALER_SAVE_PATH}' 和 '{Y_SCALER_SAVE_PATH}'。")
    print("\n后续使用模型进行预测时:")
    print("1. 加载模型: model = MLPRegression(...); model.load_state_dict(torch.load(MODEL_SAVE_PATH)); model.to(device); model.eval()")
    print(f"2. 加载 Scaler: scaler_X = joblib.load('{X_SCALER_SAVE_PATH}'); scaler_y = joblib.load('{Y_SCALER_SAVE_PATH}')")
    print("3. 准备新输入数据 (例如 new_pressure = np.array([[some_pressure_value]]))")
    print("4. 标准化新输入: new_pressure_scaled = scaler_X.transform(new_pressure)")
    print("5. 转换为 Tensor: input_tensor = torch.tensor(new_pressure_scaled, dtype=torch.float32).to(device)")
    print("6. 进行预测: with torch.no_grad(): predicted_scaled = model(input_tensor)")
    print("7. 反标准化预测结果: predicted_original_scale = scaler_y.inverse_transform(predicted_scaled.cpu().numpy())")
    print("8. predicted_original_scale 就是最终预测的位移值。")