import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os

# --- 1. 配置参数 (请根据您的实际情况修改) ---

# !!! 重要：指向您合并后的大Excel文件路径 !!!
EXCEL_FILE_PATH = r'D:\MLP_Training_Dataset\combined_fem_data.xlsx' 

# !!! 重要：定义新的模型和Scaler保存路径 !!!
MODEL_SAVE_DIR = r'D:\FEM_MLP_Model'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'mlp_model_v2.pth')
X_SCALER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'x_scaler_v2.joblib')
Y_SCALER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'y_scaler_v2.joblib')

# --- 模型和数据维度 (与之前保持一致) ---
NODE_COUNT = 63                    # 每个压力对应的节点数
OUTPUT_DIM = NODE_COUNT * 3        # 输出维度 = 63 个节点 * 3 个自由度 (U1, U2, U3) = 189
INPUT_DIM = 1                      # 输入维度 = 1 (气压 P)

# --- 超参数 (根据新的大数据集进行了调整建议) ---
HIDDEN_LAYER_1 = 128              # 第一个隐藏层神经元数量
HIDDEN_LAYER_2 = 256              # 第二个隐藏层神经元数量
HIDDEN_LAYER_3 = 128              # 第三个隐藏层神经元数量
LEARNING_RATE = 0.001             # 学习率 (可以尝试 0.0005, 0.0001 等)
BATCH_SIZE = 64                   # 批处理大小 (由于数据集增大，可以适当调大以加快训练)
NUM_EPOCHS = 800                  # 训练轮数 (数据集更大，可能需要更多轮次来充分收敛)
VALIDATION_SPLIT = 0.2            # 划分20%的数据作为验证集
RANDOM_SEED = 42                  # 随机种子，保证数据划分可复现

# 自动选择设备 (GPU优先)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"将使用的设备: {device}")


# --- 2. 数据加载与预处理函数 (与之前脚本逻辑一致) ---
def load_and_preprocess_data(file_path, node_count, output_dim, val_split, random_seed):
    """
    从大的Excel文件加载数据, 按压力分组, 排序节点, 展平位移, 标准化数据, 划分训练/验证集。
    """
    try:
        df = pd.read_excel(file_path)
        print(f"成功加载总数据集: {file_path}")
        print(f"总行数: {len(df)}")

        # 检查必需的列
        required_cols = ['Pressure', 'Node Label', 'U1', 'U2', 'U3']
        if not all(col in df.columns for col in required_cols):
            # 兼容您之前可能的文件格式
            if 'NodeLabel' in df.columns:
                df.rename(columns={'NodeLabel': 'Node Label'}, inplace=True)
            else:
                missing_cols = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Excel 文件缺少必需的列: {missing_cols}")

    except FileNotFoundError:
        print(f"错误: Excel 文件未找到于 {file_path}")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"读取或处理 Excel 文件时出错: {e}")
        return None, None, None, None, None, None

    # 按压力值分组处理数据
    grouped = df.groupby('Pressure')
    inputs_list = []
    outputs_list = []

    print("开始按压力值处理数据...")
    for pressure, group in grouped:
        if len(group) != node_count:
            print(f"警告: 压力值 {pressure:.4f} 对应的节点数量为 {len(group)}，预期为 {node_count}。将跳过此压力值。")
            continue
        
        # 按节点标签排序以保证数据顺序一致性
        group_sorted = group.sort_values(by='Node Label')
        
        # 将U1, U2, U3的位移数据展平为一个长向量
        displacements = group_sorted[['U1', 'U2', 'U3']].values.flatten()
        
        if displacements.shape[0] != output_dim:
             print(f"错误处理压力 {pressure:.4f}: 展平后的位移向量长度不匹配。")
             continue
        
        inputs_list.append(pressure)
        outputs_list.append(displacements)

    if not inputs_list:
        print("错误: 没有成功提取任何有效的数据。请检查 Excel 文件格式或内容。")
        return None, None, None, None, None, None

    # 转换为Numpy数组
    X = np.array(inputs_list).reshape(-1, INPUT_DIM)
    y = np.array(outputs_list)
    print(f"数据预处理完成。输入数据 X 形状: {X.shape}, 输出数据 y 形状: {y.shape}")

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=random_seed, shuffle=True
    )
    print(f"已划分为训练集 ({X_train.shape[0]} 个样本) 和验证集 ({X_val.shape[0]} 个样本)")

    # --- 数据标准化 ---
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # 使用训练数据拟合Scaler，并转换所有数据集
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    print("输入和输出数据已完成标准化。")

    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, scaler_X, scaler_y

# --- 3. 自定义PyTorch Dataset ---
class FingerDisplacementDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- 4. 定义MLP模型 (与之前脚本架构一致) ---
class MLPRegression(nn.Module):
    def __init__(self, input_dim, output_dim, h1, h2, h3):
        super(MLPRegression, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, h3),
            nn.Tanh(),
            nn.Linear(h3, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# --- 5. 训练与验证模型的函数 ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print("\n--- 开始训练模型 ---")
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

        # 验证阶段
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f'轮次 [{epoch+1}/{num_epochs}], '
              f'训练损失 (标准化后): {epoch_train_loss:.8f}, '
              f'验证损失 (标准化后): {epoch_val_loss:.8f}')

        # 保存验证损失最低的模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'---> 验证损失降低，最佳模型已保存至 {model_save_path}')

    print('--- 训练完成 ---')
    return train_losses, val_losses

# --- 6. 主执行流程 ---
if __name__ == '__main__':
    # 确保模型保存目录存在
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f"模型保存目录不存在，已创建: {MODEL_SAVE_DIR}")

    # (1) 加载并处理数据
    X_train, y_train, X_val, y_val, scaler_X, scaler_y = \
        load_and_preprocess_data(EXCEL_FILE_PATH, NODE_COUNT, OUTPUT_DIM, VALIDATION_SPLIT, RANDOM_SEED)

    if X_train is None:
        print("数据加载或预处理失败，程序退出。")
        exit()

    # (2) 创建Dataset和DataLoader
    train_dataset = FingerDisplacementDataset(X_train, y_train)
    val_dataset = FingerDisplacementDataset(X_val, y_val)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"\n已创建DataLoader。训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")

    # (3) 初始化模型
    model = MLPRegression(INPUT_DIM, OUTPUT_DIM, HIDDEN_LAYER_1, HIDDEN_LAYER_2, HIDDEN_LAYER_3).to(device)
    print("\n模型结构:")
    print(model)

    # (4) 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # (5) 训练模型
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, MODEL_SAVE_PATH
    )

    # (6) 保存标准化器 (Scaler)
    joblib.dump(scaler_X, X_SCALER_SAVE_PATH)
    joblib.dump(scaler_y, Y_SCALER_SAVE_PATH)
    print(f"输入数据标准化器已保存至: {X_SCALER_SAVE_PATH}")
    print(f"输出数据标准化器已保存至: {Y_SCALER_SAVE_PATH}")

    # (7) 绘制并保存损失曲线图
    plt.figure(figsize=(12, 7))
    plt.plot(train_losses, label='Training Loss (Scaled)')
    plt.plot(val_losses, label='Validation Loss (Scaled)')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE) Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log') # 使用对数刻度以便观察早期损失的快速下降
    loss_curve_path = os.path.join(MODEL_SAVE_DIR, 'loss_curve_v2.png')
    plt.savefig(loss_curve_path)
    print(f"\n损失曲线图已保存为: {loss_curve_path}")

    print(f"\n--- 任务结束 ---")
    print(f"最佳模型已保存在: '{MODEL_SAVE_PATH}'")
