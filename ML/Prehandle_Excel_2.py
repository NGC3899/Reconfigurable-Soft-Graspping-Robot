import pandas as pd
import os

# --- 1. 配置参数 (请根据您的实际情况修改) ---

# 存放所有FEM文件夹的根目录路径
# 使用 'r' 前缀可以防止路径中的反斜杠被错误解析
ROOT_FOLDER = r"D:\MLP_Training_Dataset"

# 最终合并后的大Excel文件的保存路径和文件名
OUTPUT_FILENAME = r"D:\MLP_Training_Dataset\combined_fem_data.xlsx"

# 要处理的文件夹总数
TOTAL_FOLDERS = 22

# 每个文件夹的命名前缀
FOLDER_PREFIX = "FEM_"

# 每个文件夹内要读取的Excel文件名
DATA_FILENAME = "Displacement_Results.xlsx"

# 批处理大小：每处理 N 个文件就保存一次，以节省内存
# 如果单个Excel文件很大，可以调小此数值；如果文件较小，可以适当调大
BATCH_SIZE = 3

# --- 脚本主逻辑 (无需修改以下部分) ---

def merge_fem_excel_files():
    """
    按顺序、分批次地读取多个文件夹中的Excel文件，并将它们合并成一个大的Excel文件。
    """
    print("--- 开始合并Excel文件 ---")
    
    # 用于临时存放一个批次内读取的DataFrame
    batch_dataframes = []
    
    # 标记是否是第一次写入文件，第一次写入需要包含表头
    is_first_write = True

    # 确保输出文件的目录存在
    output_dir = os.path.dirname(OUTPUT_FILENAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"输出目录不存在，已创建: {output_dir}")

    # 按顺序循环处理每一个文件夹
    for i in range(1, TOTAL_FOLDERS + 1):
        folder_name = f"{FOLDER_PREFIX}{i}"
        file_path = os.path.join(ROOT_FOLDER, folder_name, DATA_FILENAME)
        
        print(f"正在处理: {file_path}")

        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"警告：文件不存在，跳过: {file_path}")
            continue

        try:
            # 读取当前Excel文件
            df = pd.read_excel(file_path)
            
            # 将读取到的DataFrame添加到批处理列表中
            batch_dataframes.append(df)
            
            # 检查是否达到批处理大小，或者这是否是最后一个文件
            if len(batch_dataframes) == BATCH_SIZE or i == TOTAL_FOLDERS:
                
                print(f"\n...达到批处理大小({len(batch_dataframes)}个文件)，准备写入文件...")
                
                # 将列表中的所有DataFrame合并为一个
                batch_to_write = pd.concat(batch_dataframes, ignore_index=True)
                
                if is_first_write:
                    # 如果是第一次写入，直接创建新文件并写入表头
                    batch_to_write.to_excel(OUTPUT_FILENAME, index=False, engine='openpyxl')
                    print(f"成功创建并写入第一个批次到: {OUTPUT_FILENAME}")
                    is_first_write = False
                else:
                    # 如果不是第一次，以追加模式写入，不包含表头
                    # 使用 ExcelWriter 的追加模式 'a'
                    with pd.ExcelWriter(OUTPUT_FILENAME, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                        # 获取工作簿和工作表对象，以确定从哪一行开始追加
                        # 这是一个更健壮的方法，可以避免覆盖数据
                        workbook = writer.book
                        sheet_name = workbook.sheetnames[0] # 假设只有一个工作表
                        start_row = workbook[sheet_name].max_row
                        
                        batch_to_write.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False, header=False)
                    print(f"成功追加一个批次到: {OUTPUT_FILENAME}")

                # 清空批处理列表，为下一批次释放内存
                batch_dataframes = []
                print("...内存已释放，继续处理下一个批次...\n")

        except Exception as e:
            print(f"错误：处理文件 {file_path} 时发生异常: {e}")
            continue
            
    print("--- 所有文件处理完毕，合并完成！ ---")
    print(f"最终的合并文件已保存在: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    merge_fem_excel_files()
