import torch

print(f"PyTorch Version: {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"CUDA Available (PyTorch detected GPU): {cuda_available}")

if cuda_available:
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

    # 检查 cuDNN 是否可用
    cudnn_available = torch.backends.cudnn.is_available()
    print(f"cuDNN Available (PyTorch detected cuDNN): {cudnn_available}")

    if cudnn_available:
        # 获取 PyTorch 检测到的 cuDNN 版本号
        cudnn_version = torch.backends.cudnn.version()
        print(f"cuDNN Version Detected by PyTorch: {cudnn_version}")
        print("\n结论: PyTorch 成功检测到 CUDA 和 cuDNN。安装很可能是正确的。")
    else:
        print("\n结论: PyTorch 检测到 CUDA GPU，但未能检测到 cuDNN。请检查：")
        print("  1. cuDNN 文件是否已正确复制到 CUDA Toolkit 目录中。")
        print("  2. CUDA Toolkit 的 bin 目录是否在系统 PATH 环境变量中。")
        print("  3. 安装的 cuDNN 版本是否与 CUDA 12.6 和 PyTorch 版本兼容。")
else:
     print("\n结论: PyTorch 未能检测到支持 CUDA 的 GPU。请检查：")
     print("  1. NVIDIA 驱动是否正确安装。")
     print("  2. CUDA Toolkit 是否正确安装。")
     print("  3. 安装的 PyTorch 版本是否是 GPU 版本。")

# 清理 CUDA 缓存（可选）
if cuda_available:
    torch.cuda.empty_cache()