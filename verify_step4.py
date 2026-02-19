import numpy as np
import matplotlib.pyplot as plt

def verify_npz(file_path):
    # 1. 加载文件
    data = np.load(file_path, allow_pickle=True)
    
    # 2. 提取数据
    x = data['x']  # Raw Sensor Image
    y = data['y']  # Ground Truth Stack (Free Water, Sebum)
    meta = data['meta'] if 'meta' in data else "No meta"
    snr_mode = data['snr_mode'] if 'snr_mode' in data else "No snr_mode"
    
    # 3. 打印物理维度与类型 (关键自检点)
    print("=== 物理数据审计报告 ===")
    print(f"X (Sensor)  : Shape={x.shape}, Dtype={x.dtype} (期望: 256x256, uint16)")
    print(f"Y (Truth)   : Shape={y.shape}, Dtype={y.dtype} (期望: 2x256x256, float32)")
    print(f"Meta Info   : {meta}")
    print(f"SNR Mode    : {snr_mode}")
    print(f"X 数据范围  : Min={x.min()}, Max={x.max()} (是否有过曝到 4095？)")
    print(f"Y0 数据范围 : Min={y[0].min():.3f}, Max={y[0].max():.3f} (Free Water浓度)")
    print(f"Y1 数据范围 : Min={y[1].min():.3f}, Max={y[1].max():.3f} (Sebum浓度)")
    
    # 4. 可视化检查
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # 画输入图 (使用对数缩放以便看清微弱的条纹)
    im0 = axs[0].imshow(np.log1p(x), cmap='gray')
    axs[0].set_title("Input X: Raw Sensor Image\n(Look for messy streaks)")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    
    # 画真值图 1: 游离水
    im1 = axs[1].imshow(y[0], cmap='Blues', vmin=0, vmax=1.0)
    axs[1].set_title("GT Y[0]: Free Water Map\n(Should be clean)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    
    # 画真值图 2: 皮脂
    im2 = axs[2].imshow(y[1], cmap='Oranges', vmin=0, vmax=1.0)
    axs[2].set_title("GT Y[1]: Sebum Map\n(Should be clean)")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

# 替换为你的文件路径运行
verify_npz('D:\spectualemu\output\datasets\sample_00005.npz')