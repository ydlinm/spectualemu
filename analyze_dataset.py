# 分析数据集统计信息
import numpy as np
import json

# 加载元数据
with open('output/datasets/dataset_metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

print("数据集统计:")
print(f"样本数量: {meta['num_samples']}")
print(f"图像尺寸: {meta['image_size']}×{meta['image_size']}")
print(f"波长数量: {meta['num_wavelengths']}")
print(f"波长范围: {meta['wavelength_range'][0]:.0f}-{meta['wavelength_range'][1]:.0f} nm")
print(f"生成时间: {meta['generation_time']}")

# 分析样本统计
samples = meta['samples']
water_means = [s['conc_water_mean'] for s in samples]
sebum_means = [s['conc_sebum_mean'] for s in samples]
melanin_means = [s['conc_melanin_mean'] for s in samples]
adu_means = [s['sensor_adu_mean'] for s in samples]
adu_maxs = [s['sensor_adu_max'] for s in samples]

print(f"\n浓度统计 (均值±标准差):")
print(f"  水浓度: {np.mean(water_means):.3f} ± {np.std(water_means):.3f}")
print(f"  皮脂浓度: {np.mean(sebum_means):.3f} ± {np.std(sebum_means):.3f}")
print(f"  黑色素浓度: {np.mean(melanin_means):.3f} ± {np.std(melanin_means):.3f}")

print(f"\n传感器信号统计:")
print(f"  ADU 均值: {np.mean(adu_means):.1f} ± {np.std(adu_means):.1f}")
print(f"  ADU 最大值: {np.mean(adu_maxs):.0f} ± {np.std(adu_maxs):.0f}")

# 分析单个样本
print(f"\n样本详细分析 (sample_00000.npz):")
d = np.load('output/datasets/sample_00000.npz')
print(f"  输入 (x): shape={d['x'].shape}, dtype={d['x'].dtype}")
print(f"  标签 (y): shape={d['y'].shape}, dtype={d['y'].dtype}")
print(f"  元数据: 水={d['meta'][0]:.3f}, 皮脂={d['meta'][1]:.3f}, 黑色素={d['meta'][2]:.3f}, 曝光={d['meta'][3]:.2f}")
print(f"  ADU 范围: [{d['x'].min()}, {d['x'].max()}]")
print(f"  水分布范围: [{d['y'][:,:,0].min():.4f}, {d['y'][:,:,0].max():.4f}]")
print(f"  皮脂分布范围: [{d['y'][:,:,1].min():.4f}, {d['y'][:,:,1].max():.4f}]")
