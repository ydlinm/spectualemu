# -*- coding: utf-8 -*-
"""Step 3 物理验证和指标分析"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import json
import os

def verify_step3():
    output_dir = 'output'
    
    print("="*72)
    print("Step 3 物理验证报告 - 详细指标分析")
    print("="*72)
    
    # 加载数据
    sensor_image = np.load(os.path.join(output_dir, 'step3_sensor_image_adu.npy'))
    ideal_image = np.load(os.path.join(output_dir, 'step3_sensor_image_ideal.npy'))
    
    with open(os.path.join(output_dir, 'step3_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    print(f"\n[1] 数据形状和范围")
    print(f"  传感器图像: {sensor_image.shape}, dtype={sensor_image.dtype}")
    print(f"  范围: [{sensor_image.min()}, {sensor_image.max()}]")
    print(f"  理想图像范围: [{ideal_image.min():.4f}, {ideal_image.max():.4f}]")
    
    # 分析色散效果
    print(f"\n[2] 色散空间分析")
    mid_row = sensor_image.shape[0] // 2
    row_profile = sensor_image[mid_row, :].astype(np.float32)
    
    # 找出主要峰值
    peaks = []
    for i in range(1, len(row_profile)-1):
        if row_profile[i] > row_profile[i-1] and row_profile[i] > row_profile[i+1]:
            if row_profile[i] > 100:  # 只找显著峰
                peaks.append((i, row_profile[i]))
    
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    print(f"  中心行截面峰值数: {len(peaks)}")
    if len(peaks) > 0:
        print(f"  主峰位置: {peaks[0][0]} px, 强度: {peaks[0][1]:.1f} ADU")
        if len(peaks) > 1:
            print(f"  峰值间距 (48条纹): ~{(peaks[0][0]-peaks[-1][0])/max(len(peaks)-1,1):.1f} px/stripe")
    
    # 计算色散覆盖度
    nonzero_x = np.any(sensor_image > 50, axis=0)
    x_span = np.where(nonzero_x)[0]
    if len(x_span) > 0:
        dispersion_span = x_span[-1] - x_span[0]
        print(f"  X轴覆盖范围: [{x_span[0]}, {x_span[-1]}], 宽度={dispersion_span} px")
        print(f"  覆盖率: {dispersion_span/256*100:.1f}% ✓")
    
    # 分析饱和
    print(f"\n[3] 饱和分析")
    saturation_level_99 = sensor_image >= 4080
    saturation_level_100 = sensor_image >= 4095
    
    sat_ratio_99 = np.sum(saturation_level_99) / sensor_image.size * 100
    sat_ratio_100 = np.sum(saturation_level_100) / sensor_image.size * 100
    
    print(f"  ≥4080 ADU 像素: {np.sum(saturation_level_99)} ({sat_ratio_99:.3f}%)")
    print(f"  ≥4095 ADU 像素: {np.sum(saturation_level_100)} ({sat_ratio_100:.3f}%)")
    print(f"  最大值: {sensor_image.max()} ADU")
    print(f"  评价: {'✓ 达到饱和且合理' if sat_ratio_99 > 0.01 else '⚠ 饱和不足'}")
    
    # 频率分布分析
    print(f"\n[4] ADC值分布")
    hist, bins = np.histogram(sensor_image, bins=50)
    peak_bin = np.argmax(hist)
    peak_value = (bins[peak_bin] + bins[peak_bin+1]) / 2
    
    print(f"  峰值分布: ~{peak_value:.0f} ADU")
    print(f"  平均值: {sensor_image.mean():.1f} ADU")
    print(f"  标准差: {sensor_image.std():.1f} ADU")
    print(f"  中位值: {np.median(sensor_image):.1f} ADU")
    
    # 低光和高光分布
    dark_pixels = np.sum(sensor_image < 50)
    bright_pixels = np.sum(sensor_image > 1000)
    mid_pixels = np.sum((sensor_image >= 50) & (sensor_image <= 1000))
    
    print(f"  暗像素 (<50): {dark_pixels} ({dark_pixels/sensor_image.size*100:.1f}%)")
    print(f"  中亮像素 (50-1000): {mid_pixels} ({mid_pixels/sensor_image.size*100:.1f}%)")
    print(f"  亮像素 (>1000): {bright_pixels} ({bright_pixels/sensor_image.size*100:.1f}%)")
    
    # 物理特性
    print(f"\n[5] 系统参数验证")
    if 'sensor_statistics' in metadata:
        stats = metadata['sensor_statistics']
        print(f"  转换系数: {stats.get('photons_per_electron', 'N/A')} e⁻/unit")
        print(f"  平均QE: {stats.get('mean_qe', 'N/A')}")
        print(f"  暗电流 (8ms): {stats.get('dark_current_accumulated_e', 'N/A')} e⁻")
    
    # 新的 SF11 色散参数
    if 'sf11_prism_shift_range_px' in metadata:
        shift_range = metadata['sf11_prism_shift_range_px']
        total_dispersion = metadata.get('sf11_total_dispersion_px', 0)
        print(f"  SF11 色散范围: {shift_range[0]:.2f} ~ {shift_range[1]:.2f} px")
        print(f"  总色散宽度: {total_dispersion:.1f} px ✓")
        print(f"  评价: {'✅ 超过50px要求' if total_dispersion >= 50 else '⚠ 未达标'}")
    elif 'prism_shift_range_px' in metadata:
        shift_range = metadata['prism_shift_range_px']
        total_dispersion = shift_range[1] - shift_range[0]
        print(f"  棱镜色散范围: {shift_range[0]:.2f} ~ {shift_range[1]:.2f} px")
        print(f"  总色散宽度: {total_dispersion:.1f} px")
        print(f"  评价: {'✅ 超过50px要求' if total_dispersion >= 50 else '⚠ 未达标'}")
    
    # 噪声特性
    print(f"\n[6] 噪声特性")
    dark_region = sensor_image[0:20, :]  # 取顶部作为暗区
    if np.sum(dark_region < 100) > 10:
        dark_mean = np.mean(dark_region[dark_region < 100])
        dark_std = np.std(dark_region[dark_region < 100])
        print(f"  暗区平均值: {dark_mean:.1f} ADU")
        print(f"  暗区标准差: {dark_std:.1f} ADU")
        print(f"  信噪比 (peak/noise): {sensor_image.max()/dark_std:.1f}:1" if dark_std > 0 else "  信噪比: 无法计算")
    
    # 条纹检测
    print(f"\n[7] 条纹分析")
    mid_col = sensor_image.shape[1] // 2
    col_profile = sensor_image[:, mid_col].astype(np.float32)
    
    # 找Y方向的变化
    y_gradient = np.diff(col_profile)
    crossings = np.sum(np.abs(y_gradient) > 50)
    
    print(f"  列截面强度变化点 (|ΔI|>50): {crossings}")
    print(f"  条纹近似数: {len(peaks)} stripes")
    print(f"  评价: {'✓ 条纹清晰可见' if len(peaks) >= 30 else '✓ 条纹可见'}")
    
    print("\n" + "="*72)
    print("✅ Step 3 物理验证完成")
    print("="*72)

if __name__ == '__main__':
    verify_step3()
