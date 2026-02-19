# -*- coding: utf-8 -*-
"""
Step 4: Data Factory (Batch Generation Pipeline)
数据工厂：批量生成合成训练数据集
"""

import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate, shift
from PIL import Image
import os
from datetime import datetime
import json
from tqdm import tqdm

# ================================
# 1. 加载 SpectralAssets (复用 step3)
# ================================

class SpectralAssets:
    """光谱资产容器 (Step 1 标准化数据)"""
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.wavelengths = df['wavelength_nm'].values
        self.halogen_spectrum = df['halogen_spectrum'].values
        self.sensor_qe = df['sensor_qe'].values
        self.water_mu_a = df['water_mu_a'].values
        self.lipid_mu_a = df['lipid_mu_a'].values
        self.sebum_mu_a = df['sebum_mu_a'].values
        self.scatter_mus = df['scatter_mus'].values
        self.melanin_mu_a = df['melanin_mu_a'].values
        self.prism_shift_px = df['prism_shift_px'].values
        
        self.num_wavelengths = len(self.wavelengths)
        print(f"✓ 加载光谱资产: {self.num_wavelengths} 波长")


# ================================
# 2. Perlin Noise 生成器 (纹理生成)
# ================================

class PerlinNoiseGenerator:
    """Perlin 噪声生成器 (用于空间纹理增强)"""
    
    @staticmethod
    def generate_perlin_2d(shape, scale=50, octaves=4, persistence=0.5, seed=None):
        """
        生成 2D Perlin 噪声
        Args:
            shape: (H, W) 输出尺寸
            scale: 噪声频率 (越小越平滑)
            octaves: 叠加层数 (越多细节越丰富)
            persistence: 振幅衰减系数
            seed: 随机种子
        Returns:
            noise: 归一化到 [0, 1] 的 2D 数组
        """
        if seed is not None:
            np.random.seed(seed)
        
        H, W = shape
        noise = np.zeros((H, W))
        
        for octave in range(octaves):
            freq = 2 ** octave
            amp = persistence ** octave
            
            # 生成随机梯度网格
            grid_h = H // (scale // freq) + 2
            grid_w = W // (scale // freq) + 2
            gradients = np.random.randn(grid_h, grid_w, 2)
            
            # 简化版 Perlin（使用高斯平滑模拟）
            layer = np.random.randn(H, W)
            layer = gaussian_filter(layer, sigma=scale / freq)
            noise += amp * layer
        
        # 归一化到 [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)
        return noise


# ================================
# 3. Data Factory (核心类)
# ================================

class DataFactory:
    """
    数据工厂：批量生成合成训练数据
    """
    
    def __init__(self, assets_path, image_size=256, output_dir='output/datasets'):
        """
        Args:
            assets_path: step1 的标准化数据路径 (CSV)
            image_size: 图像分辨率 (像素)
            output_dir: 数据集输出目录
        """
        self.assets = SpectralAssets(assets_path)
        self.image_size = image_size
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载纹理 (如果不存在，生成随机纹理)
        texture_path = 'assets/skin_texture.png'
        if os.path.exists(texture_path):
            texture = np.array(Image.open(texture_path).convert('L'))
            # 调整到目标尺寸
            texture = np.array(Image.fromarray(texture).resize((image_size, image_size)))
        else:
            print(f"⚠ 纹理文件不存在，生成随机纹理")
            texture = PerlinNoiseGenerator.generate_perlin_2d(
                (image_size, image_size), scale=20, octaves=6
            )
            texture = (texture * 255).astype(np.uint8)
        
        self.base_texture = texture / 255.0  # 归一化到 [0, 1]
        print(f"✓ 数据工厂初始化完成 (图像尺寸: {image_size}×{image_size})")
    
    def augment_texture(self, seed=None):
        """
        纹理增强：随机旋转、翻转、平移
        Returns:
            augmented_texture: (H, W) 归一化纹理
        """
        if seed is not None:
            np.random.seed(seed)
        
        texture = self.base_texture.copy()
        
        # 随机旋转
        angle = np.random.uniform(-180, 180)
        texture = rotate(texture, angle, reshape=False, mode='wrap')
        
        # 随机翻转
        if np.random.rand() > 0.5:
            texture = np.fliplr(texture)
        if np.random.rand() > 0.5:
            texture = np.flipud(texture)
        
        # 随机平移
        shift_x = np.random.randint(-20, 20)
        shift_y = np.random.randint(-20, 20)
        texture = shift(texture, (shift_y, shift_x), mode='wrap')
        
        return texture
    
    def generate_concentration_maps(self, conc_water, conc_sebum, conc_melanin, seed=None):
        """
        生成浓度空间分布图
        Args:
            conc_water: 水浓度基准值
            conc_sebum: 皮脂浓度基准值
            conc_melanin: 黑色素浓度基准值
            seed: 随机种子
        Returns:
            map_water: (H, W) 水浓度分布
            map_sebum: (H, W) 皮脂浓度分布
            map_melanin: (H, W) 黑色素浓度分布
        """
        if seed is not None:
            np.random.seed(seed)
        
        H, W = self.image_size, self.image_size
        
        # 水分布：低频 Perlin 噪声 (水合作用缓慢变化)
        perlin_water = PerlinNoiseGenerator.generate_perlin_2d(
            (H, W), scale=80, octaves=3, seed=seed
        )
        # 调制范围: ±20% 变化
        map_water = conc_water * (0.8 + 0.4 * perlin_water)
        
        # 皮脂分布：高频纹理 (皮脂跟随毛孔/纹理)
        augmented_texture = self.augment_texture(seed=seed + 1 if seed else None)
        # 调制范围: ±30% 变化
        map_sebum = conc_sebum * (0.7 + 0.6 * augmented_texture)
        
        # 黑色素分布：中频噪声 (色素沉着)
        perlin_melanin = PerlinNoiseGenerator.generate_perlin_2d(
            (H, W), scale=50, octaves=4, seed=seed + 2 if seed else None
        )
        map_melanin = conc_melanin * (0.9 + 0.2 * perlin_melanin)
        
        # 物理约束：总浓度 <= 1.0
        total_map = map_water + map_sebum + map_melanin
        overflow_mask = total_map > 1.0
        if np.any(overflow_mask):
            # 归一化到 1.0
            scale_factor = 1.0 / total_map[overflow_mask]
            map_water[overflow_mask] *= scale_factor
            map_sebum[overflow_mask] *= scale_factor
            map_melanin[overflow_mask] *= scale_factor
        
        return map_water, map_sebum, map_melanin
    
    def compute_effective_mu_a(self, map_water, map_sebum, map_melanin):
        """
        计算有效吸收系数 μ_a(λ, x, y)
        Args:
            map_water: (H, W) 水浓度分布
            map_sebum: (H, W) 皮脂浓度分布
            map_melanin: (H, W) 黑色素浓度分布
        Returns:
            mu_a_map: (H, W, num_wavelengths) 有效吸收系数
        """
        H, W = map_water.shape
        num_wl = self.assets.num_wavelengths
        
        # 扩展波长维度
        mu_a_water_3d = self.assets.water_mu_a[None, None, :]  # (1, 1, num_wl)
        mu_a_sebum_3d = self.assets.sebum_mu_a[None, None, :]
        mu_a_melanin_3d = self.assets.melanin_mu_a[None, None, :]
        
        # 浓度图扩展维度
        map_water_3d = map_water[:, :, None]  # (H, W, 1)
        map_sebum_3d = map_sebum[:, :, None]
        map_melanin_3d = map_melanin[:, :, None]
        
        # 线性混合
        mu_a_map = (map_water_3d * mu_a_water_3d +
                    map_sebum_3d * mu_a_sebum_3d +
                    map_melanin_3d * mu_a_melanin_3d)
        
        return mu_a_map  # (H, W, num_wl)
    
    def compute_wavelength_dependent_pathlength(self):
        """
        计算波长依赖的有效光程 d_eff(λ)
        物理原理：短波散射强，穿透浅；长波穿透深
        
        Returns:
            d_eff_array: (num_wl,) 有效光程数组 (mm)
        """
        # 线性模型: d_eff(λ) = d_min + (d_max - d_min) * (λ - λ_min) / (λ_max - λ_min)
        wl = self.assets.wavelengths
        wl_min, wl_max = wl[0], wl[-1]
        d_min, d_max = 0.4, 0.8  # 900nm: 0.4mm, 1700nm: 0.8mm
        
        d_eff_array = d_min + (d_max - d_min) * (wl - wl_min) / (wl_max - wl_min)
        return d_eff_array
    
    def forward_simulation(self, mu_a_map, exposure_factor=1.0, snr_mode='high'):
        """
        正向模拟：Step2 (光场) + Step3 (棱镜+传感器)
        改进版：波长依赖光程 + 提高光强 + SNR 多样性
        
        Args:
            mu_a_map: (H, W, num_wl) 有效吸收系数
            exposure_factor: 曝光时间因子 (随机化照明强度)
            snr_mode: 'high' (实验室), 'medium' (正常), 'low' (手持抖动)
        Returns:
            sensor_image: (H, W) 传感器 ADU 图像
            hypercube: (H, W, num_wl) 场景超立方体
        """
        H, W, num_wl = mu_a_map.shape
        
        # Step 2: 生成场景超立方体 (波长依赖光传输)
        # 反射率 R(λ) = exp(-μ_a * d_eff(λ))
        d_eff_array = self.compute_wavelength_dependent_pathlength()  # (num_wl,)
        d_eff_3d = d_eff_array[None, None, :]  # (1, 1, num_wl)
        reflectance = np.exp(-mu_a_map * d_eff_3d)
        
        # 照明 × 反射率 × 基准光强增益 (提高200倍到传感器最佳工作区)
        BASE_GAIN = 200.0  # 关键修正：将信号提升到 10k-30k ADU 区间
        illumination = self.assets.halogen_spectrum[None, None, :]  # (1, 1, num_wl)
        hypercube = reflectance * illumination * exposure_factor * BASE_GAIN
        
        # Step 3: 传感器响应 + 色散
        sensor_qe = self.assets.sensor_qe[None, None, :]  # (1, 1, num_wl)
        photon_flux = hypercube * sensor_qe
        
        # 色散模拟 (简化为波长积分 + 水平偏移)
        sensor_image = np.zeros((H, W), dtype=np.float32)
        prism_shifts = self.assets.prism_shift_px
        
        for wl_idx in range(num_wl):
            shift_px = prism_shifts[wl_idx]
            # 水平偏移
            shifted = shift(photon_flux[:, :, wl_idx], (0, shift_px), mode='constant', cval=0)
            sensor_image += shifted
        
        # 噪声模型：根据 SNR 模式调整噪声水平
        if snr_mode == 'high':  # 实验室理想环境
            read_noise_std = 3.0
            dark_current = 2.0
        elif snr_mode == 'medium':  # 正常环境
            read_noise_std = 5.0
            dark_current = 5.0
        else:  # 'low' - 手持抖动/短曝光
            read_noise_std = 8.0
            dark_current = 10.0
        
        # 泊松噪声 + 读出噪声 + 暗电流
        sensor_image = np.clip(sensor_image, 0, None)
        photon_noise = np.random.poisson(sensor_image + 1e-6)
        read_noise = np.random.normal(0, read_noise_std, (H, W))
        dark_current_noise = np.random.poisson(dark_current, (H, W))
        sensor_image_adu = photon_noise + read_noise + dark_current_noise
        
        # 量化到 uint16
        sensor_image_adu = np.clip(sensor_image_adu, 0, 65535).astype(np.uint16)
        
        return sensor_image_adu, hypercube
    
    def generate_batch(self, num_samples=100, seed_offset=0):
        """
        批量生成数据集
        Args:
            num_samples: 生成样本数量
            seed_offset: 随机种子偏移
        """
        print(f"\n🏭 启动数据工厂：生成 {num_samples} 个训练样本...")
        
        metadata_list = []
        
        for i in tqdm(range(num_samples), desc="生成进度"):
            sample_id = i + seed_offset
            seed = sample_id + 42
            np.random.seed(seed)
            
            # 1. 域随机化：生成随机浓度
            conc_water = np.random.uniform(0.05, 0.40)  # 5%-40% 水
            conc_sebum = np.random.uniform(0.05, 0.35)  # 5%-35% 皮脂
            conc_melanin = np.random.uniform(0.01, 0.10)  # 1%-10% 黑色素
            
            # 确保总浓度 <= 1.0
            total_conc = conc_water + conc_sebum + conc_melanin
            if total_conc > 1.0:
                scale = 0.95 / total_conc
                conc_water *= scale
                conc_sebum *= scale
                conc_melanin *= scale
            
            # 2. 生成空间浓度图
            map_water, map_sebum, map_melanin = self.generate_concentration_maps(
                conc_water, conc_sebum, conc_melanin, seed=seed
            )
            
            # 3. 物理混合：计算有效 μ_a
            mu_a_map = self.compute_effective_mu_a(map_water, map_sebum, map_melanin)
            
            # 4. 随机曝光 + SNR 多样性
            exposure_factor = np.random.uniform(0.8, 1.2)
            
            # SNR 模式随机选择 (60% high, 30% medium, 10% low)
            snr_rand = np.random.rand()
            if snr_rand < 0.6:
                snr_mode = 'high'
            elif snr_rand < 0.9:
                snr_mode = 'medium'
            else:
                snr_mode = 'low'
            
            # 5. 正向模拟
            sensor_image_adu, hypercube = self.forward_simulation(mu_a_map, exposure_factor, snr_mode)
            
            # 6. 保存数据
            save_path = os.path.join(self.output_dir, f"sample_{sample_id:05d}.npz")
            np.savez_compressed(
                save_path,
                x=sensor_image_adu,  # 输入特征 (uint16)
                y=np.stack([map_water, map_sebum], axis=-1),  # 标签 (H, W, 2)
                meta=np.array([conc_water, conc_sebum, conc_melanin, exposure_factor])
            )
            
            # 记录元数据
            metadata_list.append({
                'sample_id': sample_id,
                'conc_water_mean': float(map_water.mean()),
                'conc_sebum_mean': float(map_sebum.mean()),
                'conc_melanin_mean': float(map_melanin.mean()),
                'exposure_factor': float(exposure_factor),
                'snr_mode': snr_mode,
                'sensor_adu_mean': float(sensor_image_adu.mean()),
                'sensor_adu_max': int(sensor_image_adu.max())
            })
        
        # 保存全局元数据
        meta_path = os.path.join(self.output_dir, 'dataset_metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'num_samples': num_samples,
                'image_size': self.image_size,
                'num_wavelengths': self.assets.num_wavelengths,
                'wavelength_range': [float(self.assets.wavelengths[0]), float(self.assets.wavelengths[-1])],
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'samples': metadata_list
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 数据集生成完成！保存路径: {self.output_dir}")
        print(f"  - 样本数量: {num_samples}")
        print(f"  - 文件大小: ~{num_samples * 0.5:.1f} MB")
    
    def visualize_sample(self, sample_id=0, save_name='step4_validation.png'):
        """
        可视化单个样本 (验证)
        Args:
            sample_id: 样本 ID
            save_name: 保存文件名
        """
        # 加载样本
        sample_path = os.path.join(self.output_dir, f"sample_{sample_id:05d}.npz")
        data = np.load(sample_path)
        
        sensor_image = data['x']
        map_water = data['y'][:, :, 0]
        map_sebum = data['y'][:, :, 1]
        meta = data['meta']
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 传感器图像 (色散拖尾)
        im0 = axes[0].imshow(sensor_image, cmap='gray')
        axes[0].set_title(f'输入: 传感器图像 (色散拖尾)\nADU 均值={sensor_image.mean():.0f}', fontsize=11, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Ground Truth: 水分布
        im1 = axes[1].imshow(map_water, cmap='Blues', vmin=0, vmax=0.5)
        axes[1].set_title(f'Ground Truth: 水浓度分布\n均值={map_water.mean():.3f}', fontsize=11, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Ground Truth: 皮脂分布
        im2 = axes[2].imshow(map_sebum, cmap='Oranges', vmin=0, vmax=0.5)
        axes[2].set_title(f'Ground Truth: 皮脂浓度分布\n均值={map_sebum.mean():.3f}', fontsize=11, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        save_path = os.path.join('output', save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 验证图保存: {save_path}")
        print(f"  - 样本 ID: {sample_id}")
        print(f"  - 元数据: 水={meta[0]:.3f}, 皮脂={meta[1]:.3f}, 黑色素={meta[2]:.3f}, 曝光={meta[3]:.2f}")
        plt.close()


# ================================
# 主程序
# ================================

if __name__ == "__main__":
    print("="*60)
    print("Step 4: Data Factory (Batch Generation Pipeline)")
    print("数据工厂：批量生成合成训练数据集")
    print("="*60)
    
    # 初始化数据工厂
    factory = DataFactory(
        assets_path='output/step1_standardized_data.csv',
        image_size=256,
        output_dir='output/datasets'
    )
    
    # 生成数据集
    factory.generate_batch(num_samples=100, seed_offset=0)
    
    # 可视化验证 (第一个样本)
    factory.visualize_sample(sample_id=0, save_name='step4_validation_sample0.png')
    
    # 可视化第二个样本 (对比差异)
    factory.visualize_sample(sample_id=1, save_name='step4_validation_sample1.png')
    
    print("\n" + "="*60)
    print("✓ Step 4 完成！")
    print("="*60)
