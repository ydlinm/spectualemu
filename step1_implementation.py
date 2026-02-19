"""
Step 1: Data Standardization, Physics Modeling, and Validation
光谱数据标准化与物理建模实现
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

class SpectralDataProcessor:
    """光谱数据处理与物理建模"""
    
    def __init__(self, data_dir='assets'):
        self.data_dir = data_dir
        # 主波长网格：900-1700nm，步长5nm
        self.master_wavelengths = np.arange(900, 1701, 5)
        self.results = {}
        
    def load_and_interpolate(self):
        """加载并插值所有数据到主波长网格"""
        print("📂 加载原始数据...")
        
        # 加载卤素灯光谱
        halogen = pd.read_csv(os.path.join(self.data_dir, 'halogen_spectrum.csv'))
        self.results['halogen_raw'] = halogen
        self.results['halogen_interp'] = np.interp(
            self.master_wavelengths, 
            halogen.iloc[:, 0], 
            halogen.iloc[:, 1],
            left=0, right=0
        )
        
        # 加载传感器量子效率
        sensor = pd.read_csv(os.path.join(self.data_dir, 'sensor_qe.csv'))
        self.results['sensor_raw'] = sensor
        self.results['sensor_interp'] = np.interp(
            self.master_wavelengths,
            sensor.iloc[:, 0],
            sensor.iloc[:, 1],
            left=0, right=0
        )
        
        # 加载水吸收系数 (TSV格式，跳过前3行注释)
        water = pd.read_csv(
            os.path.join(self.data_dir, 'water_mu_a.csv'), 
            sep='\t', 
            skiprows=3,
            encoding='utf-8',
            on_bad_lines='skip'
        )
        # 过滤掉可能的额外列（只保留前两列）
        water = water.iloc[:, :2]
        # 清理列名和数据
        water.columns = ['wavelength', 'absorption']
        # 转换为数值类型并删除无效行
        water['wavelength'] = pd.to_numeric(water['wavelength'], errors='coerce')
        water['absorption'] = pd.to_numeric(water['absorption'], errors='coerce')
        water = water.dropna()
        
        self.results['water_raw'] = water
        self.results['water_interp'] = np.interp(
            self.master_wavelengths,
            water['wavelength'].values,
            water['absorption'].values,
            left=water['absorption'].values[0], 
            right=water['absorption'].values[-1]
        )
        
        # 加载脂质吸收系数
        lipid = pd.read_csv(os.path.join(self.data_dir, 'lipid_mu_a.csv'))
        self.results['lipid_raw'] = lipid
        self.results['lipid_interp'] = np.interp(
            self.master_wavelengths,
            lipid.iloc[:, 0],
            lipid.iloc[:, 1],
            left=lipid.iloc[0, 1],
            right=lipid.iloc[-1, 1]
        )
        
        print(f"✓ 插值完成，主波长网格: {len(self.master_wavelengths)} 个点 ({self.master_wavelengths[0]}-{self.master_wavelengths[-1]} nm)")
        
    def construct_sebum_model(self):
        """构建皮脂膜模型：30%水 + 70%脂质"""
        print("\n🧪 构建皮脂膜 (Hydro-lipid Film) 模型...")
        
        mu_a_water = self.results['water_interp']
        mu_a_lipid = self.results['lipid_interp']
        
        # Sebum = 30%水 + 70%油
        self.results['sebum_interp'] = 0.3 * mu_a_water + 0.7 * mu_a_lipid
        
        # 统计特征
        water_peak_idx = np.argmax(mu_a_water)
        water_peak_wl = self.master_wavelengths[water_peak_idx]
        lipid_peak_idx = np.argmax(mu_a_lipid)
        lipid_peak_wl = self.master_wavelengths[lipid_peak_idx]
        sebum_peak_idx = np.argmax(self.results['sebum_interp'])
        sebum_peak_wl = self.master_wavelengths[sebum_peak_idx]
        
        print(f"  - Water peak: {water_peak_wl} nm (μa = {mu_a_water[water_peak_idx]:.4f})")
        print(f"  - Lipid peak: {lipid_peak_wl} nm (μa = {mu_a_lipid[lipid_peak_idx]:.4f})")
        print(f"  - Sebum peak: {sebum_peak_wl} nm (μa = {self.results['sebum_interp'][sebum_peak_idx]:.4f})")
        print("✓ 皮脂膜光谱构建完成")
        
    def generate_scattering_model(self):
        """生成散射系数 μs' (Power Law)"""
        print("\n🌊 生成散射模型 (Power Law)...")
        
        a = 1.5  # 散射幅度
        b = 1.0  # 散射指数
        lambda_0 = 500  # 参考波长 (nm)
        
        # μs'(λ) = a * (λ/λ0)^(-b)
        self.results['scatter_mus_interp'] = a * (self.master_wavelengths / lambda_0) ** (-b)
        
        print(f"  - 参数: a={a}, b={b}, λ0={lambda_0}nm")
        print(f"  - μs' @ 900nm: {self.results['scatter_mus_interp'][0]:.4f}")
        print(f"  - μs' @ 1700nm: {self.results['scatter_mus_interp'][-1]:.4f}")
        print("✓ 散射模型生成完成")
        
    def generate_melanin_model(self):
        """生成黑色素吸收 (Jacques Model)"""
        print("\n🎨 生成黑色素吸收模型 (Jacques)...")
        
        # Jacques模型参数
        C_mel = 0.05  # 黑色素浓度 (5%)
        melanin_baseline = 6.6e10  # 基准吸收系数
        
        # μa_melanin(λ) = C_mel * baseline * λ^(-3.33)
        self.results['melanin_interp'] = C_mel * melanin_baseline * (self.master_wavelengths ** (-3.33))
        
        print(f"  - 浓度: {C_mel*100}%")
        print(f"  - μa @ 900nm: {self.results['melanin_interp'][0]:.4e}")
        print(f"  - μa @ 1700nm: {self.results['melanin_interp'][-1]:.4e}")
        print("✓ 黑色素模型生成完成")
        
    def generate_prism_dispersion(self):
        """生成N-BK7玻璃棱镜色散 (Cauchy方程)"""
        print("\n🔬 计算棱镜色散 (N-BK7 Glass)...")
        
        # N-BK7 Cauchy系数 (Schott数据)
        B1, B2, B3 = 1.03961212, 0.231792344, 1.01046945
        C1, C2, C3 = 0.00600069867, 0.0200179144, 103.560653
        
        # 波长单位转换 μm
        lambda_um = self.master_wavelengths / 1000.0
        lambda_sq = lambda_um ** 2
        
        # Sellmeier方程
        n_squared = 1 + (B1*lambda_sq)/(lambda_sq-C1) + (B2*lambda_sq)/(lambda_sq-C2) + (B3*lambda_sq)/(lambda_sq-C3)
        n = np.sqrt(n_squared)
        
        # 简化色散模型：相对900nm的折射率差
        n_ref = n[0]  # 900nm折射率
        delta_n = n - n_ref
        
        # 假设像素偏移与折射率变化成正比（简化几何）
        pixel_per_delta_n = 50  # 50像素/折射率单位
        self.results['prism_shift_pixels'] = delta_n * pixel_per_delta_n
        
        print(f"  - 折射率 @ 900nm: {n[0]:.6f}")
        print(f"  - 折射率 @ 1700nm: {n[-1]:.6f}")
        print(f"  - 像素偏移范围: {self.results['prism_shift_pixels'].min():.2f} ~ {self.results['prism_shift_pixels'].max():.2f} pixels")
        print("✓ 棱镜色散计算完成")
        
    def validate_physics(self):
        """物理自检验证"""
        print("\n🛡️  物理自检验证...")
        
        errors = []
        warnings = []
        
        # 检查1：非负性
        check_arrays = {
            'halogen': self.results['halogen_interp'],
            'sensor_qe': self.results['sensor_interp'],
            'water_mu_a': self.results['water_interp'],
            'lipid_mu_a': self.results['lipid_interp'],
            'sebum_mu_a': self.results['sebum_interp'],
            'scatter_mus': self.results['scatter_mus_interp'],
            'melanin_mu_a': self.results['melanin_interp']
        }
        
        for name, arr in check_arrays.items():
            if np.any(arr < 0):
                errors.append(f"❌ {name} 存在负值！")
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                errors.append(f"❌ {name} 存在 NaN 或 Inf！")
        
        if not errors:
            print("  ✓ 所有数组通过非负性检查")
        
        # 检查2：水峰对齐
        water_peak_idx = np.argmax(self.results['water_interp'])
        water_peak_wl = self.master_wavelengths[water_peak_idx]
        
        if 1440 <= water_peak_wl <= 1460:
            print(f"  ✓ 水峰对齐正确: {water_peak_wl} nm (预期 1450±10 nm)")
        else:
            warnings.append(f"⚠️  水峰偏移: {water_peak_wl} nm (预期 1450 nm)")
        
        # 检查3：散射单调递减
        scatter = self.results['scatter_mus_interp']
        if np.all(np.diff(scatter) < 0):
            print("  ✓ 散射系数单调递减符合物理规律")
        else:
            errors.append("❌ 散射系数非单调递减！")
        
        # 检查4：数据完整性
        if len(self.master_wavelengths) == len(self.results['halogen_interp']):
            print("  ✓ 数据维度一致")
        else:
            errors.append("❌ 数据维度不匹配！")
        
        # 总结
        if errors:
            print("\n" + "\n".join(errors))
            raise ValueError("物理验证失败！")
        
        if warnings:
            print("\n" + "\n".join(warnings))
        
        print("\n✅ 物理验证通过")
        
    def save_results(self, output_dir='output'):
        """保存标准化数据"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # 创建主数据表
        df = pd.DataFrame({
            'wavelength_nm': self.master_wavelengths,
            'halogen_spectrum': self.results['halogen_interp'],
            'sensor_qe': self.results['sensor_interp'],
            'water_mu_a': self.results['water_interp'],
            'lipid_mu_a': self.results['lipid_interp'],
            'sebum_mu_a': self.results['sebum_interp'],
            'scatter_mus': self.results['scatter_mus_interp'],
            'melanin_mu_a': self.results['melanin_interp'],
            'prism_shift_px': self.results['prism_shift_pixels']
        })
        
        output_file = os.path.join(output_dir, 'step1_standardized_data.csv')
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n💾 标准化数据已保存: {output_file}")
        
        return df
        
    def run(self):
        """执行完整流程"""
        print("="*60)
        print("Step 1: 光谱数据标准化与物理建模")
        print("="*60)
        
        self.load_and_interpolate()
        self.construct_sebum_model()
        self.generate_scattering_model()
        self.generate_melanin_model()
        self.generate_prism_dispersion()
        self.validate_physics()
        df = self.save_results()
        
        print("\n" + "="*60)
        print("✅ Step 1 执行完成")
        print("="*60)
        
        return df


if __name__ == '__main__':
    processor = SpectralDataProcessor()
    result_df = processor.run()
    
    # 输出统计摘要
    print("\n📊 数据统计摘要:")
    print(result_df.describe().loc[['min', 'max', 'mean'], :].to_string())
