# SpectualEmu - 高光谱皮肤成像仿真器

[English](#english-version) | [中文](#中文版本)

---

## 中文版本

### 📋 项目简介

SpectualEmu 是一个基于物理的高光谱皮肤成像仿真系统，专门用于生成训练数据集以支持皮肤光学特性分析。该系统模拟了从光源照射到传感器成像的完整光学链路，包括：

- 🔬 **物理建模**：基于生物光学原理的皮脂膜（Sebum）、水分、黑色素光谱建模
- 🌈 **光谱处理**：900-1700nm 近红外光谱范围，161个波长通道
- 🔍 **棱镜色散**：SF11致密燧石玻璃棱镜的色散效应模拟
- 📷 **传感器仿真**：包含量子效率、噪声、饱和等真实传感器特性
- 🎨 **数据生成**：批量生成带标注的合成训练数据集

### 🎯 主要特性

#### 1. 物理精确性
- **皮脂膜模型**：30% 水 + 70% 脂质的混合模型，符合真实皮肤表面特性
- **Farrell Dipole 扩散模型**：模拟亚表面散射（SSS）
- **Fresnel 镜面反射**：表面反射的物理计算
- **波长相关 PSF**：点扩散函数随波长动态变化

#### 2. 完整的成像链路
```
卤素光源 → 光纤束 → 多孔掩模 → 皮肤表面 → 棱镜色散 → InGaAs传感器
```

#### 3. 高保真渲染
- 7×7 多孔掩模系统（49个孔径）
- 高斯光束强度剖面（渐晕效应）
- 表面 + 亚表面双 PSF 引擎
- Splatting 方法正确处理孔径重叠

### 🏗️ 系统架构

项目采用四步流水线架构（Step 1-4）：

#### Step 1: 数据标准化与物理建模
**文件**: `step1_implementation.py`

主要功能：
- 加载并插值原始光谱数据到统一波长网格（900-1700nm, 5nm步长）
- 构建皮脂膜（Sebum）光谱：`μ_a,sebum = 0.3 × μ_a,water + 0.7 × μ_a,lipid`
- 生成散射系数（Power Law 模型）
- 生成黑色素吸收（Jacques 模型）
- 计算棱镜色散偏移（N-BK7/SF11 玻璃）
- 物理完整性检查（非负性、峰值对齐、能量守恒）

**输出**：
- `step1_standardized_assets.csv` - 标准化光谱资产
- 物理验证报告

#### Step 2: 多点光源照明与皮肤交互
**文件**: `step2_implementation.py`

主要功能：
- **多孔掩模系统**：7×7 网格（49个孔径），间距 30 像素
- **光束强度剖面**：高斯型渐晕，4mm处衰减至50%
- **双 PSF 引擎**：
  - 表面 PSF：窄高斯（σ=1.5px），模拟镜面反射
  - 亚表面 PSF：Farrell Dipole 模型，波长相关
- **Splatting 渲染**：逐孔径迭代，正确处理重叠
- **皮肤纹理**：加载真实皮肤纹理图，模拟表面粗糙度

**输出**：
- `step2_scene_hypercube.npy` - 场景高光谱立方体 [256×256×161]
- `step2_metadata.json` - 元数据
- 可视化图表（光束分布、掩模模式、RGB合成、光谱曲线等）

**核心物理模型**：
- Farrell Dipole 扩散：`I_sub(r,λ) = (P/4π) × exp(-μ_eff × r) / r`
- 有效衰减系数：`μ_eff = √(3μ_a(μ_a + μ_s'))`
- Fresnel 反射：`R = ((n1-n2)/(n1+n2))²`

#### Step 3: 棱镜色散与传感器成像
**文件**: `step3_implementation.py`

主要功能：
- **棱镜色散引擎**：SF11 致密燧石玻璃
  - 使用 Sellmeier 方程计算折射率
  - 色散偏移范围：[-15, +15] 像素
  - 双线性插值确保平滑变换
- **传感器模拟**：InGaAs 传感器
  - 量子效率曲线插值
  - 卤素光源光谱响应
  - 曝光时间控制
- **噪声模型**：
  - 泊松噪声（光子散粒噪声）
  - 读噪声（高斯噪声，σ=50e⁻）
  - 暗电流（1000e⁻/s）
  - 饱和处理（满阱容量：100,000e⁻）

**输出**：
- `step3_sensor_image.npy` - 传感器图像（包含噪声和色散）
- `step3_metadata.json` - 成像参数
- 可视化图表（色散效果、噪声分析、光谱对比）

#### Step 4: 数据工厂（批量生成）
**文件**: `step4_implementation.py`

主要功能：
- **批量数据生成**：可配置生成数量（默认100个样本）
- **参数随机化**：
  - 水分浓度：0.1-0.6（均匀分布）
  - 皮脂浓度：0.05-0.4（均匀分布）
  - 黑色素浓度：0.01-0.1（均匀分布）
  - 曝光时间：50-200ms（均匀分布）
- **Perlin 噪声**：生成自然的空间纹理变化
- **数据增强**：
  - 随机旋转（0-360°）
  - 随机缩放（0.9-1.1倍）
  - 高斯平滑变化
- **数据格式**：
  - 输入 `x`：传感器 ADU 图像 [H×W]
  - 标签 `y`：水分/皮脂浓度图 [H×W×2]
  - 元数据 `meta`：[水分均值, 皮脂均值, 黑色素均值, 曝光时间]

**输出**：
- `dataset/sample_XXXXX.npz` - 单个样本数据
- `dataset_metadata.json` - 数据集元数据（包含所有样本统计信息）
- 进度条显示批量生成状态

### 📦 安装

#### 环境要求
- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- SciPy
- Pillow
- tqdm

#### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/ydlinm/spectualemu.git
cd spectualemu

# 安装依赖（推荐使用虚拟环境）
pip install -r requirements.txt

# 或者手动安装
# pip install numpy pandas matplotlib scipy pillow tqdm
```

### 🚀 快速开始

#### 0. 项目结构

```
spectualemu/
├── assets/                          # 输入数据目录
│   ├── halogen_spectrum.csv        # 卤素灯光谱
│   ├── sensor_qe.csv               # 传感器量子效率
│   ├── water_mu_a.csv              # 水吸收系数
│   └── lipid_mu_a.csv              # 脂质吸收系数
├── step1_implementation.py         # Step 1: 数据标准化
├── step2_implementation.py         # Step 2: 场景生成
├── step3_implementation.py         # Step 3: 传感器成像
├── step4_implementation.py         # Step 4: 批量数据生成
├── verify_step3.py                 # 验证脚本
├── analyze_dataset.py              # 数据分析脚本
├── requirements.txt                # Python依赖
└── README.md                       # 本文档
```

#### 1. 准备输入数据

确保 `assets/` 目录包含以下文件：
- `halogen_spectrum.csv` - 卤素灯光谱
- `sensor_qe.csv` - 传感器量子效率
- `water_mu_a.csv` - 水吸收系数
- `lipid_mu_a.csv` - 脂质吸收系数

#### 2. 运行完整流水线

```bash
# Step 1: 数据标准化
python step1_implementation.py

# Step 2: 场景生成
python step2_implementation.py

# Step 3: 传感器成像
python step3_implementation.py

# Step 4: 批量数据生成
python step4_implementation.py
```

#### 3. 验证结果

```bash
# 验证 Step 3 输出
python verify_step3.py

# 分析生成的数据集
python analyze_dataset.py
```

### 📊 输出文件说明

| 文件 | 大小 | 说明 |
|------|------|------|
| `step1_standardized_assets.csv` | ~100 KB | 标准化光谱数据 |
| `step2_scene_hypercube.npy` | ~84 MB | 场景高光谱立方体 [256×256×161] |
| `step3_sensor_image.npy` | ~260 KB | 传感器图像（单通道灰度）|
| `dataset/sample_*.npz` | ~1 MB/个 | 训练样本（压缩格式）|
| `dataset_metadata.json` | <1 KB | 数据集统计信息 |
| `*.png` | 1-3 MB | 可视化图表 |

### 🔬 技术细节

#### 光谱范围
- **波长**：900-1700 nm（近红外）
- **步长**：5 nm
- **通道数**：161

#### 关键波长
- **1210 nm**：脂质吸收峰
- **1450 nm**：水吸收峰
- **1700 nm**：黑色素特征波段

#### 物理参数
- **像素大小**：15 μm
- **输出分辨率**：256×256 像素
- **孔径间距**：30 像素（450 μm）
- **光束直径**：80 像素（1.2 mm）
- **表面反射率**：~4%（Fresnel, n=1.4）

#### 传感器参数
- **类型**：InGaAs
- **满阱容量**：100,000 e⁻
- **读噪声**：50 e⁻
- **暗电流**：1000 e⁻/s
- **ADU 位深**：16-bit（0-65535）

### 📈 性能指标

- **Step 1 执行时间**：<1 秒
- **Step 2 执行时间**：~10 秒
- **Step 3 执行时间**：~5 秒
- **Step 4 执行时间**：~5 秒/样本（取决于参数设置）
- **内存占用**：~500 MB（主要用于高光谱立方体）

### 🎨 可视化示例

系统自动生成多种可视化图表：

1. **光束强度分布**：显示多孔掩模和高斯渐晕效果
2. **RGB 合成图像**：从高光谱数据合成的伪彩色图像
3. **光谱曲线**：中心 vs 边缘孔径的光谱对比
4. **空间强度热图**：特定波长的空间分布
5. **色散效果**：棱镜色散前后对比
6. **噪声分析**：各种噪声源的贡献

### 📚 物理模型参考

1. **Farrell et al.** - Dipole 扩散模型用于亚表面散射
2. **Jacques (1998)** - 皮肤组织光学特性
3. **Sellmeier 方程** - 玻璃折射率色散
4. **Beer-Lambert 定律** - 光吸收衰减

### 🐛 故障排除

#### 常见问题

**Q: 生成的光谱曲线是平坦的**
A: 确保在 Splatting 循环中正确应用了 `halogen_spectrum[wl_idx]`

**Q: 看不到亚表面散射（SSS）尾部**
A: 使用对数尺度可视化（`np.log10(intensity + eps)`）

**Q: 边缘孔径强度接近零**
A: 检查光束剖面参数，确保 4mm 处强度约为 50%

**Q: 出现 NaN 或 Inf 值**
A: 检查插值边界处理，使用 `left=0, right=0` 参数

### 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

### 📧 联系方式

项目维护者：ydlinm
- GitHub: [@ydlinm](https://github.com/ydlinm)

### 🙏 致谢

感谢所有为该项目做出贡献的研究人员和开发者。

---

## English Version

### 📋 Project Overview

SpectualEmu is a physics-based hyperspectral skin imaging simulation system designed to generate training datasets for skin optical property analysis. The system simulates the complete optical chain from light source illumination to sensor imaging, including:

- 🔬 **Physics Modeling**: Sebum, water, and melanin spectral modeling based on bio-optical principles
- 🌈 **Spectral Processing**: 900-1700nm NIR spectral range with 161 wavelength channels
- 🔍 **Prism Dispersion**: SF11 dense flint glass prism dispersion simulation
- 📷 **Sensor Simulation**: Realistic sensor characteristics including QE, noise, and saturation
- 🎨 **Data Generation**: Batch generation of annotated synthetic training datasets

### 🎯 Key Features

#### 1. Physical Accuracy
- **Sebum Model**: 30% water + 70% lipid mixture, matching real skin surface properties
- **Farrell Dipole Diffusion Model**: Subsurface scattering (SSS) simulation
- **Fresnel Specular Reflection**: Physical calculation of surface reflection
- **Wavelength-Dependent PSF**: Point spread function varies dynamically with wavelength

#### 2. Complete Imaging Chain
```
Halogen Source → Fiber Bundle → Multi-Pore Mask → Skin Surface → Prism Dispersion → InGaAs Sensor
```

#### 3. High-Fidelity Rendering
- 7×7 multi-pore mask system (49 apertures)
- Gaussian beam intensity profile (vignetting effect)
- Dual PSF engine (surface + subsurface)
- Splatting method for proper aperture overlap handling

### 🏗️ System Architecture

Four-step pipeline architecture (Step 1-4):

#### Step 1: Data Standardization and Physics Modeling
**File**: `step1_implementation.py`

Key functions:
- Load and interpolate raw spectral data to unified wavelength grid (900-1700nm, 5nm step)
- Construct Sebum spectrum: `μ_a,sebum = 0.3 × μ_a,water + 0.7 × μ_a,lipid`
- Generate scattering coefficients (Power Law model)
- Generate melanin absorption (Jacques model)
- Calculate prism dispersion shift (N-BK7/SF11 glass)
- Physics sanity checks (non-negativity, peak alignment, energy conservation)

**Outputs**:
- `step1_standardized_assets.csv` - Standardized spectral assets
- Physics validation report

#### Step 2: Multi-Point Source Illumination & Skin Interaction
**File**: `step2_implementation.py`

Key functions:
- **Multi-Pore Mask System**: 7×7 grid (49 apertures), 30-pixel spacing
- **Beam Intensity Profile**: Gaussian vignetting, 50% at 4mm
- **Dual PSF Engine**:
  - Surface PSF: Narrow Gaussian (σ=1.5px) for specular reflection
  - Subsurface PSF: Farrell Dipole model, wavelength-dependent
- **Splatting Rendering**: Per-aperture iteration, correct overlap handling
- **Skin Texture**: Load real skin texture for surface roughness simulation

**Outputs**:
- `step2_scene_hypercube.npy` - Scene hyperspectral cube [256×256×161]
- `step2_metadata.json` - Metadata
- Visualization plots (beam distribution, mask pattern, RGB composite, spectra, etc.)

**Core Physics Models**:
- Farrell Dipole diffusion: `I_sub(r,λ) = (P/4π) × exp(-μ_eff × r) / r`
- Effective attenuation: `μ_eff = √(3μ_a(μ_a + μ_s'))`
- Fresnel reflection: `R = ((n1-n2)/(n1+n2))²`

#### Step 3: Prism Dispersion & Sensor Imaging
**File**: `step3_implementation.py`

Key functions:
- **Prism Dispersion Engine**: SF11 dense flint glass
  - Sellmeier equation for refractive index
  - Dispersion shift range: [-15, +15] pixels
  - Bilinear interpolation for smooth transformation
- **Sensor Simulation**: InGaAs sensor
  - Quantum efficiency curve interpolation
  - Halogen source spectral response
  - Exposure time control
- **Noise Model**:
  - Poisson noise (photon shot noise)
  - Read noise (Gaussian, σ=50e⁻)
  - Dark current (1000e⁻/s)
  - Saturation handling (full well: 100,000e⁻)

**Outputs**:
- `step3_sensor_image.npy` - Sensor image (with noise and dispersion)
- `step3_metadata.json` - Imaging parameters
- Visualization plots (dispersion effect, noise analysis, spectral comparison)

#### Step 4: Data Factory (Batch Generation)
**File**: `step4_implementation.py`

Key functions:
- **Batch Data Generation**: Configurable sample count (default: 100)
- **Parameter Randomization**:
  - Water concentration: 0.1-0.6 (uniform distribution)
  - Sebum concentration: 0.05-0.4 (uniform distribution)
  - Melanin concentration: 0.01-0.1 (uniform distribution)
  - Exposure time: 50-200ms (uniform distribution)
- **Perlin Noise**: Generate natural spatial texture variations
- **Data Augmentation**:
  - Random rotation (0-360°)
  - Random scaling (0.9-1.1×)
  - Gaussian smoothing variations
- **Data Format**:
  - Input `x`: Sensor ADU image [H×W]
  - Label `y`: Water/Sebum concentration map [H×W×2]
  - Metadata `meta`: [water_mean, sebum_mean, melanin_mean, exposure_time]

**Outputs**:
- `dataset/sample_XXXXX.npz` - Individual sample data
- `dataset_metadata.json` - Dataset metadata (all sample statistics)
- Progress bar for batch generation

### 📦 Installation

#### Requirements
- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- SciPy
- Pillow
- tqdm

#### Installation Steps

```bash
# Clone the repository
git clone https://github.com/ydlinm/spectualemu.git
cd spectualemu

# Install dependencies (virtual environment recommended)
pip install -r requirements.txt

# Or install manually
# pip install numpy pandas matplotlib scipy pillow tqdm
```

### 🚀 Quick Start

#### 0. Project Structure

```
spectualemu/
├── assets/                          # Input data directory
│   ├── halogen_spectrum.csv        # Halogen lamp spectrum
│   ├── sensor_qe.csv               # Sensor quantum efficiency
│   ├── water_mu_a.csv              # Water absorption coefficient
│   └── lipid_mu_a.csv              # Lipid absorption coefficient
├── step1_implementation.py         # Step 1: Data standardization
├── step2_implementation.py         # Step 2: Scene generation
├── step3_implementation.py         # Step 3: Sensor imaging
├── step4_implementation.py         # Step 4: Batch data generation
├── verify_step3.py                 # Verification script
├── analyze_dataset.py              # Data analysis script
├── requirements.txt                # Python dependencies
└── README.md                       # This document
```

#### 1. Prepare Input Data

Ensure the `assets/` directory contains:
- `halogen_spectrum.csv` - Halogen lamp spectrum
- `sensor_qe.csv` - Sensor quantum efficiency
- `water_mu_a.csv` - Water absorption coefficient
- `lipid_mu_a.csv` - Lipid absorption coefficient

#### 2. Run the Complete Pipeline

```bash
# Step 1: Data standardization
python step1_implementation.py

# Step 2: Scene generation
python step2_implementation.py

# Step 3: Sensor imaging
python step3_implementation.py

# Step 4: Batch data generation
python step4_implementation.py
```

#### 3. Verify Results

```bash
# Verify Step 3 output
python verify_step3.py

# Analyze generated dataset
python analyze_dataset.py
```

### 📊 Output Files

| File | Size | Description |
|------|------|-------------|
| `step1_standardized_assets.csv` | ~100 KB | Standardized spectral data |
| `step2_scene_hypercube.npy` | ~84 MB | Scene hyperspectral cube [256×256×161] |
| `step3_sensor_image.npy` | ~260 KB | Sensor image (single-channel grayscale) |
| `dataset/sample_*.npz` | ~1 MB each | Training samples (compressed) |
| `dataset_metadata.json` | <1 KB | Dataset statistics |
| `*.png` | 1-3 MB | Visualization plots |

### 🔬 Technical Details

#### Spectral Range
- **Wavelength**: 900-1700 nm (near-infrared)
- **Step Size**: 5 nm
- **Channels**: 161

#### Key Wavelengths
- **1210 nm**: Lipid absorption peak
- **1450 nm**: Water absorption peak
- **1700 nm**: Melanin characteristic band

#### Physical Parameters
- **Pixel Size**: 15 μm
- **Output Resolution**: 256×256 pixels
- **Aperture Spacing**: 30 pixels (450 μm)
- **Beam Diameter**: 80 pixels (1.2 mm)
- **Surface Reflectance**: ~4% (Fresnel, n=1.4)

#### Sensor Parameters
- **Type**: InGaAs
- **Full Well Capacity**: 100,000 e⁻
- **Read Noise**: 50 e⁻
- **Dark Current**: 1000 e⁻/s
- **ADU Bit Depth**: 16-bit (0-65535)

### 📈 Performance Metrics

- **Step 1 Execution Time**: <1 second
- **Step 2 Execution Time**: ~10 seconds
- **Step 3 Execution Time**: ~5 seconds
- **Step 4 Execution Time**: ~5 seconds/sample (depends on settings)
- **Memory Usage**: ~500 MB (mainly for hyperspectral cube)

### 🎨 Visualization Examples

The system automatically generates various visualization plots:

1. **Beam Intensity Distribution**: Shows multi-pore mask and Gaussian vignetting
2. **RGB Composite Image**: Pseudo-color image synthesized from hyperspectral data
3. **Spectral Curves**: Center vs edge aperture spectral comparison
4. **Spatial Intensity Heatmap**: Spatial distribution at specific wavelengths
5. **Dispersion Effect**: Before and after prism dispersion comparison
6. **Noise Analysis**: Contribution of various noise sources

### 📚 Physics Model References

1. **Farrell et al.** - Dipole diffusion model for subsurface scattering
2. **Jacques (1998)** - Optical properties of skin tissue
3. **Sellmeier Equation** - Glass refractive index dispersion
4. **Beer-Lambert Law** - Light absorption attenuation

### 🐛 Troubleshooting

#### Common Issues

**Q: Generated spectral curves are flat**
A: Ensure `halogen_spectrum[wl_idx]` is properly applied in the splatting loop

**Q: Cannot see subsurface scattering (SSS) tail**
A: Use logarithmic scale visualization (`np.log10(intensity + eps)`)

**Q: Edge aperture intensities are near zero**
A: Check beam profile parameters, ensure ~50% intensity at 4mm

**Q: NaN or Inf values appear**
A: Check interpolation boundary handling, use `left=0, right=0` parameters

### 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

### 📧 Contact

Project Maintainer: ydlinm
- GitHub: [@ydlinm](https://github.com/ydlinm)

### 🙏 Acknowledgments

Thanks to all researchers and developers who contributed to this project.

---

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{spectualemu2024,
  title={SpectualEmu: A Physics-Based Hyperspectral Skin Imaging Simulator},
  author={ydlinm},
  year={2024},
  url={https://github.com/ydlinm/spectualemu}
}
```
