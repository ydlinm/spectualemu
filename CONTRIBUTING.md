# Contributing to SpectualEmu

感谢您对 SpectualEmu 项目感兴趣！我们欢迎各种形式的贡献。

Thank you for your interest in SpectualEmu! We welcome contributions of all kinds.

## 如何贡献 / How to Contribute

### 报告问题 / Reporting Issues

如果您发现了 bug 或有功能建议，请创建一个 Issue：

If you find a bug or have a feature suggestion, please create an Issue:

1. 检查是否已存在相关 Issue / Check if a related Issue already exists
2. 提供详细的问题描述 / Provide a detailed description
3. 包含复现步骤（如果是 bug）/ Include reproduction steps (if it's a bug)
4. 附上相关的错误信息或截图 / Attach relevant error messages or screenshots

### 提交代码 / Submitting Code

1. **Fork 仓库 / Fork the repository**
   ```bash
   git clone https://github.com/your-username/spectualemu.git
   cd spectualemu
   ```

2. **创建特性分支 / Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **进行修改 / Make your changes**
   - 遵循现有的代码风格 / Follow the existing code style
   - 添加必要的注释 / Add necessary comments
   - 确保代码可以正常运行 / Ensure the code runs properly

4. **测试您的修改 / Test your changes**
   ```bash
   python step1_implementation.py
   python step2_implementation.py
   # 测试相关功能 / Test related functionality
   ```

5. **提交修改 / Commit your changes**
   ```bash
   git add .
   git commit -m "描述您的修改 / Describe your changes"
   ```

6. **推送到您的 Fork / Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **创建 Pull Request / Create a Pull Request**
   - 提供清晰的 PR 标题和描述 / Provide a clear PR title and description
   - 说明修改的目的和影响 / Explain the purpose and impact of changes
   - 链接相关的 Issues（如果有）/ Link related Issues (if any)

## 代码规范 / Code Guidelines

### Python 代码风格 / Python Code Style
- 遵循 PEP 8 规范 / Follow PEP 8 guidelines
- 使用有意义的变量名 / Use meaningful variable names
- 添加文档字符串（docstrings）/ Add docstrings
- 保持函数简洁（<50行）/ Keep functions concise (<50 lines)

### 物理模型修改 / Physics Model Changes
- 提供参考文献 / Provide references
- 添加物理验证代码 / Add physics validation code
- 在注释中说明物理意义 / Explain physical meaning in comments

### 文档更新 / Documentation Updates
- 同步更新中英文文档 / Update both Chinese and English docs
- 保持格式一致 / Maintain consistent formatting
- 添加必要的示例 / Add necessary examples

## 优先级 / Priorities

我们特别欢迎以下方面的贡献：

We especially welcome contributions in:

1. **Bug 修复** / Bug fixes
2. **性能优化** / Performance optimization
3. **新的物理模型** / New physics models
4. **文档改进** / Documentation improvements
5. **测试用例** / Test cases
6. **可视化功能** / Visualization features

## 行为准则 / Code of Conduct

- 尊重所有贡献者 / Respect all contributors
- 提供建设性的反馈 / Provide constructive feedback
- 保持友好和专业 / Be friendly and professional

## 获取帮助 / Getting Help

如果您有任何问题，可以：

If you have any questions, you can:

- 创建一个 Issue / Create an Issue
- 在 Pull Request 中提问 / Ask in your Pull Request
- 联系维护者 / Contact the maintainer

## 许可 / License

通过贡献代码，您同意您的贡献将在 MIT 许可证下发布。

By contributing code, you agree that your contributions will be licensed under the MIT License.

---

再次感谢您的贡献！

Thank you again for your contribution!
