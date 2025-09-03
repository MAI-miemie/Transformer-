# Transformer 深度学习项目

基于Transformer架构实现的两个深度学习任务：中文姓名性别预测和英文姓名生成。

## 📁 项目结构

```
Transformer-/
├── Transformer性别预测/
│   ├── Trans_性别预测.ipynb
│   └── Chinese_Names_Corpus_Gender（120W）.txt
├── Transformer名字生成/
│   ├── Trans_名字生成.ipynb
│   └── English_Cn_Name_Corpus（48W）.txt
└── README.md
```

## 🚀 项目概述

### 项目1: 中文姓名性别预测
- **任务**: 基于Transformer Encoder架构预测中文姓名的性别
- **数据集**: 120万中文姓名性别标注数据
- **模型**: Transformer Encoder + 分类头
- **特点**: 使用自注意力机制捕捉姓名中的性别特征

### 项目2: 英文姓名生成
- **任务**: 基于Transformer Decoder架构生成英文姓名
- **数据集**: 48万英文姓名数据
- **模型**: Transformer Decoder + 生成头
- **特点**: 支持指定起始字符，自回归生成完整姓名

## 🏗️ 技术架构

### Transformer Encoder (性别预测)
- **位置编码**: 正弦余弦位置编码
- **多头自注意力**: 8头注意力机制
- **前馈网络**: 两层全连接网络
- **残差连接**: 每个子层后都有残差连接
- **层归一化**: 提升训练稳定性

### Transformer Decoder (姓名生成)
- **掩码自注意力**: 防止看到未来token
- **位置编码**: 与Encoder相同的位置编码
- **前馈网络**: 特征提取和变换
- **输出层**: 线性层映射到词汇表

## 📊 数据集说明

### 中文姓名性别数据集
- **文件名**: `Chinese_Names_Corpus_Gender（120W）.txt`
- **数据量**: 约120万条记录
- **格式**: CSV格式，包含姓名和性别标签
- **标签**: 男(0)、女(1)

### 英文姓名数据集
- **文件名**: `English_Cn_Name_Corpus（48W）.txt`
- **数据量**: 约48万条记录
- **格式**: 纯文本，每行一个姓名
- **特点**: 包含英文和中文姓名

## 🛠️ 环境要求

```python
# 核心依赖
torch >= 1.9.0
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.3.0

# 可选依赖
jupyter >= 1.0.0
ipykernel >= 6.0.0
```

## 📖 使用方法

### 1. 性别预测任务

```python
# 运行完整的训练流程
from Transformer性别预测.Trans_性别预测 import *

# 准备数据
train, test, dict, dict_size = prepare_gender_data()

# 训练模型
model = train_gender_classifier(train, test, dict, dict_size)

# 预测性别
test_gender_predictions(model, dict)
```

### 2. 姓名生成任务

```python
# 运行完整的训练流程
from Transformer名字生成.Trans_名字生成 import *

# 准备数据
dat, dict, charset_size = prepare_name_data()

# 训练模型
model = train_name_generator(dat, dict, charset_size)

# 生成姓名
generated_names = generate_names(model, dict, num_names=10, max_len=8)
```

## ⚙️ 模型参数配置

### 性别预测模型
```python
# 默认参数
embed_size = 64          # 嵌入维度
num_layers = 3           # Transformer层数
heads = 8                # 注意力头数
qk_dim = 64             # Q/K投影维度
ff_hidden_size = 128     # 前馈网络隐藏层大小
dropout = 0.1            # Dropout概率
max_length = 10          # 最大序列长度
```

### 姓名生成模型
```python
# 默认参数
embed_size = 64          # 嵌入维度
num_layers = 3           # Transformer层数
heads = 8                # 注意力头数
qk_dim = 64             # Q/K投影维度
ff_hidden_size = 128     # 前馈网络隐藏层大小
dropout = 0.1            # Dropout概率
max_length = 10          # 最大序列长度
```

## 📈 训练结果

### 性别预测性能
- **训练集大小**: 10,000个样本
- **测试集大小**: 1,000个样本
- **准确率**: 约97.9%
- **训练时间**: 约5个epoch

### 姓名生成性能
- **训练集大小**: 约48万个姓名
- **词汇表大小**: 50个常用字符
- **生成质量**: 可生成符合语言习惯的姓名
- **训练时间**: 约20个epoch

## 🎯 特色功能

### 性别预测
- 支持变长姓名输入
- 自动字符频率统计
- 常见字符过滤
- 实时性别预测

### 姓名生成
- 支持指定起始字符
- 可调节生成温度
- 自动长度控制
- 批量姓名生成

## 🏗️ 代码结构详解

### 核心组件
1. **位置编码**: `PositionalEncoding` - 为序列添加位置信息
2. **自注意力**: `SelfAttention` - 多头注意力机制
3. **前馈网络**: `FeedForward` - 特征变换和提取
4. **编码器块**: `EncoderBlock` - 完整的编码器单元
5. **解码器块**: `DecoderBlock` - 完整的解码器单元

### 数据处理
1. **字符索引化**: 将字符转换为数字索引
2. **序列填充**: 统一序列长度
3. **One-hot编码**: 字符的向量表示
4. **数据增强**: 随机采样和打乱

## 🚨 注意事项

1. **数据文件路径**: 确保数据文件在正确的相对路径下
2. **内存要求**: 大数据集训练需要足够的内存
3. **GPU加速**: 建议使用GPU进行训练
4. **依赖版本**: 注意PyTorch和其他包的版本兼容性

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

### 贡献方式
1. Fork本仓库
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 📄 许可证

本项目采用MIT许可证。

## 👨‍💻 作者

- **MAI-miemie** - 项目创建者
- **GitHub**: [@MAI-miemie](https://github.com/MAI-miemie)

## 🙏 致谢

感谢以下开源项目和数据集：
- PyTorch深度学习框架
- Transformer架构论文
- 中文姓名性别数据集
- 英文姓名语料库

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: [项目Issues页面](https://github.com/MAI-miemie/Transformer-/issues)

---

⭐ 如果这个项目对您有帮助，请给个Star支持一下！
