# Transformer 深度学习项目

基于Transformer架构实现的两个深度学习任务：中文姓名性别预测和中文姓名生成。

## 项目结构

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

## 项目概述

### 项目1: 中文姓名性别预测
- **任务**: 基于Transformer Encoder-Decoder架构预测中文姓名的性别
- **数据集**: 120万中文姓名性别标注数据
- **模型**: Transformer Encoder-Decoder + 分类头
- **特点**: 使用自注意力机制捕捉姓名中的性别特征

### 项目2: 中文姓名生成
- **任务**: 基于Transformer Encoder-Decoder架构生成中文姓名
- **数据集**: 48万中文姓名数据
- **模型**: Transformer Encoder-Decoder + 生成头
- **特点**: 支持指定起始字符，自回归生成完整姓名，防过拟合技术

## 技术架构

### Transformer Encoder-Decoder (两个项目通用)
- **位置编码**: 正弦余弦位置编码
- **多头自注意力**: 4-8头注意力机制
- **前馈网络**: 两层全连接网络
- **残差连接**: 每个子层后都有残差连接
- **层归一化**: 提升训练稳定性
- **交叉注意力**: Decoder与Encoder的交互机制

### 防过拟合技术 (名字生成项目)
- **Dropout正则化**: 0.3的高dropout率
- **早停机制**: 验证损失不改善时自动停止
- **学习率调度**: ReduceLROnPlateau自动调整
- **梯度裁剪**: 防止梯度爆炸
- **权重初始化**: Xavier初始化

## 数据集说明

### 中文姓名性别数据集
- **文件名**: `Chinese_Names_Corpus_Gender（120W）.txt`
- **数据量**: 约120万条记录
- **格式**: CSV格式，包含姓名和性别标签
- **标签**: 男(0)、女(1)

### 中文姓名数据集
- **文件名**: `English_Cn_Name_Corpus（48W）.txt`
- **数据量**: 约48万条记录
- **格式**: 纯文本，每行一个姓名
- **特点**: 中文姓名数据，字典大小100个常用字符

## 环境要求

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

## 使用方法

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
train, test, dict, charset_size = prepare_name_data()

# 训练模型（包含防过拟合技术）
model = train_name_generator_improved(train, test, dict, charset_size, 
                                     patience=10, min_delta=0.001, min_train_loss=0.5)

# 生成姓名（支持温度控制）
generated_name = model.generate("阿", dict, temperature=1.0, top_k=10)
```

## 模型参数配置

### 性别预测模型
```python
# 默认参数
embed_size = 64          # 嵌入维度
num_layers = 2           # Transformer层数
heads = 4                # 注意力头数
qk_dim = 64             # Q/K投影维度
ff_hidden_size = 128     # 前馈网络隐藏层大小
dropout = 0.3            # Dropout概率
max_length = 10          # 最大序列长度
```

### 姓名生成模型（防过拟合版本）
```python
# 默认参数
embed_size = 64          # 嵌入维度
num_layers = 2           # Transformer层数
heads = 4                # 注意力头数
qk_dim = 64             # Q/K投影维度
ff_hidden_size = 128     # 前馈网络隐藏层大小
dropout = 0.3            # 高Dropout概率
max_length = 10          # 最大序列长度
batch_size = 258         # 批次大小
learning_rate = 0.0005   # 学习率
patience = 10            # 早停耐心值
min_train_loss = 0.5     # 最低训练损失阈值
```

## 训练结果

### 性别预测性能
- **训练集大小**: 10,000个样本
- **测试集大小**: 1,000个样本
- **准确率**: 约97.9%
- **训练时间**: 约5个epoch

### 姓名生成性能（防过拟合版本）
- **训练集大小**: 185,955个姓名
- **验证集大小**: 46,489个姓名
- **字典大小**: 100个常用字符
- **生成质量**: 支持温度控制和Top-k采样
- **训练时间**: 3个epoch（早停机制）
- **过拟合控制**: 成功防止过拟合，训练和验证损失同步下降

## 特色功能

### 性别预测
- 支持变长姓名输入
- 自动字符频率统计
- 常见字符过滤
- 实时性别预测

### 姓名生成（防过拟合版本）
- 支持指定起始字符
- 可调节生成温度（0.5-2.0）
- Top-k采样提高生成质量
- 自动长度控制
- 批量姓名生成
- 详细的训练过程监控
- 早停机制防止过拟合
- 学习率自动调度

## 生成示例

### 不同温度下的生成效果

#### 温度 0.5 (保守生成)
- 阿 -> 阿布普安勒沙迪赫迪赫
- 库 -> 库卡尼谢谢阿巴普卡巴
- 卡 -> 卡拉耶尼谢卡尼谢谢谢

#### 温度 1.0 (平衡生成)
- 阿 -> 阿布普费安费林阿迪迪
- 库 -> 库卡阿谢尼普普赫赫卡
- 卡 -> 卡兹哈谢尼耶沃谢哈基

#### 温度 2.0 (创造性生成)
- 阿 -> 阿布卡尼卡阿迪德拉阿
- 库 -> 库巴谢古阿赫谢林哈阿
- 卡 -> 卡拉尼谢普尼谢阿基普

### 批量生成测试结果
- 总生成数: 20
- 唯一名字数: 20
- 多样性比例: 100.00%

## 代码结构详解

### 核心组件
1. **位置编码**: `NamePositionalEncoding` - 为序列添加位置信息
2. **多头自注意力**: `NameMultiHeadAttention` - 多头注意力机制
3. **前馈网络**: `NameFeedForward` - 特征变换和提取
4. **编码器层**: `NameTransformerEncoderLayer` - 完整的编码器单元
5. **解码器层**: `NameTransformerDecoderLayer` - 完整的解码器单元
6. **生成器**: `ImprovedNameGenerator` - 防过拟合的名字生成器

### 数据处理
1. **字符索引化**: 将字符转换为数字索引
2. **序列填充**: 统一序列长度
3. **数据分割**: 训练集和验证集分割
4. **数据增强**: 随机采样和打乱

### 训练优化
1. **早停机制**: 防止过拟合
2. **学习率调度**: 自动调整学习率
3. **梯度裁剪**: 防止梯度爆炸
4. **权重初始化**: Xavier初始化

## 注意事项

1. **数据文件路径**: 确保数据文件在正确的相对路径下
2. **内存要求**: 大数据集训练需要足够的内存
3. **GPU加速**: 建议使用GPU进行训练
4. **依赖版本**: 注意PyTorch和其他包的版本兼容性
5. **训练监控**: 关注训练和验证损失的变化趋势

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

### 贡献方式
1. Fork本仓库
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 许可证

本项目采用MIT许可证。

## 作者

- **MAI-miemie** - 项目创建者
- **GitHub**: [@MAI-miemie](https://github.com/MAI-miemie)

## 致谢

感谢以下开源项目和数据集：
- PyTorch深度学习框架
- Transformer架构论文
- 中文姓名性别数据集
- 中文姓名语料库

## 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: [项目Issues页面](https://github.com/MAI-miemie/Transformer-/issues)

---

如果这个项目对您有帮助，请给个Star支持一下！
