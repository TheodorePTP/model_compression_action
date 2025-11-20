# 🧪 知识蒸馏（Knowledge Distillation）

知识蒸馏是一种**模型压缩技术**，通过让小型“学生模型”（Student）学习大型“教师模型”（Teacher）的输出行为，实现**性能逼近教师、体积远小于教师**的目标。

## 核心思想
- 教师模型提供 **软标签（Soft Targets）**，包含类别间相似性信息
- 学生模型同时学习 **真实标签（Hard Labels）** 和 **教师输出（Soft Labels）**
- 引入 **温度参数（Temperature, T）** 平滑概率分布，提升知识迁移效果

## 典型应用场景
- 移动端部署轻量模型（如 MobileNet + KD）
- 大语言模型能力迁移（如 Llama → TinyLLM）
- 提升小模型泛化能力（有时甚至超越教师！）

## 本目录内容
- [经典方法](classic_methods.md)：Hinton KD, FitNets, Attention Transfer...
- [论文分析](paper_analysis)：对经典方法的论文逐个解析
- [实践指南](practical_guide.md)：PyTorch 实现、超参设置、避坑建议