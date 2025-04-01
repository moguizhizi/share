## 概述

本项目实现了一个基于 RAG（`Retrieval-Augmented Generation`）的 RL 引导代码助手，首先通过检索系统获取相关代码片段和生成答案来解决用户关于代码的问题。系统包含以下核心模块：

1. **检索**：通过检索系统获取相关代码片段，辅助生成答案。
2. **主知识**：结合生成模型，生成关于代码的答案。
3. **强化学习（RL）优化**：使用 RL 优化生成结果，支持代码相关任务。

## RAG 管道

1. **数据集**：使用 CodeSearchNet 数据集（Python 子集），从中提取了 1000 个函数的代码（`func_code_string`）和文档字符串（`func_documentation_string`）作为语料库。
2. **检索**：使用 SentenceTransformer（`all-MiniLM-L6-v2` 模型）将用户查询和代码片段编码为嵌入向量，计算余弦相似度，检索 Top-3 相关代码片段。
3. **生成**：使用 Salesforce/codegen-350M-mono 模型生成答案。将检索到的代码片段注入提示（prompt），生成 3 个候选答案。
4. **GPU 加速**：所有计算（嵌入、余弦相似度、生成）都在 GPU 上运行（如果可用），通过 `torch.device` 动态选择设备。

## 奖励函数

设计了一个简单的奖励函数，综合考虑以下因素：

1. **正确性**：候选答案是否包含查询中的关键词。
    1. 计算公式：`correctness = 关键词匹配数 / 关键词总数`。
    2. 关键词匹配基于小写字符串比较。
2. **简洁性**：候选答案的长度。
    1. 计算公式：`brevity = max(0, 1 - (length - 50) / 150)`。
    2. 假设 50-200 字为理想长度。
3. **总奖励**：综合正确性和简洁性。
    1. 计算公式：`reward = 0.7 * correctness + 0.3 * brevity`。
    2. 权重分配：正确性占 70%，简洁性占 30%。

## RL 优化

使用强化学习对候选答案进行优化，具体步骤如下：

1. 使用奖励函数对 3 个候选答案进行排序。
2. 按奖励降序排序，并基于 softmax 概率（`exp(rewards) / sum(exp(rewards))`）加权采样最佳答案，模拟 PPO（`Proximal Policy Optimization`）。

## 技术栈

项目使用的技术依赖如下：

1. **模型**：`sentence-transformers`（`all-MiniLM-L6-v2`）
2. **生产**：`transformers`（`Salesforce/codegen-350M-mono`）
3. **数据集**：`datasets`（`CodeSearchNet`）
4. **计算**：`torch`（支持 GPU 加速）、`numpy`

## 未来展望
1. **优化奖励函数，加入语法检查**。
2. **扩展数据集，支持更多语言**。
3. **增加交互性和详细输出**。




