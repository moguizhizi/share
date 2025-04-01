本项目实现了一个基于 RAG（Retrieval-Augmented Generation）的 RL 引导代码助手，旨在通过检索相关代码片段并生成答案来解决用户关于代码的查询。
系统结合了检索、生成和强化学习（RL）优化三个核心模块，支持代码相关的任务。

**RAG 管道：**

*  数据集：使用 CodeSearchNet 数据集（Python 子集），从中提取了 1000 个函数的代码（func_code_string）和文档字符串（func_documentation_string）作为语料库。
*  检索：使用 SentenceTransformer（all-MiniLM-L6-v2 模型）将用户查询和代码片段编码为嵌入向量，计算余弦相似度，检索 Top-3 相关代码片段。
*  生成：使用 Salesforce/codegen-350M-mono 模型生成答案。将检索到的代码片段注入提示（prompt），生成 3 个候选答案。
*  GPU 加速：所有计算（嵌入、余弦相似度、生成）都在 GPU 上运行（如果可用），通过 torch.device 动态选择设备。
