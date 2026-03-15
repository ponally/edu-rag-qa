# 教育场景智能问答系统（RAG）

基于检索增强生成（RAG）技术构建的教育场景智能问答系统，
支持对课程文档进行精准问答，解决大模型幻觉与知识时效性问题。

## 技术栈
- LangChain：RAG 链路编排
- FAISS：向量索引与相似度检索
- HuggingFace Sentence Transformers：本地文本向量化
- DeepSeek API：大语言模型推理
- Flask：RESTful API 接口

## 系统架构
用户提问 → FAISS检索相关文档块 → 注入上下文 → DeepSeek生成回答

## 快速开始
1. 安装依赖：pip install -r requirements.txt
2. 配置 API Key：复制 .env.example 为 .env，填入你的 DeepSeek API Key
3. 运行：python rag_app.py