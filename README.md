# 教育场景智能问答系统（RAG）

基于检索增强生成（RAG）技术构建的教育场景智能问答系统，支持对课程文档进行精准多轮问答，解决大模型幻觉与知识时效性问题。

## 技术栈

| 组件 | 技术 |
|------|------|
| RAG 链路编排 | LangChain |
| 向量索引与检索 | FAISS |
| 文本向量化 | Sentence-Transformers（本地推理） |
| 大语言模型 | DeepSeek API |
| API 服务 | Flask |

## 系统架构

```
文档输入（txt / PDF / URL）
        ↓
   文档加载 & 分块
   RecursiveCharacterTextSplitter
   chunk_size=500, overlap=50
        ↓
   本地向量化
   paraphrase-multilingual-MiniLM-L12-v2
        ↓
   FAISS 向量索引（持久化至磁盘）
        ↓
用户提问 → query 向量化 → Top-3 相似块检索
        ↓
   注入上下文 + 对话历史
   ConversationalRetrievalChain
        ↓
   DeepSeek 生成回答
        ↓
   非流式（/ask）或流式 SSE（/ask_stream）返回
```

## 核心特性

- **多格式文档支持**：支持 txt、PDF、网页 URL 三种输入来源
- **向量索引持久化**：首次构建后将索引保存至磁盘，后续启动直接加载，避免重复计算
- **多轮对话**：基于 ConversationBufferMemory 维护对话历史；ConversationalRetrievalChain 自动对含指代词的 query 结合历史做改写再检索
- **流式输出**：新增 `/ask_stream` 接口，基于 SSE（Server-Sent Events）协议逐 token 实时推送，用户体验类似 ChatGPT 打字机效果
- **RESTful API**：Flask 封装完整接口，支持文档上传、问答（非流式/流式）、会话重置

## API 接口

### POST `/upload`
上传文档并（重新）构建向量索引。
请求体：`{"source": "course.txt"}`，支持本地 .txt / .pdf 或网页 URL。

### POST `/ask`
提问接口（非流式），支持多轮对话。
请求体：`{"question": "你的问题"}`
返回：`{"status": "ok", "answer": "模型回答"}`

### POST `/ask_stream`
提问接口（流式），SSE 协议逐 token 推送。
请求体：`{"question": "你的问题"}`
返回 SSE 流，每条格式为 `data: {"token": "..."}\n\n`，结束时发送 `data: {"done": true}\n\n`

### POST `/reset`
重置当前会话的对话历史（不清除向量索引）。

### GET `/health`
服务健康检查，返回索引加载状态和流式支持状态。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（复制 .env.example 为 .env，填入 DeepSeek API Key）
cp .env.example .env

# 3. 启动服务
python rag_app.py
```

## 已知局限与优化方向

| 局限 | 优化方案 |
|------|----------|
| 通用 Embedding 模型对专业领域术语理解有限 | 替换为领域适配的 Embedding 模型 |
| 仅单路向量检索，可能漏召回相关文档 | 引入多路召回（向量检索 + BM25），加 Reranker 重排序 |
| ConversationBufferMemory 完整保存历史，长对话 Prompt 过长 | 换用 ConversationSummaryMemory 压缩历史 |
| 缺乏答案质量评估体系 | 引入 RAGAs 框架量化 Faithfulness、Answer Relevancy、Context Recall |
| 流式接口暂不支持多轮对话历史 | 将对话历史管理与流式生成结合，实现完整流式多轮对话 |