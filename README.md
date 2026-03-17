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
      返回答案
```

## 核心特性

- **多格式文档支持**：支持 txt、PDF、网页 URL 三种输入来源
- **向量索引持久化**：首次构建后将索引保存至磁盘，后续启动直接加载，避免重复计算
- **多轮对话**：基于 ConversationBufferMemory 维护对话历史，支持上下文连续问答；ConversationalRetrievalChain 自动对含指代词的 query 做改写再检索
- **RESTful API**：Flask 封装完整接口，支持文档上传、问答、会话重置

## API 接口

### POST `/upload`
上传文档并（重新）构建向量索引。

请求体：
```json
{"source": "course.txt"}
```
`source` 可以是本地文件路径（.txt / .pdf）或网页 URL。

返回：
```json
{"status": "ok", "message": "文档 course.txt 索引构建完成"}
```

---

### POST `/ask`
提问接口，支持多轮对话。

请求体：
```json
{"question": "什么是向量数据库？"}
```

返回：
```json
{"status": "ok", "answer": "向量数据库是..."}
```

---

### POST `/reset`
重置当前会话的对话历史（不清除向量索引）。

---

### GET `/health`
服务健康检查，返回索引加载状态。

## 快速开始

**1. 安装依赖**
```bash
pip install -r requirements.txt
```

**2. 配置 API Key**
```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
# DEEPSEEK_API_KEY=your_key_here
```

**3. 启动服务**
```bash
python rag_app.py
```
服务启动后默认加载 `course.txt`，若已有持久化索引则直接加载，无需重新构建。

**4. 调用示例**
```bash
# 上传新文档
curl -X POST http://localhost:5000/upload \
  -H "Content-Type: application/json" \
  -d '{"source": "your_doc.pdf"}'

# 提问
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "请介绍一下课程的主要内容"}'

# 重置对话
curl -X POST http://localhost:5000/reset
```

## 已知局限与优化方向

| 局限 | 优化方案 |
|------|----------|
| 通用 Embedding 模型对专业领域术语理解有限 | 替换为领域适配的 Embedding 模型 |
| 仅单路向量检索，可能漏召回相关文档 | 引入多路召回（向量检索 + BM25），加 Reranker 重排序 |
| ConversationBufferMemory 完整保存历史，长对话 Prompt 过长 | 换用 ConversationSummaryMemory 压缩历史 |
| 缺乏答案质量评估体系 | 引入 RAGAs 框架量化 Faithfulness、Answer Relevancy、Context Recall |