from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from flask import Flask, request, jsonify, Response, stream_with_context

from dotenv import load_dotenv
import os
import json

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
INDEX_PATH = "faiss_index"

app = Flask(__name__)

# ---- 全局组件 ----
embeddings = None
vectorstore = None
qa_chain = None
streaming_llm = None  # 专用于流式输出的LLM实例


# ---- 工具函数：根据来源加载文档 ----
def load_document(source: str):
    if source.endswith(".pdf"):
        loader = PyPDFLoader(source)
    elif source.startswith("http://") or source.startswith("https://"):
        loader = WebBaseLoader(source)
    else:
        loader = TextLoader(source, encoding="utf-8")
    return loader.load()


# ---- 工具函数：构建或加载向量索引 ----
def build_or_load_vectorstore(source: str, force_rebuild: bool = False):
    global embeddings, vectorstore

    if embeddings is None:
        print("正在加载 Embedding 模型...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("Embedding 模型加载完成")

    if os.path.exists(INDEX_PATH) and not force_rebuild:
        print("检测到已有索引，直接从磁盘加载...")
        vectorstore = FAISS.load_local(
            INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        print("索引加载完成")
    else:
        print(f"正在加载文档：{source}")
        documents = load_document(source)
        print(f"文档加载完成，共 {len(documents)} 个文档")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        print(f"文档分块完成，共 {len(chunks)} 个块")

        print("正在构建向量索引...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_PATH)
        print(f"向量索引构建完成，已持久化至 {INDEX_PATH}/")

    return vectorstore


# ---- 工具函数：构建对话链（非流式）----
def build_qa_chain(vs):
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.3,
        api_key=API_KEY,
        base_url=BASE_URL
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vs.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        verbose=False
    )
    return chain


# ---- 工具函数：构建流式LLM实例 ----
def build_streaming_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        temperature=0.3,
        streaming=True,
        api_key=API_KEY,
        base_url=BASE_URL
    )


# ==============================
# Flask API 接口
# ==============================

@app.route("/upload", methods=["POST"])
def upload():
    """
    上传文档并（重新）构建索引。
    请求体：{"source": "文件路径或URL"}
    """
    global qa_chain, streaming_llm
    data = request.get_json()
    source = data.get("source", "course.txt")

    try:
        vs = build_or_load_vectorstore(source, force_rebuild=True)
        qa_chain = build_qa_chain(vs)
        streaming_llm = build_streaming_llm()
        return jsonify({"status": "ok", "message": f"文档 {source} 索引构建完成"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """
    提问接口（非流式），支持多轮对话。
    请求体：{"question": "你的问题"}
    返回：{"answer": "模型回答"}
    """
    global qa_chain
    if qa_chain is None:
        return jsonify({"status": "error", "message": "请先调用 /upload 接口加载文档"}), 400

    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"status": "error", "message": "question 不能为空"}), 400

    try:
        result = qa_chain.invoke({"question": question})
        return jsonify({"status": "ok", "answer": result["answer"]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/ask_stream", methods=["POST"])
def ask_stream():
    """
    提问接口（流式），边生成边返回，使用 SSE 协议。
    请求体：{"question": "你的问题"}
    返回：text/event-stream，每条消息格式为 data: {"token": "..."}\n\n
         生成完毕后发送 data: {"done": true}\n\n

    实现说明：
    - 先用向量检索获取相关文档块（Top-3）
    - 构建包含检索内容的 Prompt
    - 用 streaming=True 的 LLM 流式生成，逐 token 通过 SSE 推送
    """
    global streaming_llm, vectorstore
    if streaming_llm is None or vectorstore is None:
        return jsonify({"status": "error", "message": "请先调用 /upload 接口加载文档"}), 400

    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"status": "error", "message": "question 不能为空"}), 400

    def generate():
        try:
            # 第一步：向量检索相关文档块
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])

            # 第二步：构建 Prompt
            prompt = f"""你是一个教育助手，请根据以下课程文档内容回答问题。
如果文档中没有相关信息，请直接说不知道，不要编造内容。

文档内容：
{context}

问题：{question}

请用中文回答："""

            # 第三步：流式生成，逐 token 推送
            for chunk in streaming_llm.stream(prompt):
                token = chunk.content
                if token:
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

            # 发送结束信号
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    """重置对话历史（不清除索引）。"""
    global qa_chain
    if vectorstore is not None:
        qa_chain = build_qa_chain(vectorstore)
        return jsonify({"status": "ok", "message": "对话历史已重置"})
    return jsonify({"status": "error", "message": "索引尚未构建"}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "index_loaded": vectorstore is not None,
        "streaming_supported": True
    })


# ==============================
# 启动
# ==============================

if __name__ == "__main__":
    vs = build_or_load_vectorstore("course.txt")
    qa_chain = build_qa_chain(vs)
    streaming_llm = build_streaming_llm()

    print("\n=== 教育智能问答系统启动 ===")
    print("API 服务运行在 http://localhost:5000")
    print("接口列表：")
    print("  POST /upload      - 上传文档并构建索引")
    print("  POST /ask         - 提问（非流式，支持多轮对话）")
    print("  POST /ask_stream  - 提问（流式，SSE 实时推送）")
    print("  POST /reset       - 重置对话历史")
    print("  GET  /health      - 健康检查\n")
    app.run(host="0.0.0.0", port=5000, debug=False)