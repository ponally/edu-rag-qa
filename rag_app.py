from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from flask import Flask, request, jsonify

from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
INDEX_PATH = "faiss_index"

app = Flask(__name__)

# ---- 全局组件 ----
embeddings = None
vectorstore = None
qa_chain = None


# ---- 工具函数：根据来源加载文档 ----
def load_document(source: str):
    """
    支持三种输入来源：
    - .pdf 文件路径
    - http/https 网页 URL
    - 其他默认作为 txt 文件路径
    """
    if source.endswith(".pdf"):
        loader = PyPDFLoader(source)
    elif source.startswith("http://") or source.startswith("https://"):
        loader = WebBaseLoader(source)
    else:
        loader = TextLoader(source, encoding="utf-8")
    return loader.load()


# ---- 工具函数：构建或加载向量索引 ----
def build_or_load_vectorstore(source: str, force_rebuild: bool = False):
    """
    优先从磁盘加载已有索引，避免重复构建。
    force_rebuild=True 时强制重新构建（用于上传新文档后更新索引）。
    """
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


# ---- 工具函数：构建对话链 ----
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


# ==============================
# Flask API 接口
# ==============================

@app.route("/upload", methods=["POST"])
def upload():
    """
    上传文档并（重新）构建索引。
    请求体：{"source": "文件路径或URL"}
    """
    global qa_chain
    data = request.get_json()
    source = data.get("source", "course.txt")

    try:
        vs = build_or_load_vectorstore(source, force_rebuild=True)
        qa_chain = build_qa_chain(vs)
        return jsonify({"status": "ok", "message": f"文档 {source} 索引构建完成"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """
    提问接口，支持多轮对话。
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


@app.route("/reset", methods=["POST"])
def reset():
    """
    重置对话历史（不清除索引）。
    """
    global qa_chain
    if vectorstore is not None:
        qa_chain = build_qa_chain(vectorstore)
        return jsonify({"status": "ok", "message": "对话历史已重置"})
    return jsonify({"status": "error", "message": "索引尚未构建"}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "index_loaded": vectorstore is not None})


# ==============================
# 启动
# ==============================

if __name__ == "__main__":
    # 启动时自动加载默认文档（如已有索引则直接加载，无需重新构建）
    vs = build_or_load_vectorstore("course.txt")
    qa_chain = build_qa_chain(vs)
    print("\n=== 教育智能问答系统启动 ===")
    print("API 服务运行在 http://localhost:5000")
    print("接口列表：")
    print("  POST /upload  - 上传文档并构建索引")
    print("  POST /ask     - 提问（支持多轮对话）")
    print("  POST /reset   - 重置对话历史")
    print("  GET  /health  - 健康检查\n")
    app.run(host="0.0.0.0", port=5000, debug=False)