from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# ---- 配置 DeepSeek ----
API_KEY = "sk-3c051f3bc5e4411898221a3012259e80"
BASE_URL = "https://api.deepseek.com"

# ---- 1. 加载文档 ----
loader = TextLoader("course.txt", encoding="utf-8")
documents = loader.load()
print(f"文档加载完成，共 {len(documents)} 个文档")

# ---- 2. 文档分块 ----
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"文档分块完成，共 {len(chunks)} 个块")

# ---- 3. 本地 Embedding + 建索引 ----
print("正在加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
print("向量索引构建完成")

# ---- 4. 构建对话链 ----
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    api_key=API_KEY,
    base_url=BASE_URL
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    verbose=True
)

# ---- 5. 对话循环 ----
print("\n=== 教育智能问答系统启动 ===")
print("输入 'quit' 退出\n")

while True:
    question = input("你：")
    if question.lower() == "quit":
        break
    result = qa_chain.invoke({"question": question})
    print(f"助手：{result['answer']}\n")