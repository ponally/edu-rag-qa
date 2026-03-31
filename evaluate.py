import os
import traceback
import warnings
import logging
from dotenv import load_dotenv

# ---- 关闭已知无害 warning（需在 ragas.metrics 导入前执行）----
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"You are sending unauthenticated requests to the HF Hub\..*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"LangchainEmbeddingsWrapper is deprecated.*",
)

# HuggingFace/Transformers 这类提示很多是通过 logging 打印，不走 warnings.filterwarnings
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# 兼容不同 huggingface_hub 内部 logger 命名
for name in [
    "huggingface_hub.utils",
    "huggingface_hub.file_download",
    "huggingface_hub.utils._auth",
    "transformers.modeling_utils",
]:
    logging.getLogger(name).setLevel(logging.ERROR)

from ragas import evaluate
from ragas.metrics import Faithfulness, ContextRecall, AnswerRelevancy
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"

# 尽量降低 Transformers/HF 的运行时诊断输出
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()
except Exception:
    pass

try:
    from huggingface_hub.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

# ---- 关闭已知无害 warning ----
# 1) ragas.metrics 的 DeprecationWarning（当前 ragas 0.4.x 下可运行，但提示弃用）
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
)

# 2) HuggingFace 未鉴权访问提示：如果没有配置 HF_TOKEN 就屏蔽；如果配置了则先登录避免提示
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
    except Exception:
        # 登录失败时仍继续运行（后续会被 warning filter 屏蔽）
        pass
else:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"You are sending unauthenticated requests to the HF Hub\..*",
    )

# ---- 加载已有索引 ----
print("加载Embedding模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---- 配置LLM ----
from ragas.llms import llm_factory
from openai import OpenAI

# 用于生成答案
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=60,
    max_retries=2,
)

# 用于RAGAs评估
openai_client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=60,
    max_retries=2,
)
ragas_llm = llm_factory("deepseek-chat", client=openai_client)

ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)


def _score_to_float(value):
    if isinstance(value, list):
        if not value:
            raise ValueError("评估结果为空列表，无法计算分数。")
        return float(value[0])
    return float(value)

# ---- 测试集 ----
test_questions = [
    "如何礼貌地请求看菜单？",
    "点餐时如何表达想要和别人一样的菜？",
    "如何告知服务员自己对坚果过敏？",
    "牛排几分熟用英语怎么表达？",
    "I'd like和I want有什么区别？",
    "如何请求打包？",
    "Could I和Can I有什么区别？",
    "餐厅里tip是什么意思？",
    "如何询问一道菜是否辣？",
    "结账时应该说什么？"
]

ground_truths = [
    "Could I see the menu, please?",
    "I'll have the same.",
    "I'm allergic to nuts.",
    "rare是一分熟，medium是五分熟，well-done是全熟。",
    "I'd like比I want更礼貌正式，适合点餐场景。",
    "Can I get this to go?",
    "Could I比Can I更礼貌，适合正式餐厅。",
    "tip是小费的意思。",
    "Is this dish spicy?",
    "Excuse me, could we have the bill, please?"
]

# ---- 用RAG系统生成答案和检索上下文 ----
print("生成答案和检索上下文...", flush=True)
answers = []
contexts = []

for question in test_questions:
    # 检索相关文档
    docs = retriever.invoke(question)
    context = [doc.page_content for doc in docs]
    contexts.append(context)

    # 生成答案
    context_text = "\n\n".join(context)
    prompt = f"""你是一个教育助手，请根据以下课程文档内容回答问题。
如果文档中没有相关信息，请直接说不知道，不要编造内容。

文档内容：
{context_text}

问题：{question}

请用中文回答："""

    response = llm.invoke(prompt)
    answers.append(response.content)
    print(f"Q: {question}")
    print(f"A: {response.content[:50]}...")
    print()

# ---- 构建RAGAs数据集 ----
dataset = Dataset.from_dict({
    "user_input": test_questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths
})

# ---- 运行RAGAs评估 ----
print("运行RAGAs评估...", flush=True)
sample_dataset = Dataset.from_dict({
    "user_input": test_questions[:1],
    "response": answers[:1],
    "retrieved_contexts": contexts[:1],
    "reference": ground_truths[:1],
})

print("先进行1条样本评估，验证链路...", flush=True)
try:
    sample_result = evaluate(
        dataset=sample_dataset,
        metrics=[
            Faithfulness(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ],
        raise_exceptions=True,
        show_progress=False,
    )
    sample_answer_relevancy = evaluate(
        dataset=sample_dataset,
        metrics=[
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=2),
        ],
        raise_exceptions=True,
        show_progress=False,
    )
    print("1条样本评估完成：", {
        "faithfulness": sample_result["faithfulness"],
        "context_recall": sample_result["context_recall"],
        "answer_relevancy": sample_answer_relevancy["answer_relevancy"],
    })
except Exception as e:
    print("1条样本评估失败，详细错误如下：")
    print(repr(e))
    traceback.print_exc()
    raise

print("开始全量评估...", flush=True)
faithfulness_scores = []
context_recall_scores = []
answer_relevancy_scores = []

for i in range(len(test_questions)):
    single_dataset = Dataset.from_dict({
        "user_input": [test_questions[i]],
        "response": [answers[i]],
        "retrieved_contexts": [contexts[i]],
        "reference": [ground_truths[i]],
    })

    print(f"评估第{i + 1}/{len(test_questions)}条...")
    try:
        single_result = evaluate(
            dataset=single_dataset,
            metrics=[
                Faithfulness(llm=ragas_llm),
                ContextRecall(llm=ragas_llm),
            ],
            raise_exceptions=True,
            show_progress=False,
        )

        f_score = _score_to_float(single_result["faithfulness"])
        c_score = _score_to_float(single_result["context_recall"])
        single_answer_result = evaluate(
            dataset=single_dataset,
            metrics=[
                AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=2),
            ],
            raise_exceptions=True,
            show_progress=False,
        )
        a_score = _score_to_float(single_answer_result["answer_relevancy"])
        faithfulness_scores.append(f_score)
        context_recall_scores.append(c_score)
        answer_relevancy_scores.append(a_score)
        print(
            f"  faithfulness={f_score:.3f}, context_recall={c_score:.3f}, answer_relevancy={a_score:.3f}"
        )
    except Exception as e:
        print(f"  第{i + 1}条评估失败，已跳过。错误: {repr(e)}")
        continue

if not faithfulness_scores or not context_recall_scores or not answer_relevancy_scores:
    raise RuntimeError("所有样本都评估失败，无法计算平均分。")

result = {
    "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
    "context_recall": sum(context_recall_scores) / len(context_recall_scores),
    "answer_relevancy": sum(answer_relevancy_scores) / len(answer_relevancy_scores),
}

print("\n=== RAGAs评估结果 ===")
print(f"Faithfulness（忠实度）：{result['faithfulness']:.3f}")
print(f"Context Recall（上下文召回率）：{result['context_recall']:.3f}")
print(f"Answer Relevancy（答案相关性）：{result['answer_relevancy']:.3f}")
print(
    f"\n综合得分：{(result['faithfulness'] + result['context_recall'] + result['answer_relevancy']) / 3:.3f}"
)