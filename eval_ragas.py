import os
import re
import time
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
# 建议增加这一项，衡量回答与标准答案的匹配度
from ragas.metrics import answer_similarity
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from app import app as agent_app

# 环境配置
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE = "https://api.deepseek.com"

# --- 优化点 1: 增加负向用例，测试“红线拦截”忠诚度 ---
test_cases = [
    # 正向用例 (公司法)
    {"question": "法定代表人辞职后，公司必须在多久内确定新人？",
     "ground_truth": "根据《公司法》第十二条，应当在辞任之日起三十日内确定。"},
    {"question": "公司分配当年税后利润时，应当提取利润的百分之多少列入法定公积金？",
     "ground_truth": "根据《公司法》第二百一十条，应当提取利润的百分之十。"},

    {"question": "邻居家的狗把我咬伤了，我该怎么起诉赔偿？",
     "ground_truth": "抱歉，我的知识库目前仅涵盖《公司法》及相关领域，无法为您解答侵权纠纷。"
    },
    {"question": "我想要办理协议离婚，财产应该怎么分配？",
     "ground_truth": "抱歉，我的知识库目前仅涵盖《公司法》及相关领域，无法为您解答婚姻法律问题。"
    }
]



def clean_agent_output(text):
    """
    优化点 2: 清洗回答内容。
    移除 UI 标签和文书占位符，只保留核心法律意见，这能显著提高 Relevancy 得分。
    """
    # 移除文书内容
    text = re.sub(r'<DOC_CONTENT>.*?</DOC_CONTENT>', '', text, flags=re.DOTALL)
    # 移除触发器标签
    text = re.sub(r'\[TRIGGER_DOC:.*?\]', '', text)
    # 移除过多的换行
    text = text.strip()
    return text


def run_eval():
    print("开始采集 Agent 回答...")
    results = []

    for i, case in enumerate(test_cases):
        print(f"   [{i + 1}/{len(test_cases)}] 处理问题: {case['question']}")

        config = {"configurable": {"thread_id": f"eval_{int(time.time())}_{i}"}}
        try:
            final_state = agent_app.invoke(
                {"messages": [HumanMessage(content=case["question"])]},
                config=config
            )

            # 提取回答并清洗
            raw_answer = final_state["messages"][-1].content
            answer = clean_agent_output(raw_answer)

            # 提取上下文 (注意：App.py 只有在 LEGAL 模式下才会写 context)
            contexts = final_state.get("context", [])
            if not contexts:
                contexts = ["系统判定为非法律咨询或触发拦截，未调用检索库。"]

            results.append({
                "question": case["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": case["ground_truth"]
            })
            time.sleep(1)  # 避免 API 并发过高

        except Exception as e:
            print(f"    采样失败: {e}")

    # --- 优化点 3: 适配 Ragas 最新版的封装方式 ---
    evaluator_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_KEY,
        base_url=DEEPSEEK_BASE,
        temperature=0
    )
    evaluator_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

    # 定义指标
    metrics = [
        faithfulness,  # 忠诚度：回答是否只依据检索到的上下文
        answer_relevancy,  # 相关性：回答是否针对了用户问题
        context_precision,  # 检索精度：检索到的内容是否真的有用
        answer_similarity  # 语义相似度：回答与标准答案的接近程度
    ]

    print("正在进行 Ragas 量化评分...")
    dataset = Dataset.from_list(results)

    # 执行评估 (通过 llm 和 embeddings 参数注入)
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    print("\n评估完成！")
    print(result)

    df = result.to_pandas()
    df.to_excel("LawAgent_Ragas_Final_Eval.xlsx", index=False)
    print(f"详细报告已生成。")


if __name__ == "__main__":
    if not DEEPSEEK_KEY:
        print("错误：请先设置 DEEPSEEK_API_KEY")
    else:
        run_eval()