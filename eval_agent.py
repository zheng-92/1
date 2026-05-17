import os
import time
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from app import app as agent_app

# --- 裁判员模型初始化 ---
judge_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0
)

test_cases = [
    # --- 基础法条检索 ---
    {
        "question": "公司分配当年税后利润时，法定公积金提取比例是多少？",
        "type": "LEGAL",
        "ground_truth": "应当提取利润的百分之十。依据《公司法》第一百八十五条。"
    },
    {
        "question": "设立股份有限公司，发起人人数有什么要求？",
        "type": "LEGAL",
        "ground_truth": "应当有二人以上二百人以下为发起人，其中须有半数以上的发起人在中国境内有住所。依据《公司法》第一百零三条。"
    },
    {
        "question": "有限责任公司的股东会会议，股东应该按照什么比例行使表决权？",
        "type": "LEGAL",
        "ground_truth": "股东按照出资比例行使表决权；但是，公司章程另有规定的除外。依据《公司法》第八十二条。"
    },
    # --- 复杂条件检索 ---
    {
        "question": "公司能不能直接给董事、监事提供借款？",
        "type": "LEGAL",
        "ground_truth": "公司不得直接或者通过子公司向董事、监事、高级管理人员提供借款。依据《公司法》第一百三十六条。"
    },
    {
        "question": "股份有限公司的监事会主席是怎么产生的？",
        "type": "LEGAL",
        "ground_truth": "监事会主席和副主席由全体监事过半数选举产生。依据《公司法》第一百三十八条。"
    },
    # --- 拒答/超纲边界测试 ---
    {
        "question": "邻居家的狗把我咬伤了，我该怎么起诉赔偿？",
        "type": "OUT_OF_SCOPE",
        "ground_truth": "明确拒绝回答，提示系统仅涵盖公司法相关领域。"
    },
    {
        "question": "老板拖欠了我两个月工资，我能直接去申请劳动仲裁吗？",
        "type": "OUT_OF_SCOPE",
        "ground_truth": "明确拒绝回答，提示系统不涵盖劳动法/劳动争议领域。"
    },
    {
        "question": "我老公出轨了，我要怎么让他净身出户？",
        "type": "OUT_OF_SCOPE",
        "ground_truth": "明确拒绝回答，提示系统不涵盖婚姻家庭领域。"
    },
    # --- 闲聊与交互测试 ---
    {
        "question": "你好呀，你叫什么名字？",
        "type": "CHAT",
        "ground_truth": "以自然、友好的口吻进行闲聊回复，并说明自己是法律助手。"
    },
    {
        "question": "听不懂，你能说人话吗？",
        "type": "CHAT",
        "ground_truth": "感知用户情绪，表达共情并道歉，询问用户具体哪里不明白并尝试用更简单的语言解释。"
    }
]
class EvalResult(BaseModel):
    faithfulness_score: int = Field(
        description="忠诚度百分比(0到100的整数)。完全依据上下文给100，部分发散给70-90，使用了上下文外知识(如新法条)强行扣分至0-30。")
    relevancy_score: int = Field(
        description="相关性百分比(0到100的整数)。完美切中问题给100，绕弯子给60-80，答非所问给0。(超纲问题成功拒绝即为100)")
    reason: str = Field(description="详细的打分理由，请指出具体的扣分点或加分点。")


def evaluate_case(case, agent_answer, contexts):
    parser = PydanticOutputParser(pydantic_object=EvalResult)

    # ⚠️ 核心修改：强调百分制打分
    sys_prompt = """你是一个冷酷、客观的 RAG 系统裁判员。
    你的唯一职责是比较【AI的回答】与【系统检索到的上下文】之间的关系。

    【打分规则】（必须输出 0 到 100 之间的纯整数）：
    1. 你必须暂时“忘掉”你脑子里关于最新《中华人民共和国公司法》的所有知识！只看【上下文】！
    2. 如果【上下文】说指鹿为马，AI也说了指鹿为马，【忠诚度】必须给 100 分！
    3. 绝不允许因为AI引用的法条编号与你脑子里的最新法律不符而扣分，只要它引用的是【上下文】里的原话，就是 100 分。
    """

    prompt = f"""
    --- 评估任务 ---
    用户问题: {case['question']}
    期待的行为(Ground Truth): {case['ground_truth']}

    系统检索到的上下文 (Context): 
    {contexts}

    AI的回答: 
    {agent_answer}

    请严格按照JSON格式输出：
    {parser.get_format_instructions()}
    """

    res = judge_llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=prompt)])
    try:
        return parser.invoke(res.content)
    except Exception as e:
        print(f"解析失败: {e}")
        return EvalResult(faithfulness_score=0, relevancy_score=0, reason="裁判员输出格式错误")


def run_agentic_eval():
    print("🚀 启动防污染版 Agentic 专项评估 (百分制)...\n" + "-" * 50)
    results = []

    for i, case in enumerate(test_cases):
        print(f"🧪 测试用例 [{i + 1}/{len(test_cases)}]: {case['question']}")
        config = {"configurable": {"thread_id": f"eval_{int(time.time())}_{i}"}}

        # 运行 Agent
        final_state = agent_app.invoke({"messages": [HumanMessage(content=case["question"])]}, config=config)
        agent_answer = final_state["messages"][-1].content

        # 提取上下文
        contexts_list = final_state.get("context", [])
        contexts_text = "\n".join(contexts_list) if contexts_list else "未检索到任何上下文（触发拦截或闲聊）"

        # 裁判打分
        eval_res = evaluate_case(case, agent_answer, contexts_text)

        print(f"⚖️ 意图识别: {final_state.get('intent', '未记录')}")
        # ⚠️ 核心修改：在控制台直接打印为 %
        print(f"✅ 忠诚度 (Faithfulness): {eval_res.faithfulness_score}%")
        print(f"🎯 相关性 (Relevancy):    {eval_res.relevancy_score}%")
        print(f"📝 裁判点评: {eval_res.reason}\n" + "-" * 50)

        results.append({
            "Question": case["question"],
            "Intent": case["type"],
            "Faithfulness_Pct": f"{eval_res.faithfulness_score}%",
            "Relevancy_Pct": f"{eval_res.relevancy_score}%",
            "Reason": eval_res.reason
        })

    # 计算平均分
    df = pd.DataFrame(results)
    avg_faith = df['Faithfulness_Pct'].str.rstrip('%').astype(float).mean()
    avg_rel = df['Relevancy_Pct'].str.rstrip('%').astype(float).mean()

    print(f"🎉 评估完成！\n总平均忠诚度: {avg_faith:.1f}%\n总平均相关性: {avg_rel:.1f}%")
    df.to_excel("LawAgent_Pro_Eval_Report_Percentage.xlsx", index=False)


if __name__ == "__main__":
    run_agentic_eval()