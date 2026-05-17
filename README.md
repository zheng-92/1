LawAgent Pro: 基于多智能体协作的专家级法律助手
LawAgent Pro 是一款专为《中华人民共和国公司法》（2023修订版）设计的深度 RAG（检索增强生成）法律助手。它通过 LangGraph 状态机 实现了从意图识别到多专家协同推理，最后到自动化文书起草的完整法律咨询闭环。
# 核心亮点
1. 基于 LangGraph 的状态机多智能体编排
采用 LangGraph 构建有状态的循环图架构：
  动态规划器 (Planner)：将复杂的法律诉求拆解为 [Evidence]（法条溯源）与 [Strategy]（实战建模）双维度任务。
  并行节点调度 (Parallel Map-Reduce)：通过条件边（Conditional Edges）实现 Evidence Agent 与 Strategy Agent 的并发执行，利用 多线程/并发机制 显著降低端到端延迟（TTFT）。
  自愈式执行器 (Executor)：强制采用 Pydantic 结构化输出，对各专家节点的建议进行逻辑聚合，并触发文书生成 Skill。
2. 行业级 RAG 检索链路优化
针对法律文本极其严谨的特性，构建了高精度的检索管道：
  基于法条结构的语义切分 (Semantic Regex Chunking)：build_vector_db.py 采用自定义正则表达式，精确识别“第x条”结构进行物理切片，有效避免了传统固定长度切片导致的法条语义断裂。
  混合检索策略 (Hybrid Search)：结合 ChromaDB 向量索引（语义关联）与 BM25 算法（精确关键词/法条编号匹配），确保召回的广度与精度。
  深度重排 (Cross-Encoder Re-ranking)：集成 BAAI/bge-reranker-v2-m3 模型，对 Top-20 候选集进行二次语义重分，彻底解决 RAG 系统的“检索噪声”干扰。
3. 长短期混合记忆与上下文重写
  增量式备忘录 (Summarizer)：自动维护《法律案件备忘录》，动态提取持股比例、争议金额、对话阶段等核心事实。
  语境无关重写 (Query Rewriter)：利用 LLM 根据历史记忆自动补全用户提问中的代词（如将“他没给钱”重写为“股东张三未按期足额缴纳 500 万出资款”），大幅提升检索命中率。
4. 生产级评估框架 (Judge-LLM)
  系统内置了独立的 Agentic Eval 脚本（eval_agent.py），采用“裁判员模型”机制：
  Faithfulness（忠诚度）：强制对比回答与检索到的 Context，严厉拦截“幻觉”输出。
  Relevancy（相关度）：多维打分确保建议切中用户核心诉求。
  量化报告：自动输出百分制 Excel 报告，为工程迭代提供数据支撑。
# 技术架构
graph TD
    User((用户输入)) --> Summarize[对话摘要/记忆更新]
    Summarize --> Router{意图识别路由}
    Router -- LEGAL --> Rewriter[Query 重写优化]
    Rewriter --> Planner[多任务并发规划]
    Planner --> Evidence[Evidence Agent: 法条查证]
    Planner --> Strategy[Strategy Agent: 实战建议]
    Evidence --> Rerank[BGE Reranker 二次重排]
    Strategy --> Rerank
    Rerank --> Executor[结构化推理响应]
    Executor --> WordGen[python-docx 文书生成]
    Router -- CHAT --> Executor
    Executor --> UI[Streamlit 流式交互输出]
# 项目目录结构
LawAgent_Project/
├── law_interpret.txt    # 语料库：2023最高法司法解释
├── law_main.txt         # 语料库：2023公司法正文
├── law_chroma_db/       # 向量数据库存储目录
├── app.py               # 核心主程序 (Streamlit UI + LangGraph 并发重排逻辑)
├── build_vector_db.py   # RAG 预处理：语义正则切片 + 向量库构建
├── eval_agent.py        # 离线评估：基于 Judge-LLM 的双维度量化打分
├── requirements.txt     # 环境依赖清单
├── .env                 # 环境变量 (存储 API Key)
└── README.md            # 项目说明文档

# 快速开始
1. 环境准备
建议使用 Python 3.9+。
Bash
  pip install -r requirements.txt
2. 配置 API Key
在根目录创建 .env 文件：
  DEEPSEEK_API_KEY=sk-你的key
3. 运行项目
Bash
  # 构建/更新向量数据库
  python build_vector_db.py

  # 启动 Streamlit Web 界面
  streamlit run app.py


# 技术栈清单
LLM: DeepSeek-Chat (V3)
Orchestration: LangGraph, LangChain
Retrieval: ChromaDB, BM25, Cross-Encoder (Rerank)
Embedding: BAAI/bge-large-zh-v1.5
Engineering: SQLite (In-graph Checkpointing), Pydantic (Schema Validation)
UI/Export: Streamlit, python-docx