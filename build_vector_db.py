# build_vector_db.py
import re
from langchain_core.documents import Document
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 环境配置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = r'D:\zjq\python\RAG\models_cache'


def load_and_tag(file_path, doc_type):
    """
    亮点：读取文件并注入元数据标签，采用基于法条结构的语义切分 (Semantic Chunking)
    """
    if not os.path.exists(file_path):
        print(f"⚠️ 警告：文件 {file_path} 不存在，跳过。")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    documents = []

    # 核心优化：使用正则表达式，按“第xxx条”进行切分
    # 匹配逻辑：找到“第x条”，并向后一直匹配，直到遇到下一个“第x条”或文本结束
    pattern = re.compile(r'(第[一二三四五六七八九十百千]+条[\s\S]*?)(?=\n第[一二三四五六七八九十百千]+条|$)',
                         re.MULTILINE)
    matches = pattern.findall(content)

    if matches:
        for match in matches:
            text = match.strip()
            # 过滤掉太短的无意义文本
            if len(text) > 10:
                documents.append(Document(page_content=text, metadata={"type": doc_type, "source": file_path}))
    else:
        # 降级策略：如果没有匹配到“第X条”（可能是一些说明性文字），则按双换行（段落）切分
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 10]
        for p in paragraphs:
            documents.append(Document(page_content=p, metadata={"type": doc_type, "source": file_path}))

    print(f"✅ 已加载 {file_path}，共 {len(documents)} 条记录。")
    return documents


def build_database():
    db_dir = "./law_chroma_db"

    # 检查并清理旧数据库
    if os.path.exists(db_dir):
        print(f"🧹 正在清理旧数据库：{db_dir}")
        shutil.rmtree(db_dir)

    print("🚀 正在构建多维法律知识库...")

    # 加载带标签的数据
    all_docs = []
    all_docs += load_and_tag("law_main.txt", "法律正文")
    all_docs += load_and_tag("law_interpret.txt", "司法解释")

    if not all_docs:
        print("❌ 错误：没有找到任何可用的法律文本，请检查文件名！")
        return

    # 初始化 Embedding 模型
    print("🧠 正在加载 Embedding 模型 (BGE)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={'device': 'cpu'}
    )

    # 构建并持久化向量库
    print("📦 正在写入 Chroma 向量库...")
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=db_dir
    )
    print(f"✨ 成功！高级法律知识库已构建完成，保存在 {db_dir}")


if __name__ == "__main__":
    build_database()