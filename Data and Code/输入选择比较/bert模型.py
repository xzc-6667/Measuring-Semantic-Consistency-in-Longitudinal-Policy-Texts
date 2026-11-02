import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ========== 基础配置 ==========

KEYWORDS_DIR = r'D:\models\农业农村部一号文件'
KEYWORDS_FILENAME = '2005关键词.txt'
KEYWORDS_PATH = os.path.join(KEYWORDS_DIR, KEYWORDS_FILENAME)

EXCEL_PATH = r'D:\models\四川省委一号文件\2005四川省一号文件_分句结果.xlsx'
OUTPUT_EXCEL_PATH = r'D:\models\四川省委一号文件\2005结果\2005四川省一号文件_bert关键词匹配结果_专家关键词.xlsx'

LOCAL_MODEL_PATH = r'D:\models\bert\bert-base-chinese'
SENTENCE_COLUMN = '句子内容'


# ========== 加载本地模型 ==========
print(f"🔍 加载本地中文BERT模型：{LOCAL_MODEL_PATH}")
if not os.path.exists(LOCAL_MODEL_PATH):
    raise FileNotFoundError(f"本地模型路径不存在：{LOCAL_MODEL_PATH}")

model = SentenceTransformer(LOCAL_MODEL_PATH)
print("✅ 本地模型加载成功！")


# ========== 读取关键词 ==========
print(f"\n📁 读取关键词文件：{KEYWORDS_PATH}")

try:
    with open(KEYWORDS_PATH, 'r', encoding='gbk') as f:
        KEYWORDS = [line.strip() for line in f if line.strip()]
except UnicodeDecodeError:
    with open(KEYWORDS_PATH, 'r', encoding='gb2312') as f:
        KEYWORDS = [line.strip() for line in f if line.strip()]

if len(KEYWORDS) == 0:
    raise ValueError("关键词文件为空！")

print(f"✅ 加载关键词 {len(KEYWORDS)} 个：{KEYWORDS[:5]}...")

keyword_vecs = model.encode(KEYWORDS, convert_to_tensor=True).cpu().numpy()
print("✅ 关键词向量计算完成")


# ========== 读取 Excel ==========
print("\n📊 读取 Excel 分句结果...")
if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError(f"Excel 文件不存在：{EXCEL_PATH}")

df = pd.read_excel(EXCEL_PATH)

if SENTENCE_COLUMN not in df.columns:
    raise KeyError(f"Excel 中未找到列名：{SENTENCE_COLUMN}")

# ---- 保留原始行号（关键修复） ----
df_clean = df.dropna(subset=[SENTENCE_COLUMN]).copy()
df_clean = df_clean[df_clean[SENTENCE_COLUMN].str.strip() != ''].copy()
df_clean = df_clean.reset_index(drop=False)   # ⭐ 保留原始行号 old index

sentences = df_clean[SENTENCE_COLUMN].tolist()
total_sentences = len(sentences)
print(f"✅ 有效句子：{total_sentences} 条")


# ========== 上下文 + 相似度计算（不会错位） ==========
def calculate_sentence_similarity(row_idx, df, keyword_vecs, model):
    """
    row_idx: df_clean 中的行号（从 0 开始）
    """

    start_idx = max(0, row_idx - 2)
    end_idx = min(len(df) - 1, row_idx + 2)

    # ⭐ 不再用 list，而是直接用 DataFrame — 不丢句、不重复、顺序完全一致
    context_sentences = df.loc[start_idx:end_idx, SENTENCE_COLUMN].tolist()
    context_text = "。".join(context_sentences)

    # 相似度：取最大值
    context_vec = model.encode(context_text, convert_to_tensor=True).cpu().numpy()
    similarities = cosine_similarity([context_vec], keyword_vecs)[0]
    max_similarity = float(np.max(similarities))

    return context_text, round(max_similarity, 4)


# ========== 逐句处理 ==========
print("\n🧠 开始逐句计算相似度...")

context_list = []
similarity_list = []

for idx in range(total_sentences):
    if (idx + 1) % 10 == 0:
        print(f"🔄 已处理 {idx + 1}/{total_sentences} 句")

    context_text, sim = calculate_sentence_similarity(
        row_idx=idx,
        df=df_clean,
        keyword_vecs=keyword_vecs,
        model=model
    )

    context_list.append(context_text)
    similarity_list.append(sim)

# 验证长度
assert len(context_list) == total_sentences
assert len(similarity_list) == total_sentences

print("✅ 上下文与相似度长度完全对应，无重复、无缺失")


# ========== 保存 Excel ==========
print("\n💾 保存结果中...")

df_clean['上下文片段'] = context_list
df_clean['与阶段五关键词相似度（最大值）'] = similarity_list

df_clean.to_excel(OUTPUT_EXCEL_PATH, index=False, engine='openpyxl')

print(f"🎉 结果已保存：{OUTPUT_EXCEL_PATH}")


# ========== 结果统计 ==========
print("\n📈 结果统计：")
print(f"关键词数量：{len(KEYWORDS)}")
print(f"句子数量：{total_sentences}")
print(f"平均相似度：{np.mean(similarity_list):.4f}")
print(f"最高相似度：{np.max(similarity_list):.4f}")
print(f"最低相似度：{np.min(similarity_list):.4f}")
