import pandas as pd
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics.pairwise import cosine_similarity

# ========== 基础配置 ==========
KEYWORDS_DIR = r'D:\models\农业农村部一号文件'
KEYWORDS_FILENAME = '2005关键词.txt'
KEYWORDS_PATH = os.path.join(KEYWORDS_DIR, KEYWORDS_FILENAME)

EXCEL_PATH = r'D:\models\四川省委一号文件\2005四川省一号文件_分句结果.xlsx'
OUTPUT_EXCEL_PATH = r'D:\models\四川省委一号文件\2005结果\2005四川省一号文件_bge关键词匹配结果.xlsx'
LOCAL_MODEL_PATH = r'D:\models\bge-base-zh-v1.5'   # ← 你的 BGE 本地模型目录
SENTENCE_COLUMN = '句子内容'

# ========== 1. 使用 SentenceTransformer 加载本地 BGE 模型 ==========

print(f"🔍 使用 SentenceTransformer 加载本地 BGE 模型：{LOCAL_MODEL_PATH}")

word_embedding_model = models.Transformer(
    model_name_or_path=LOCAL_MODEL_PATH,
    max_seq_length=512
)

pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

print("✅ 本地 BGE 模型加载成功！")

# ========== 2. 加载关键词（处理编码） ==========
print(f"\n📁 读取关键词文件：{KEYWORDS_PATH}")

if not os.path.exists(KEYWORDS_PATH):
    raise FileNotFoundError(f"关键词文件不存在：{KEYWORDS_PATH}")

try:
    with open(KEYWORDS_PATH, 'r', encoding='gbk') as f:
        KEYWORDS = [line.strip() for line in f if line.strip()]
except UnicodeDecodeError:
    with open(KEYWORDS_PATH, 'r', encoding='utf-8') as f:
        KEYWORDS = [line.strip() for line in f if line.strip()]

print(f"✅ 关键词数量：{len(KEYWORDS)} 个 → 示例：{KEYWORDS[:5]}")

# 关键词向量
keyword_vecs = model.encode(KEYWORDS, convert_to_tensor=True).cpu().numpy()
print("✅ 关键词向量计算完成")

# ========== 3. 读取 Excel ==========
print("\n📊 读取 Excel 分句文件...")

df = pd.read_excel(EXCEL_PATH)

if SENTENCE_COLUMN not in df.columns:
    raise KeyError(f"Excel中未找到列：{SENTENCE_COLUMN}")

df_clean = df.dropna(subset=[SENTENCE_COLUMN]).copy()
df_clean = df_clean[df_clean[SENTENCE_COLUMN].str.strip() != ''].copy()
sentences = df_clean[SENTENCE_COLUMN].tolist()
total_sentences = len(sentences)

print(f"✅ 有效句子：{total_sentences} 条")

# ========== 4. 相似度计算 ==========

def calculate_sentence_similarity(sentence_idx, all_sentences, keyword_vecs, model):
    """上下文：前2句 + 当前句 + 后2句，取最大相似度"""

    total = len(all_sentences)
    start_idx = max(0, sentence_idx - 2)
    end_idx = min(total - 1, sentence_idx + 2)

    context_sentences = all_sentences[start_idx:end_idx + 1]
    context_text = "。".join(context_sentences)

    context_vec = model.encode(context_text, convert_to_tensor=True).cpu().numpy()
    similarities = cosine_similarity([context_vec], keyword_vecs)[0]

    return context_text, float(np.max(similarities))

print("\n🧠 正在计算相似度...")

context_list = []
similarity_list = []

for idx in range(total_sentences):
    if (idx + 1) % 10 == 0:
        print(f"🔄 已处理 {idx + 1}/{total_sentences}")

    context_text, sim = calculate_sentence_similarity(
        sentence_idx=idx,
        all_sentences=sentences,
        keyword_vecs=keyword_vecs,
        model=model
    )

    context_list.append(context_text)
    similarity_list.append(round(sim, 4))

print("✅ 相似度计算完成")

# ========== 5. 保存结果 ==========
df_clean['上下文片段'] = context_list
df_clean['与阶段五关键词相似度（最大值）'] = similarity_list

df_clean.to_excel(OUTPUT_EXCEL_PATH, index=False, engine='openpyxl')
print(f"💾 结果保存到：{OUTPUT_EXCEL_PATH}")

# ========== 6. 简单统计 ==========
print("\n📈 统计结果：")
print(f"平均相似度：{np.mean(similarity_list):.4f}")
print(f"最大：{np.max(similarity_list):.4f}")
print(f"最小：{np.min(similarity_list):.4f}")
