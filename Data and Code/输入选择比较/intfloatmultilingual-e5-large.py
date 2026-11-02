
import pandas as pd
import os
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ========== 基础配置 ==========
KEYWORDS_DIR = r'D:\models\农业农村部一号文件'
KEYWORDS_FILENAME = '2005关键词.txt'
KEYWORDS_PATH = os.path.join(KEYWORDS_DIR, KEYWORDS_FILENAME)

EXCEL_PATH = r'D:\models\四川省委一号文件\2005四川省一号文件_分句结果.xlsx'
OUTPUT_EXCEL_PATH = r'D:\models\四川省委一号文件\2005结果\2005四川省一号文件_E5关键词匹配结果_专家关键词.xlsx'
LOCAL_MODEL_PATH = r'D:\models\intfloatmultilingual-e5-large'
SENTENCE_COLUMN = '句子内容'


# ========== 工具函数：净化关键词（核心新增） ==========
def clean_keyword(kw):
    """
    净化关键词：去除隐形无效字符、控制字符、非中文字符（保留纯中文）
    """
    # 去除控制字符（如换行、制表符、不可见字符）
    kw = re.sub(r'[\x00-\x1F\x7F]', '', kw)
    # 去除全角/半角空格
    kw = kw.strip().replace('　', '').replace(' ', '')
    # 只保留中文字符（过滤特殊符号、英文、数字等）
    kw = re.sub(r'[^\u4e00-\u9fa5]', '', kw)
    return kw


# ========== 加载本地E5模型 ==========
print(f"🔍 加载本地E5多语言模型：{LOCAL_MODEL_PATH}")
if not os.path.exists(LOCAL_MODEL_PATH):
    raise FileNotFoundError(f"本地模型路径不存在：{LOCAL_MODEL_PATH}")

model = SentenceTransformer(LOCAL_MODEL_PATH)
print("✅ E5模型加载成功！")

# ========== 读取+净化关键词（核心修复） ==========
print(f"\n📁 读取关键词文件：{KEYWORDS_PATH}")

try:
    with open(KEYWORDS_PATH, 'r', encoding='gbk') as f:
        content = f.read().strip()
        # 按逗号分割 → 初步清洗
        raw_keywords = [kw.strip() for kw in content.split(',') if kw.strip()]
except UnicodeDecodeError:
    with open(KEYWORDS_PATH, 'r', encoding='gb2312') as f:
        content = f.read().strip()
        raw_keywords = [kw.strip() for kw in content.split(',') if kw.strip()]

# 净化关键词 + 过滤净化后为空的关键词
KEYWORDS = []
for kw in raw_keywords:
    cleaned_kw = clean_keyword(kw)
    if cleaned_kw:  # 只保留净化后非空的关键词
        KEYWORDS.append(cleaned_kw)

if len(KEYWORDS) == 0:
    raise ValueError("关键词文件为空或净化后无有效关键词！请检查关键词是否为纯中文")

print(f"✅ 原始关键词 {len(raw_keywords)} 个，净化后有效关键词 {len(KEYWORDS)} 个")
print(f"净化后关键词：{KEYWORDS}")  # 显示所有净化后的关键词，方便检查

# ========== 关键词编码（单独编码+NaN容错） ==========
print("\n🧮 计算关键词向量（单独编码+NaN过滤）...")
keyword_with_prefix = [f"query: {kw}" for kw in KEYWORDS]
keyword_vecs = []
valid_keywords = []  # 记录编码成功的关键词

for idx, (kw, kw_with_prefix) in enumerate(zip(KEYWORDS, keyword_with_prefix)):
    try:
        # 单独编码每个关键词（避免批量编码时单个异常影响整体）
        vec = model.encode(
            kw_with_prefix,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=1  # 单关键词编码，提高稳定性
        ).cpu().numpy()

        # 检查是否为NaN向量
        if np.isnan(vec).any():
            print(f"⚠️  关键词「{kw}」编码后为NaN向量，已过滤")
            continue

        keyword_vecs.append(vec)
        valid_keywords.append(kw)
    except Exception as e:
        print(f"⚠️  关键词「{kw}」编码失败（错误：{str(e)}），已过滤")
        continue

# 转换为numpy数组（必须确保有有效向量）
if len(keyword_vecs) == 0:
    raise ValueError("所有关键词编码失败！请检查关键词内容或模型文件是否损坏")

keyword_vecs = np.array(keyword_vecs)
print(f"✅ 编码成功的关键词：{len(valid_keywords)} 个")
print(f"编码成功的关键词列表：{valid_keywords}")

# ========== 读取 Excel + 加强数据清洗 ==========
print("\n📊 读取 Excel 分句结果 + 加强数据清洗...")
if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError(f"Excel 文件不存在：{EXCEL_PATH}")

df = pd.read_excel(EXCEL_PATH)

if SENTENCE_COLUMN not in df.columns:
    raise KeyError(f"Excel 中未找到列名：{SENTENCE_COLUMN}")

df_clean = df.copy()
df_clean[SENTENCE_COLUMN] = df_clean[SENTENCE_COLUMN].astype(str)  # 统一转为字符串
df_clean = df_clean[df_clean[SENTENCE_COLUMN].str.strip().notna()]
df_clean = df_clean[df_clean[SENTENCE_COLUMN].str.strip() != '']
df_clean = df_clean.reset_index(drop=False)  # 保留原始行号

sentences = df_clean[SENTENCE_COLUMN].tolist()
total_sentences = len(sentences)

if total_sentences == 0:
    raise ValueError("清洗后无有效句子，请检查Excel数据！")

print(f"✅ 有效句子：{total_sentences} 条（已过滤所有空值/纯空格）")


# ========== 上下文 + 相似度计算（容错优化） ==========
def calculate_sentence_similarity(row_idx, df, keyword_vecs, model):
    start_idx = max(0, row_idx - 2)
    end_idx = min(len(df) - 1, row_idx + 2)

    context_sentences = df.loc[start_idx:end_idx, SENTENCE_COLUMN].tolist()
    context_text = "。".join(context_sentences).strip()

    if not context_text:
        print(f"⚠️  第{row_idx}句的上下文为空，相似度设为0.0")
        return "", 0.0

    context_with_prefix = f"passage: {context_text}"
    try:
        context_vec = model.encode(
            context_with_prefix,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=1
        ).cpu().numpy()

        if np.isnan(context_vec).any():
            print(f"⚠️  第{row_idx}句的上下文编码后存在NaN向量，相似度设为0.0")
            return context_text, 0.0

        # 计算最大相似度
        similarities = cosine_similarity([context_vec], keyword_vecs)[0]
        max_similarity = float(np.max(similarities))
        return context_text, round(max_similarity, 4)
    except Exception as e:
        print(f"⚠️  第{row_idx}句计算失败（错误：{str(e)}），相似度设为0.0")
        return context_text, 0.0


# ========== 逐句处理 ==========
print("\n🧠 开始逐句计算E5相似度（含上下2句上下文）...")

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

# 校验结果长度
assert len(context_list) == total_sentences
assert len(similarity_list) == total_sentences
print("✅ 上下文与相似度计算完成（无缺失）")

# ========== 保存 Excel ==========
print("\n💾 保存结果中...")

df_clean['上下文片段'] = context_list
df_clean['与阶段五关键词相似度（最大值）'] = similarity_list

# 新增：保存编码成功的关键词列表到Excel（方便核对）
with pd.ExcelWriter(OUTPUT_EXCEL_PATH, engine='openpyxl') as writer:
    df_clean.to_excel(writer, sheet_name='匹配结果', index=False)
    # 新建sheet存储有效关键词
    pd.DataFrame({'编码成功的关键词': valid_keywords}).to_excel(writer, sheet_name='有效关键词', index=False)

print(f"🎉 结果已保存：{OUTPUT_EXCEL_PATH}")

# ========== 结果统计（优化） ==========
print("\n📈 结果统计（E5模型）：")
print(f"原始关键词数量：{len(raw_keywords)}")
print(f"净化后有效关键词数量：{len(KEYWORDS)}")
print(f"编码成功的关键词数量：{len(valid_keywords)}")
print(f"有效句子数量：{total_sentences}")
print(f"平均相似度：{np.mean(similarity_list):.4f}")
print(f"最高相似度：{np.max(similarity_list):.4f}")
print(f"最低相似度：{np.min(similarity_list):.4f}")
print(f"相似度中位数：{np.median(similarity_list):.4f}")
print(f"相似度>0.5的句子数：{sum(1 for s in similarity_list if s > 0.5)}")