import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# ========== 配置参数 ==========
LOCAL_MODEL_PATH = r'D:\models\bert\bert-base-chinese'
DATA_DIR = r'D:\models\农业农村部一号文件\解读关键词'
FILES = {
    'no_stopword': '关键词未政策停用.txt',
    'with_stopword': '关键词政策停用.txt',
    'expert': '专家解读关键词.txt'
}

# ========== 加载本地BERT模型 ==========
print("正在加载模型...")
model = SentenceTransformer(LOCAL_MODEL_PATH)


# ========== 数据解析函数 ==========
def parse_keyword_file(file_path, is_expert=False):
    data = {}
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return data

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().replace('\ufeff', '')  # 去BOM
        if not line:
            continue
        if is_expert:
            match = re.search(r'(\d{4})\s*[:：]\s*(.*)', line)
        else:
            match = re.search(r'(\d{4})\s*年关键词.*?[:：]\s*(.*)', line)
        if match:
            year = int(match.group(1))
            kws = match.group(2).replace('"', '').replace("'", "")
            keywords = []
            for kw in kws.split(','):
                if '/' in kw:
                    keywords.extend([k.strip() for k in kw.split('/') if k.strip()])
                else:
                    keywords.append(kw.strip())
            data[year] = list(set([k for k in keywords if k]))
        else:
            # 只有当行内容看起来像数据但没匹配上时才警告，避免空行干扰
            if len(line) > 5 and ('20' in line):
                print(f"⚠️ 未匹配行：{line[:50]}")
    return data


# ========== 读取数据 ==========
print("正在读取数据...")
no_stopword_data = parse_keyword_file(os.path.join(DATA_DIR, FILES['no_stopword']))
with_stopword_data = parse_keyword_file(os.path.join(DATA_DIR, FILES['with_stopword']))
expert_data = parse_keyword_file(os.path.join(DATA_DIR, FILES['expert']), is_expert=True)

common_years = sorted(list(set(no_stopword_data.keys()) & set(with_stopword_data.keys()) & set(expert_data.keys())))
print(f"📅 共同年份：{common_years}")


# ========== 核心语义相似度 ==========
def core_semantic_similarity(keywords, expert_keywords, model):
    if not keywords or not expert_keywords:
        return 0.0
    kw_emb = model.encode(keywords)
    expert_emb = model.encode(expert_keywords)
    sim_matrix = cosine_similarity(kw_emb, expert_emb)
    max_sims = np.max(sim_matrix, axis=0)  # 每个专家词找最匹配提取词
    return round(float(np.mean(max_sims)), 4)


# ========== 计算年度指标 ==========
print("正在计算相似度...")
results = []
for year in common_years:
    no_stop_kw = no_stopword_data[year]
    with_stop_kw = with_stopword_data[year]
    expert_kw = expert_data[year]

    sim_no_stop = core_semantic_similarity(no_stop_kw, expert_kw, model)
    sim_with_stop = core_semantic_similarity(with_stop_kw, expert_kw, model)

    results.append({
        '年份': year,
        'BERT相似度_未加停用': sim_no_stop,
        'BERT相似度_加入停用': sim_with_stop,
        '差异值': sim_with_stop - sim_no_stop
    })

df = pd.DataFrame(results)
output_excel_path = os.path.join(DATA_DIR, '关键词BERT相似度分析.xlsx')
df.to_excel(output_excel_path, index=False)
print(f"💾 结果已保存至: {output_excel_path}")

# ========== 输出分析 ==========
# 直接使用 BERT 相似度列来计算均值
avg_no_stop = df['BERT相似度_未加停用'].mean()
avg_with_stop = df['BERT相似度_加入停用'].mean()
better_count = (df['差异值'] > 0).sum()

print("\n🎯 每年BERT相似度效果：")
print(df.round(4))

print("\n🎯 分析结论：")
if avg_with_stop > avg_no_stop:
    print(f"✅ 加入停用词结果整体更好（平均相似度高 {avg_with_stop - avg_no_stop:.4f}）")
elif avg_with_stop < avg_no_stop:
    print(f"❌ 未加停用词结果整体更好（平均相似度高 {avg_no_stop - avg_with_stop:.4f}）")
else:
    print("⚖️ 两者平均相似度相当")
print(f"在{len(common_years)}个年份中，有{better_count}个年份加入停用词更优")