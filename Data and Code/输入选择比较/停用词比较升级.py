import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings('ignore')

# ================= 1. 配置参数 (保持不变) =================
LOCAL_MODEL_PATH = r'D:\models\bert\bert-base-chinese'
DATA_DIR = r'D:\models\农业农村部一号文件\解读关键词'
FILES = {
    'no_stopword': '关键词未政策停用.txt',
    'with_stopword': '关键词政策停用.txt',
    'expert': '专家解读关键词.txt'
}

# 全局绘图设置 (适配英文论文标准)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_context("paper", font_scale=1.1)
colors = {'red': '#d62728', 'blue': '#1f77b4', 'green': '#2ca02c', 'gray': '#7f7f7f'}

# ================= 2. 数据解析与模型加载 (保持不变) =================
print("🧠 正在加载本地 BERT 模型...")
model = SentenceTransformer(LOCAL_MODEL_PATH)

def parse_keyword_file(file_path, is_expert=False):
    """稳健的文本读取函数，支持跨行正则解析"""
    data = {}
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return data

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().replace('\ufeff', '')

    if is_expert:
        pattern = re.compile(r'(\d{4})\s*[:：]\s*(.*?)(?=\d{4}\s*[:：]|\Z)', re.DOTALL)
    else:
        pattern = re.compile(r'(\d{4})\s*年关键词.*?[:：]\s*(.*?)(?=\d{4}\s*年关键词|\Z)', re.DOTALL)

    matches = pattern.finditer(content)
    for match in matches:
        year = int(match.group(1))
        kws_raw = match.group(2).replace('\n', '').replace('"', '').replace("'", "")
        keywords = []
        for kw in kws_raw.split(','):
            kw = kw.strip()
            if not kw: continue
            if '/' in kw:
                keywords.extend([k.strip() for k in kw.split('/') if k.strip()])
            else:
                keywords.append(kw)
        data[year] = list(set(keywords))
    return data


def core_semantic_similarity(keywords, expert_keywords, bert_model):
    """计算核心语义相似度"""
    if not keywords or not expert_keywords:
        return 0.0
    kw_emb = bert_model.encode(keywords)
    expert_emb = bert_model.encode(expert_keywords)
    sim_matrix = cosine_similarity(kw_emb, expert_emb)
    max_sims = np.max(sim_matrix, axis=0)
    return round(float(np.mean(max_sims)), 4)


# ================= 3. 数据流转与相似度计算 (保持不变) =================
print("📂 正在读取与解析数据...")
no_stopword_data = parse_keyword_file(os.path.join(DATA_DIR, FILES['no_stopword']))
with_stopword_data = parse_keyword_file(os.path.join(DATA_DIR, FILES['with_stopword']))
expert_data = parse_keyword_file(os.path.join(DATA_DIR, FILES['expert']), is_expert=True)

common_years = sorted(list(set(no_stopword_data.keys()) & set(with_stopword_data.keys()) & set(expert_data.keys())))
print(f"📅 锁定有效计算年份 ({len(common_years)}年)：{common_years[0]}-{common_years[-1]}")

print("⚙️ 正在执行高维向量映射与相似度计算...")
results = []
for year in common_years:
    sim_no_stop = core_semantic_similarity(no_stopword_data[year], expert_data[year], model)
    sim_with_stop = core_semantic_similarity(with_stopword_data[year], expert_data[year], model)
    results.append({
        'Year': year,
        'No_Stop': sim_no_stop,
        'With_Stop': sim_with_stop,
        'Diff': sim_with_stop - sim_no_stop
    })

df = pd.DataFrame(results)

# ================= 4. 统计显著性检验 (总体分析) =================
t_stat, p_value_t = stats.ttest_rel(df['With_Stop'], df['No_Stop'])

if p_value_t < 0.001:
    stars = "***"
elif p_value_t < 0.01:
    stars = "**"
elif p_value_t < 0.05:
    stars = "*"
else:
    stars = "ns"

print("\n" + "=" * 40)
print(f"📊 统计检验结果速览: t={t_stat:.2f}, p={p_value_t:.4e} ({stars})")
print("=" * 40 + "\n")

# ================= 5. 核心：自动化完美组合图渲染 (全英文化) =================
print("🖌️ 正在渲染【全英文版】组合图表...")

fig = plt.figure(figsize=(13, 10))
gs = GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

# === 5.1 上半部分：年度趋势对比图 ===
ax_trend = fig.add_subplot(gs[0, :])

ax_trend.plot(df['Year'], df['With_Stop'], 'o-', color=colors['red'], label='Proposed (Ours)', linewidth=2.5, markersize=7)
ax_trend.plot(df['Year'], df['No_Stop'], 's--', color=colors['blue'], label='Baseline', linewidth=1.5, alpha=0.7)
ax_trend.fill_between(df['Year'], df['No_Stop'], df['With_Stop'], where=(df['With_Stop'] > df['No_Stop']),
                      interpolate=True, color=colors['red'], alpha=0.15)

ax_trend.set_ylabel('BERT Core Semantic Similarity Score', fontsize=12)
ax_trend.set_title('A. Annual Semantic Similarity Trend: Before and After Text Denoising', fontweight='bold', fontsize=14, loc='left')
ax_trend.legend(loc='upper right', fontsize=11, edgecolor='gray', framealpha=0.9)
ax_trend.grid(True, linestyle='--', alpha=0.5)
ax_trend.set_xticks(df['Year'])
ax_trend.set_xticklabels(df['Year'], rotation=45, fontsize=10)

# === 5.2 左下角：绝对性能增益柱状图 ===
ax_gain = fig.add_subplot(gs[1, 0])

bar_colors = [colors['red'] if x >= 0 else colors['green'] for x in df['Diff']]
ax_gain.bar(df['Year'], df['Diff'], color=bar_colors, alpha=0.8)
ax_gain.axhline(0, color='black', linewidth=1)

max_diff = df['Diff'].max()
max_year = df.loc[df['Diff'].idxmax(), 'Year']
ax_gain.set_ylim(df['Diff'].min() - 0.01, max_diff + 0.02)
ax_gain.text(max_year, max_diff + 0.003, f'+{max_diff:.3f}', ha='center', fontsize=9, color=colors['red'])

ax_gain.set_ylabel('Absolute Similarity Gain (Ours - Baseline)', fontsize=12)
ax_gain.set_xlabel('Policy Year', fontsize=12)
ax_gain.set_title('B. Distribution of Annual Performance Gains', fontweight='bold', fontsize=14, loc='left')
ax_gain.set_xticks(df['Year'])
ax_gain.set_xticklabels(df['Year'], rotation=90, fontsize=9)
ax_gain.grid(axis='y', linestyle='--', alpha=0.5)

# === 5.3 右下角：总体统计分布箱线图 ===
ax_box = fig.add_subplot(gs[1, 1])

# 灰色配对轨迹连线
for i in range(len(df)):
    ax_box.plot([0, 1], [df['No_Stop'][i], df['With_Stop'][i]], color='gray', alpha=0.15, linestyle='-', linewidth=0.8, zorder=1)

# 数据格式转换
df_long = pd.melt(df, id_vars=['Year'], value_vars=['No_Stop', 'With_Stop'], var_name='Method', value_name='Score')
df_long['Method'] = df_long['Method'].map({'No_Stop': 'Baseline', 'With_Stop': 'Proposed (Ours)'})

# 显式声明箱线图渲染顺序，确保与上述[0, 1]的灰色连线匹配
plot_order = ['Baseline', 'Proposed (Ours)']

sns.boxplot(x='Method', y='Score', data=df_long, order=plot_order, width=0.4, palette=[colors['blue'], colors['red']],
            boxprops={'alpha': 0.7, 'linewidth': 1.5}, ax=ax_box)
sns.swarmplot(x='Method', y='Score', data=df_long, order=plot_order, size=6, color='0.25', alpha=0.6, ax=ax_box)

y_max = df_long['Score'].max()
y_line = y_max + 0.015
ax_box.plot([0, 0, 1, 1], [y_line - 0.005, y_line, y_line, y_line - 0.005], lw=1.2, c='black')
ax_box.text(0.5, y_line + 0.002, f'{stars}\n(p={p_value_t:.4f})\nt={t_stat:.2f}', ha='center', va='bottom',
            color='black', fontweight='bold', fontsize=10)

ax_box.set_title('C. Overall Score Distribution and Paired t-Test', fontweight='bold', fontsize=14, loc='left')
ax_box.set_ylabel('Score Distribution', fontsize=12)
ax_box.set_xlabel('')
ax_box.set_ylim(df_long['Score'].min() - 0.02, y_max + 0.05)
ax_box.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()

save_path = os.path.join(DATA_DIR, 'Figure_Stopword_Optimization_Merged_EN.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n🎉 完美英文版组合图谱已生成: {save_path}")