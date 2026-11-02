import os
import re
import numpy as np
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colorbar import ColorbarBase
import matplotlib

# 使用非交互式后端，适合保存图像
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= 配置字体 =================
# 优先使用 SimHei (黑体)，如果没有则使用备选字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# ========== 加载模型 ==========
print("🔍 加载BERT模型中...")
model = SentenceTransformer('shibing624/text2vec-base-chinese')

# ========== 配置路径 ==========
province_dir = r'C:\Users\user\Desktop\word\江苏省委一号文件'
keywords_dir = r'C:\Users\user\Desktop\word\农业农村部一号文件\扩展语义簇'
output_dir = r'C:\Users\user\Desktop\word\农业农村部一号文件\图片\省级响应'
os.makedirs(output_dir, exist_ok=True)

# ========== 阶段与年份 ==========
stages = {
    '阶段一_2004-2007': list(range(2004, 2008)),
    '阶段二_2008-2012': list(range(2008, 2013)),
    '阶段三_2013-2017': list(range(2013, 2018)),
    '阶段四_2018-2022': list(range(2018, 2023)),
    '阶段五_2023-2026': list(range(2023, 2027)),
}

# ========== 读取关键词 ==========
stage_keywords = {}
print("📁 正在读取各阶段关键词文件...")
for stage in stages:
    fname = f'{stage}_keywords.txt'
    fpath = os.path.join(keywords_dir, fname)
    if not os.path.exists(fpath):
        print(f"⚠️ 找不到关键词文件: {fpath}")
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        keywords = [line.strip() for line in f if line.strip()]
        stage_keywords[stage] = keywords
        print(f"✅ 已加载：{stage}，关键词数量：{len(keywords)}")

# ========== 读取省级文件 ==========
province_files = {}
print("\n📁 读取江苏省委一号文件内容...")
if os.path.exists(province_dir):
    for filename in sorted(os.listdir(province_dir)):
        match = re.match(r'(\d{4})江苏省委一号文件\.txt', filename)
        if match:
            year = int(match.group(1))
            with open(os.path.join(province_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read().strip()
                province_files[year] = content  # 改用字典存储，方便查找缺失年份
                print(f"✅ 读取年份：{year}，文件：{filename}")
else:
    print(f"❌ 文件夹不存在: {province_dir}")

# ========== 强制构建连续时间轴 ==========
# 从2004到2026，确保即使文件缺失，年份行也会保留
years = list(range(2004, 2027))
n_stages = len(stages)

# 使用 np.nan 初始化矩阵，代表缺失值
foresight_matrix = np.full((len(years), n_stages), np.nan)
response_matrix = np.full((len(years), n_stages), np.nan)
persistence_matrix = np.full((len(years), n_stages), np.nan)


# ========== 中文分句 ==========
def split_sentences(text):
    text = re.sub(r'[。！？；?!;]', '。', text)
    sentences = [s.strip() for s in text.split('。') if s.strip()]
    return sentences


# ========== 相似度阈值 ==========
SIMILARITY_THRESHOLD = 0.5

# ========== 计算三类指标 ==========
print("\n🧠 正在计算前瞻度 / 响应度 / 持续性...")
for i, year in enumerate(years):
    if year not in province_files:
        print(f"⚠️ 年份 {year} 缺失文件，已记作 NaN")
        continue  # 直接跳过，矩阵中保留为 np.nan

    print(f"\n🗂️ 处理年份：{year}")
    content = province_files[year]
    sentences = split_sentences(content)
    if len(sentences) == 0:
        print("⚠️ 该文件无句子，跳过")
        continue
    sentence_vecs = model.encode(sentences, convert_to_tensor=True)

    for j, (stage, keywords) in enumerate(stage_keywords.items()):
        keyword_vecs = model.encode(keywords, convert_to_tensor=True)

        match_count = 0
        for sent_vec in sentence_vecs:
            sims = cosine_similarity([sent_vec.cpu().numpy()], keyword_vecs.cpu().numpy())[0]
            max_sim = max(sims)
            if max_sim >= SIMILARITY_THRESHOLD:
                match_count += 1

        match_ratio = match_count / len(sentences)

        stage_years = stages[stage]
        if year < min(stage_years):
            foresight_matrix[i, j] = match_ratio
            print(f"🔮 前瞻度 → 阶段：{stage}，匹配句比例：{match_ratio:.4f}")
        elif min(stage_years) <= year <= max(stage_years):
            response_matrix[i, j] = match_ratio
            print(f"📣 响应度 → 阶段：{stage}，匹配句比例：{match_ratio:.4f}")
        elif year > max(stage_years):
            persistence_matrix[i, j] = match_ratio
            print(f"♻ 持续性 → 阶段：{stage}，匹配句比例：{match_ratio:.4f}")

# ========== 数据整理 ==========
original_stages = list(stages.keys())

# 创建数值矩阵和指标类型矩阵
n_rows = len(years)
n_cols = len(original_stages)
matrix = np.full((n_rows, n_cols), np.nan)
metrics = [[None for _ in range(n_cols)] for _ in range(n_rows)]

for i, year in enumerate(years):
    for j, stage in enumerate(original_stages):
        stage_years = stages[stage]
        if year < min(stage_years):
            metrics[i][j] = '前瞻度'
            matrix[i][j] = foresight_matrix[i, j]
        elif min(stage_years) <= year <= max(stage_years):
            metrics[i][j] = '响应度'
            matrix[i][j] = response_matrix[i, j]
        elif year > max(stage_years):
            metrics[i][j] = '持续性'
            matrix[i][j] = persistence_matrix[i, j]

# ========== 绘制可视化图表 (中文版) ==========
cmap_dict = {
    '持续性': plt.cm.Blues,
    '响应度': plt.cm.Reds,
    '前瞻度': plt.cm.Greens
}

vmin, vmax = 0.0, 1.0

fig = plt.figure(figsize=(16, 10))
# 稍微把主图往上提一点，留出底部空间写备注
main_ax = fig.add_axes([0.1, 0.15, 0.6, 0.75])

# 绘制热力图矩阵
for i in range(n_rows):
    for j in range(n_cols):
        metric = metrics[i][j]
        value = matrix[i][j]

        # 判断是否为 NaN (缺失值)
        if np.isnan(value):
            # 缺失值画成浅灰色，并加上斜线以示区分
            rect = plt.Rectangle((j, i), 1, 1, facecolor='#EAEAEA', edgecolor='white', hatch='//')
            main_ax.add_patch(rect)
        else:
            cmap = cmap_dict[metric]
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            color = cmap(norm(value))
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white')
            main_ax.add_patch(rect)
            # 正常显示数值
            main_ax.text(
                j + 0.5, i + 0.5, f"{value:.3f}",
                ha='center', va='center', fontsize=9, color='black'
            )

# ======================
# 坐标轴与标签配置
# ======================
display_stages_cn = [s.replace('_', '\n') for s in original_stages]

main_ax.set_xticks(np.arange(n_cols) + 0.5)
main_ax.set_xticklabels(display_stages_cn, fontsize=12)

main_ax.set_yticks(np.arange(n_rows) + 0.5)
main_ax.set_yticklabels(years, fontsize=12)

main_ax.set_xlim(0, n_cols)
main_ax.set_ylim(0, n_rows)
main_ax.invert_yaxis()

# ======================
# 标题与备注
# ======================
main_ax.set_title("政策响应度时空演化图谱", fontsize=18, fontweight='bold', pad=20)

# 在图表左下方添加缺失值说明
fig.text(0.1, 0.08, "* 注：江苏省 2005 年因缺乏同等效力公开文本，作结构性缺失（NaN）处理。",
         fontsize=12, color='#555555', va='center')

# ======================
# 渐变长条图例
# ======================
bar_width = 0.02
bar_height = 0.75
spacing = 0.05
start_x = 0.72

colorbars = {
    '前瞻度': plt.cm.Greens,
    '响应度': plt.cm.Reds,
    '持续性': plt.cm.Blues
}
labels_cn = ['前瞻度', '响应度', '持续性']

for i, (name, cmap) in enumerate(colorbars.items()):
    x_pos = start_x + i * (bar_width + spacing)
    cbar_ax = fig.add_axes([x_pos, 0.15, bar_width, bar_height])
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

text_positions = [start_x + (bar_width + spacing) * i + bar_width / 2 for i in range(3)]
for x, label in zip(text_positions, labels_cn):
    fig.text(x, 0.11, label, ha='center', va='top', fontsize=12, fontweight='bold')

# 保存图片
save_path = os.path.join(output_dir, 'Policy_Responsiveness_Heatmap_jiangsu_Chinese_扩展.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ 图片已保存至：{save_path}")