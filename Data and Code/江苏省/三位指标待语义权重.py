import os
import re
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colorbar import ColorbarBase
import matplotlib

# 使用非交互式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= 配置字体 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 加载模型 ==========
print("🔍 加载BERT模型中...")
model = SentenceTransformer('shibing624/text2vec-base-chinese')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"💻 使用设备: {device}")

# ========== 配置路径 ==========
# 两省的文本文件夹
jiangsu_dir = r'C:\Users\user\Desktop\word\江苏省委一号文件'
sichuan_dir = r'C:\Users\user\Desktop\word\四川省委一号文件'

# 统一使用带有权重的 JSON 聚类结果作为中央基准
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

all_years = list(range(2004, 2027))
n_rows = len(all_years)
n_cols = len(stages)
original_stages = list(stages.keys())

# ========== 读取加权语义簇 (JSON) ==========
stage_clusters = {}
print("\n📁 正在读取中央政策各阶段加权语义簇...")
for stage in original_stages:
    fname = f'{stage}_clusters.json'
    fpath = os.path.join(keywords_dir, fname)
    if not os.path.exists(fpath):
        print(f"⚠️ 找不到聚类文件: {fpath}")
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        stage_clusters[stage] = json.load(f)

# 提前计算各阶段的质心向量（避免在双省份循环中重复计算）
print("⚙️ 预计算中央阶段加权语义质心...")
stage_centroids = {}
for stage in original_stages:
    clusters_data = stage_clusters.get(stage, {})
    if not clusters_data: continue

    centroid_vecs = []
    for core_word, data in clusters_data.items():
        words_to_encode = [core_word] + list(data['related'].keys())
        weights = [data['core_weight']] + list(data['related'].values())

        word_embeddings = model.encode(words_to_encode, convert_to_tensor=True)
        weight_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)
        weighted_sum = torch.sum(word_embeddings * weight_tensor, dim=0)
        centroid = weighted_sum / torch.sum(weight_tensor)
        centroid_vecs.append(centroid.cpu().numpy())
    stage_centroids[stage] = np.array(centroid_vecs)


# ========== 文件读取函数 ==========
def load_province_texts(directory, pattern_str):
    province_dict = {}
    if os.path.exists(directory):
        for filename in sorted(os.listdir(directory)):
            match = re.match(pattern_str, filename)
            if match:
                year = int(match.group(1))
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                    province_dict[year] = f.read().strip()
    else:
        print(f"❌ 文件夹不存在: {directory}")
    return province_dict


def split_sentences(text):
    text = re.sub(r'[。！？；?!;]', '。', text)
    return [s.strip() for s in text.split('。') if s.strip()]


print("\n📁 读取地方一号文件...")
jiangsu_dict = load_province_texts(jiangsu_dir, r'(\d{4})江苏省委一号文件\.txt')
sichuan_dict = load_province_texts(sichuan_dir, r'(\d{4})四川省一号文件\.txt')  # 按照你给的正则匹配

# ========== 核心计算逻辑 ==========
SIMILARITY_THRESHOLD = 0.55


def calculate_matrices(province_dict, name):
    f_mat = np.full((n_rows, n_cols), np.nan)
    r_mat = np.full((n_rows, n_cols), np.nan)
    p_mat = np.full((n_rows, n_cols), np.nan)

    print(f"\n🧠 开始计算 [{name}] 前瞻度 / 响应度 / 持续性...")
    for i, year in enumerate(all_years):
        content = province_dict.get(year, "")
        sentences = split_sentences(content)

        if len(sentences) == 0:
            print(f"  - {year} 年无数据，跳过")
            continue

        sentence_vecs = model.encode(sentences, convert_to_tensor=True)

        for j, stage in enumerate(original_stages):
            centroids = stage_centroids.get(stage)
            if centroids is None or len(centroids) == 0: continue

            match_count = 0
            for sent_vec in sentence_vecs:
                sims = cosine_similarity([sent_vec.cpu().numpy()], centroids)[0]
                if max(sims) >= SIMILARITY_THRESHOLD:
                    match_count += 1

            match_ratio = match_count / len(sentences)
            stage_years = stages[stage]

            if year < min(stage_years):
                f_mat[i, j] = match_ratio
            elif min(stage_years) <= year <= max(stage_years):
                r_mat[i, j] = match_ratio
            elif year > max(stage_years):
                p_mat[i, j] = match_ratio

    return f_mat, r_mat, p_mat


sc_f, sc_r, sc_p = calculate_matrices(sichuan_dict, "四川省")
js_f, js_r, js_p = calculate_matrices(jiangsu_dict, "江苏省")

# ========== 并排可视化绘图 ==========
print("\n🎨 正在绘制双省份对比热力图...")
vmin, vmax = 0.3, 1.0
cmap_dict = {'持续性': plt.cm.Blues, '响应度': plt.cm.Reds, '前瞻度': plt.cm.Greens}

fig = plt.figure(figsize=(22, 12))  # 增加画布宽度以容纳双图

# 定义左右两个绘图区和右侧色带区
ax_sc = fig.add_axes([0.05, 0.1, 0.38, 0.75])  # 四川图
ax_js = fig.add_axes([0.48, 0.1, 0.38, 0.75])  # 江苏图


def draw_province_heatmap(ax, f_mat, r_mat, p_mat, title):
    for i in range(n_rows):
        for j in range(n_cols):
            stage_name = original_stages[j]
            stage_years = stages[stage_name]
            year = all_years[i]

            if year < min(stage_years):
                metric, value = '前瞻度', f_mat[i, j]
            elif min(stage_years) <= year <= max(stage_years):
                metric, value = '响应度', r_mat[i, j]
            else:
                metric, value = '持续性', p_mat[i, j]

            if np.isnan(value):
                rect = plt.Rectangle((j, i), 1, 1, facecolor='#F5F5F5', edgecolor='white')
                ax.add_patch(rect)
                ax.text(j + 0.5, i + 0.5, "NaN", ha='center', va='center', fontsize=10, color='#A0A0A0',
                        fontstyle='italic')
            else:
                cmap = cmap_dict[metric]
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                color = cmap(norm(value))
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white')
                ax.add_patch(rect)
                ax.text(j + 0.5, i + 0.5, f"{value:.3f}", ha='center', va='center', fontsize=11, color='black')

    display_stages_cn = [s.replace('_', '\n') for s in original_stages]
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(display_stages_cn, fontsize=13)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(all_years, fontsize=13)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)


# 绘制两侧热力图
draw_province_heatmap(ax_sc, sc_f, sc_r, sc_p, "四川省政策响应度时空演化图谱")
draw_province_heatmap(ax_js, js_f, js_r, js_p, "江苏省政策响应度时空演化图谱")

# 移除江苏图的Y轴年份标签（因为左右图对齐，保留左侧四川的Y轴即可保持画面整洁）
ax_js.set_yticklabels([])

# 绘制全局图例 (放置在最右侧)
bar_width, bar_height, spacing, start_x = 0.015, 0.75, 0.04, 0.89
colorbars = {'前瞻度': plt.cm.Greens, '响应度': plt.cm.Reds, '持续性': plt.cm.Blues}
labels_cn = ['前瞻度', '响应度', '持续性']

for i, (name, cmap) in enumerate(colorbars.items()):
    x_pos = start_x + i * (bar_width + spacing)
    cbar_ax = fig.add_axes([x_pos, 0.1, bar_width, bar_height])
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

text_positions = [start_x + (bar_width + spacing) * i + bar_width / 2 for i in range(3)]
for x, label in zip(text_positions, labels_cn):
    fig.text(x, 0.06, label, ha='center', va='top', fontsize=14, fontweight='bold')

plt.suptitle("川苏两省政策传导语义一致性时空对比分析", fontsize=26, fontweight='bold', y=0.98)

# 保存
save_path = os.path.join(output_dir, 'Policy_Responsiveness_Comparison_川苏.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ 合并对比图已保存至：{save_path}")