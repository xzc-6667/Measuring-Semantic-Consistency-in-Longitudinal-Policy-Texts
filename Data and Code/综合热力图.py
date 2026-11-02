import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colorbar import ColorbarBase
import matplotlib
import matplotlib.pyplot as plt

# 设置绘图后端和字体
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 加载模型 ==========
# 注意：论文中提到使用了 BERT-base-chinese 或 BGE 等，此处保持你代码中的模型，
# 但如果要严格对应论文提到的 BGE，建议更换模型名称，例如 'BAAI/bge-base-zh-v1.5'
print("🔍 加载预训练模型中...")
model = SentenceTransformer('shibing624/text2vec-base-chinese')

# ========== 2. 配置路径 ==========
province_dir = r'C:\Users\user\Desktop\word\四川省委一号文件'
keywords_dir = r'C:\Users\user\Desktop\word\农业农村部一号文件\关键词'
output_dir = r'C:\Users\user\Desktop\word\农业农村部一号文件\图片\省级响应'
os.makedirs(output_dir, exist_ok=True)


# ========== 3. 核心函数：构建5句上下文窗口 [对应论文 3.2.2] ==========
def split_sentences(text):
    # 简单的分句逻辑
    text = re.sub(r'[。！？；?!;]', '。', text)
    sentences = [s.strip() for s in text.split('。') if s.strip()]
    return sentences


def build_context_windows(sentences, window_size=2):
    """
    构建上下文窗口：目标句 + 前2句 + 后2句
    如果开头或结尾不足，用空字符串补齐（逻辑上直接拼接即可，空串不影响）
    """
    windows = []
    n = len(sentences)
    for i in range(n):
        # 获取切片范围
        start_idx = max(0, i - window_size)
        end_idx = min(n, i + window_size + 1)

        # 提取上下文列表
        context_parts = sentences[start_idx:end_idx]

        # 拼接成一个长文本作为模型输入
        window_text = " ".join(context_parts)
        windows.append(window_text)
    return windows


# ========== 4. 数据准备 ==========
stages = {
    '阶段一_2004-2007': list(range(2004, 2008)),
    '阶段二_2008-2012': list(range(2008, 2013)),
    '阶段三_2013-2017': list(range(2013, 2018)),
    '阶段四_2018-2022': list(range(2018, 2023)),
    '阶段五_2023-2025': list(range(2023, 2026)),
}

# 读取关键词并预先计算“中央政策综合向量” [对应论文 4.1.4]
stage_vectors = {}
print("📁 正在读取并编码各阶段关键词...")

for stage_name, years_list in stages.items():
    fname = f'{stage_name}_keywords.txt'
    fpath = os.path.join(keywords_dir, fname)

    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            keywords = [line.strip() for line in f if line.strip()]

        if keywords:
            # 1. 编码所有关键词 -> (N, 768)
            kw_embeddings = model.encode(keywords)
            # 2. 取算术平均值得到“综合向量” -> (1, 768)
            # 论文引用: "...将基准关键词集中各关键词转换为词向量并取算术平均值"
            mean_vec = np.mean(kw_embeddings, axis=0).reshape(1, -1)
            stage_vectors[stage_name] = mean_vec
            print(f"✅ {stage_name}: 已构建综合向量，关键词数: {len(keywords)}")
        else:
            print(f"⚠️ {stage_name}: 关键词文件为空")
    else:
        print(f"⚠️ 未找到文件: {fpath}")

# ========== 5. 计算前瞻度/响应度/持续性 ==========
# 初始化结果容器
results = []
# 遍历年份
file_list = sorted([f for f in os.listdir(province_dir) if f.endswith('.txt')])

print("\n🧠 开始计算语义匹配度...")
for filename in file_list:
    match = re.match(r'(\d{4})', filename)
    if not match: continue

    year = int(match.group(1))
    file_path = os.path.join(province_dir, filename)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # 1. 分句
    sentences = split_sentences(content)
    if not sentences: continue

    # 2. 构建5句上下文窗口 [修正点]
    windows = build_context_windows(sentences, window_size=2)

    # 3. 编码地方文件窗口 -> (M, 768)
    # 论文引用: "...将地方文件的5句上下文窗口输入模型生成句向量"
    window_vecs = model.encode(windows)

    # 4. 与各阶段中央向量进行计算
    for stage_name, stage_vec in stage_vectors.items():
        stage_years = stages[stage_name]

        # 计算余弦相似度: (M, 768) vs (1, 768) -> (M, 1)
        sim_scores = cosine_similarity(window_vecs, stage_vec)

        # 取均值作为该文件对该阶段的匹配度
        # 论文引用: "语义契合均值"
        # 这里的 mean_score 就是最终的[0,1]区间的数值
        mean_score = np.mean(sim_scores)

        # 判定指标类型
        metric_type = ''
        if year < min(stage_years):
            metric_type = '前瞻度'
        elif min(stage_years) <= year <= max(stage_years):
            metric_type = '响应度'
        else:
            metric_type = '持续性'

        results.append({
            'Year': year,
            'Stage': stage_name,
            'Metric': metric_type,
            'Value': mean_score
        })

    print(f"✅ 完成年份: {year} 计算")

# ========== 6. 数据转换与可视化 ==========
df = pd.DataFrame(results)

# 转换数据格式以适配绘图矩阵
# 行：年份，列：阶段
years = sorted(df['Year'].unique())
stage_names = list(stages.keys())
n_rows = len(years)
n_cols = len(stage_names)

matrix_value = np.zeros((n_rows, n_cols))
matrix_metric = [['' for _ in range(n_cols)] for _ in range(n_rows)]

for r, y in enumerate(years):
    for c, s in enumerate(stage_names):
        # 筛选对应值
        item = df[(df['Year'] == y) & (df['Stage'] == s)]
        if not item.empty:
            matrix_value[r, c] = item['Value'].values[0]
            matrix_metric[r][c] = item['Metric'].values[0]

# --- 绘图部分 (保持原风格，仅微调归一化) ---
# 注意：由于使用了平均相似度，数值可能会集中在 0.4-0.8 之间，
# 为了视觉效果，建议绘图时进行 Normalize 或者动态调整 vmin/vmax
vmin = np.min(matrix_value)
vmax = np.max(matrix_value)

fig = plt.figure(figsize=(16, 10))
main_ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])

# 颜色映射
cmap_dict = {
    '持续性': plt.cm.Blues,
    '响应度': plt.cm.Reds,
    '前瞻度': plt.cm.Greens
}
metric_label_en = {
    '持续性': 'Persistence',
    '响应度': 'Responsiveness',
    '前瞻度': 'Foresight'
}

for i in range(n_rows):
    for j in range(n_cols):
        metric = matrix_metric[i][j]
        value = matrix_value[i][j]
        if metric in cmap_dict:
            cmap = cmap_dict[metric]
            # 归一化：为了让颜色差异更明显，可以基于每一列（每一阶段）或全局做归一化
            # 这里保持全局归一化
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            color = cmap(norm(value))
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white')
            main_ax.add_patch(rect)
            main_ax.text(j + 0.5, i + 0.5, f"{value:.3f}",
                         ha='center', va='center', fontsize=8, color='black')

# 坐标轴设置
display_stages_en = [f"Stage {k + 1}" for k in range(len(stage_names))]
main_ax.set_xticks(np.arange(n_cols) + 0.5)
main_ax.set_xticklabels(display_stages_en, fontsize=12)
main_ax.set_yticks(np.arange(n_rows) + 0.5)
main_ax.set_yticklabels(years, fontsize=12)
main_ax.set_xlim(0, n_cols)
main_ax.set_ylim(0, n_rows)
main_ax.invert_yaxis()
main_ax.set_title("Policy Responsiveness Measurement (Semantic Context Window)", fontsize=16)

# 图例绘制 (保持不变)
bar_width = 0.02
bar_height = 0.8
spacing = 0.05
start_x = 0.72
colorbars = {'前瞻度': plt.cm.Greens, '响应度': plt.cm.Reds, '持续性': plt.cm.Blues}

for i, (name, cmap) in enumerate(colorbars.items()):
    x_pos = start_x + i * (bar_width + spacing)
    cbar_ax = fig.add_axes([x_pos, 0.1, bar_width, bar_height])
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

text_positions = [start_x + (bar_width + spacing) * i + bar_width / 2 for i in range(3)]
labels_en = [metric_label_en[name] for name in ['前瞻度', '响应度', '持续性']]
for x, label in zip(text_positions, labels_en):
    fig.text(x, 0.05, label, ha='center', va='top', fontsize=10)

plt.savefig(os.path.join(output_dir, 'Policy_Responsiveness_Heatmap_Corrected_en.png'), dpi=300, bbox_inches='tight')
print("✅ 处理完成，图片已保存。")