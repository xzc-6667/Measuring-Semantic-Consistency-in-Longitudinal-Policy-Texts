import os
import re
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from scipy.spatial.distance import jensenshannon
import jieba
import matplotlib

# 强制使用标准的无头后端，避免绘图报错
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase

# ================= 1. 基础配置与变量定义 =================
base_path = r'C:\Users\user\Desktop\word'
central_dir = os.path.join(base_path, '农业农村部一号文件')
jiangsu_dir = os.path.join(base_path, '江苏省委一号文件')
sichuan_dir = os.path.join(base_path, '四川省委一号文件')
output_dir = os.path.join(base_path, '图片', '省级响应')
os.makedirs(output_dir, exist_ok=True)

stages = {
    'Stage 1_2004-2007': list(range(2004, 2008)),
    'Stage 2_2008-2012': list(range(2008, 2013)),
    'Stage 3_2013-2017': list(range(2013, 2018)),
    'Stage 4_2018-2022': list(range(2018, 2023)),
    'Stage 5_2023-2026': list(range(2023, 2027)),
}
original_stages = list(stages.keys())
all_years = list(range(2004, 2027))


# ================= 2. 函数定义区 (必须在外部) =================
def load_texts(directory, pattern):
    res = {}
    if not os.path.exists(directory): return res
    for fname in os.listdir(directory):
        match = re.match(pattern, fname)
        if match:
            with open(os.path.join(directory, fname), 'r', encoding='utf-8') as f:
                res[int(match.group(1))] = f.read().strip()
    return res


def load_stopwords():
    stopwords = set()
    hit_stopwords_path = r"C:\Users\user\Desktop\word\chinese_stopwords.txt"
    try:
        with open(hit_stopwords_path, "r", encoding="utf-8") as f:
            for line in f:
                stopwords.add(line.strip())
    except FileNotFoundError:
        print(f"⚠️ 未找到哈工大停用词文件，请检查路径: {hit_stopwords_path}")

    # 严格对齐论文附录 A 的 106 个专属停用词
    domain_stopwords = {
        '继续', '推进', '加强', '加大', '开展', '完善', '建立', '健全', '实施', '发展',
        '有效', '提高', '加快', '落实', '深化', '规范', '扩大', '积极', '稳步',
        '全面', '发展', '牵头', '提高', '做好',
        "支持", "增加", "加强", "进一步", "推动", "推进", "提升", "促进",
        "保障", "开展", "实施", "深化", "搞好", "持续", "深入", "强化",
        "必须", "加快", "优化", "着力", "确保", "推动落实", "不断", "建设",
        '工程', '项目', '行动', '工作', '服务', '机制', '体系', '制度', '改革',
        '政策', '措施', '管理', '重点', '问题', '方面', '任务', '目标', '内容',
        '成效', '基础', '能力', '水平', '路径', '作用', '要求', '建设', '治理',
        '国家', '社会', '政府', '地方', '各级', '相关', '组织', '部门', '机构', '单位',
        '文件', '一号', '中共中央', '党中央', '分析', '句子', '清洗', '原始', '综合',
        '数总', '后前', '两句', '投入', '坚持', '有关', '保护', '力度', '加工', '发挥', '小型',
        '切实', '推广', '条件', '稳定', '鼓励', '一批',
        '农业', '农村', '农民', '社会主义'
    }
    stopwords.update(domain_stopwords)
    return stopwords


# 全局加载停用词（只执行一次）
stop_words = load_stopwords()


def preprocess(text):
    words = jieba.lcut(text)
    return [w for w in words if len(w) > 1 and w not in stop_words]


def get_dense_topic_dist(model, bow):
    dist = np.zeros(model.num_topics)
    for topic_id, prob in model.get_document_topics(bow, minimum_probability=0.0):
        dist[topic_id] = prob
    return dist


def build_lda_matrix(prov_tokens, name, best_lda, original_stages, central_dists, dictionary):
    matrix = np.full((len(all_years), len(original_stages)), np.nan)
    diags, offs = [], []

    for i, year in enumerate(all_years):
        if year not in prov_tokens: continue
        prov_dist = get_dense_topic_dist(best_lda, dictionary.doc2bow(prov_tokens[year]))

        for j, stage in enumerate(original_stages):
            cent_dist = central_dists[stage]
            # 基于 Jensen-Shannon 散度计算相似度
            jsd = jensenshannon(prov_dist, cent_dist)
            sim = 1 - jsd
            matrix[i, j] = sim

            stage_years = stages[stage]
            if min(stage_years) <= year <= max(stage_years):
                diags.append(sim)
            else:
                offs.append(sim)

    mean_diag = np.mean(diags) if diags else 0
    mean_off = np.mean(offs) if offs else 0
    contrast = mean_diag - mean_off

    print(f"\n📊 [{name}] LDA Matrix Stats:")
    print(f"   - Diagonal Mean (Responsiveness): {mean_diag:.4f}")
    print(f"   - Off-Diagonal Mean (Noise): {mean_off:.4f}")
    print(f"   - Discriminant Contrast: {contrast:.4f}")

    return matrix


# ================= 3. 主程序入口 (多进程安全保护) =================
if __name__ == '__main__':
    print("📂 Loading documents...")
    central_dict = load_texts(central_dir, r'(\d{4}).*\.txt')
    sc_dict = load_texts(sichuan_dir, r'(\d{4})四川省一号文件\.txt')
    js_dict = load_texts(jiangsu_dir, r'(\d{4})江苏省委一号文件\.txt')

    all_docs_tokenized = []

    # 1. 预处理中央文本
    central_stage_tokens = {}
    for stage, years in stages.items():
        stage_text = "".join([central_dict.get(y, "") for y in years])
        tokens = preprocess(stage_text)
        central_stage_tokens[stage] = tokens
        all_docs_tokenized.append(tokens)

    # 2. 预处理地方文本
    sc_tokens, js_tokens = {}, {}
    for y in all_years:
        if y in sc_dict:
            t = preprocess(sc_dict[y])
            sc_tokens[y] = t
            all_docs_tokenized.append(t)
        if y in js_dict:
            t = preprocess(js_dict[y])
            js_tokens[y] = t
            all_docs_tokenized.append(t)

    # 3. 构建字典和语料库
    dictionary = corpora.Dictionary(all_docs_tokenized)
    corpus = [dictionary.doc2bow(text) for text in all_docs_tokenized]

    # 4. 训练 LDA 模型并基于 Coherence (C_v) 寻找最优 K
    print("\n🧠 Training LDA Model (Finding optimal K based on Coherence)...")
    best_lda, best_k, max_cv = None, 0, -1
    for k in [8, 12, 16]:
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, random_state=42)
        coherence_model = CoherenceModel(model=lda, texts=all_docs_tokenized, dictionary=dictionary, coherence='c_v')
        cv = coherence_model.get_coherence()
        print(f"   - K={k}, Coherence C_v = {cv:.4f}")
        if cv > max_cv:
            max_cv, best_lda, best_k = cv, lda, k

    print(f"🏆 Best LDA Model selected with K={best_k} (C_v = {max_cv:.4f})")

    # 5. 获取中央基准分布
    central_dists = {}
    for stage in original_stages:
        bow = dictionary.doc2bow(central_stage_tokens[stage])
        central_dists[stage] = get_dense_topic_dist(best_lda, bow)

    # 6. 计算 LDA 的中央-地方一致性矩阵
    sc_lda_mat = build_lda_matrix(sc_tokens, "Sichuan", best_lda, original_stages, central_dists, dictionary)
    js_lda_mat = build_lda_matrix(js_tokens, "Jiangsu", best_lda, original_stages, central_dists, dictionary)

    # ================= 7. 绘制并保存对比图表 =================
    print("\n🎨 Drawing LDA Baseline Heatmaps...")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    n_rows, n_cols = len(all_years), len(original_stages)

    # 强制将比例尺统一为 0 到 1，与主模型图片保持完全一致
    vmin, vmax = 0.0, 1.0

    fig = plt.figure(figsize=(22, 12.5))
    ax_sc = fig.add_axes([0.05, 0.12, 0.38, 0.72])
    ax_js = fig.add_axes([0.48, 0.12, 0.38, 0.72])

    stages_en_labels = {
        'Stage 1_2004-2007': 'Stage I\n2004-2007',
        'Stage 2_2008-2012': 'Stage II\n2008-2012',
        'Stage 3_2013-2017': 'Stage III\n2013-2017',
        'Stage 4_2018-2022': 'Stage IV\n2018-2022',
        'Stage 5_2023-2026': 'Stage V\n2023-2026'
    }
    cmap_dict = {'Persistence': plt.cm.Blues, 'Responsiveness': plt.cm.Reds, 'Foresight': plt.cm.Greens}

    def draw_lda_heatmap(ax, matrix, title):
        for i in range(n_rows):
            for j in range(n_cols):
                stage_name = original_stages[j]
                stage_years = stages[stage_name]
                year = all_years[i]
                value = matrix[i, j]

                if year < min(stage_years):
                    metric = 'Foresight'
                elif min(stage_years) <= year <= max(stage_years):
                    metric = 'Responsiveness'
                else:
                    metric = 'Persistence'

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

        display_stages_en = [stages_en_labels[s] for s in original_stages]
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_xticklabels(display_stages_en, fontsize=13)
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_yticklabels(all_years, fontsize=13)
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)

    draw_lda_heatmap(ax_sc, sc_lda_mat, "LDA Baseline: Policy Responsiveness (Sichuan)")
    draw_lda_heatmap(ax_js, js_lda_mat, "LDA Baseline: Policy Responsiveness (Jiangsu)")
    ax_js.set_yticklabels([])

    bar_width, bar_height, spacing, start_x = 0.015, 0.72, 0.04, 0.89
    labels_en = ['Foresight', 'Responsive-\nness', 'Persistence']
    colorbars = {'Foresight': plt.cm.Greens, 'Responsiveness': plt.cm.Reds, 'Persistence': plt.cm.Blues}

    for i, (name, cmap) in enumerate(colorbars.items()):
        x_pos = start_x + i * (bar_width + spacing)
        cbar_ax = fig.add_axes([x_pos, 0.12, bar_width, bar_height])
        cb = ColorbarBase(cbar_ax, cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

    text_positions = [start_x + (bar_width + spacing) * i + bar_width / 2 for i in range(3)]
    for x, label in zip(text_positions, labels_en):
        fig.text(x, 0.08, label, ha='center', va='top', fontsize=14, fontweight='bold')

    plt.suptitle("Baseline Comparison: LDA-based Semantic Consistency Matrix", fontsize=26, fontweight='bold', y=0.96)

    save_path = os.path.join(output_dir, 'LDA_Baseline_Matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ LDA Baseline Heatmap successfully saved to: {save_path}")