import os
import re
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import CountVectorizer
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


# ================= 2. 函数定义区 =================
def load_texts(directory, pattern):
    res = {}
    if not os.path.exists(directory): return res
    for fname in os.listdir(directory):
        match = re.match(pattern, fname)
        if match:
            with open(os.path.join(directory, fname), 'r', encoding='utf-8') as f:
                res[int(match.group(1))] = f.read().strip()
    return res


def split_sentences(text):
    return [s.strip() for s in re.sub(r'[。！？；?!;]', '。', text).split('。') if s.strip()]


def load_stopwords():
    stopwords = set()
    hit_stopwords_path = r"C:\Users\user\Desktop\word\chinese_stopwords.txt"
    try:
        with open(hit_stopwords_path, "r", encoding="utf-8") as f:
            for line in f:
                stopwords.add(line.strip())
    except FileNotFoundError:
        print(f"⚠️ 未找到哈工大停用词文件，请检查路径: {hit_stopwords_path}")

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


# 全局加载停用词
stop_words = load_stopwords()


def preprocess_for_bertopic(text):
    words = jieba.lcut(text)
    clean_words = [w for w in words if len(w) > 1 and w not in stop_words]
    return " ".join(clean_words)


# ================= 3. 主程序入口 (防多进程报错) =================
if __name__ == '__main__':
    print("📂 Loading documents...")
    central_dict = load_texts(central_dir, r'(\d{4}).*\.txt')
    sc_dict = load_texts(sichuan_dir, r'(\d{4})四川省一号文件\.txt')
    js_dict = load_texts(jiangsu_dir, r'(\d{4})江苏省委一号文件\.txt')

    print("✂️ Splitting sentences and applying domain-specific denoising...")
    all_sentences = []
    doc_metadata = []  # 记录每一句话的来源: ("Central"/"Sichuan"/"Jiangsu", stage/year)

    # 处理中央文件
    for stage, years in stages.items():
        stage_text = "".join([central_dict.get(y, "") for y in years])
        sents = split_sentences(stage_text)
        for s in sents:
            clean_s = preprocess_for_bertopic(s)
            if len(clean_s.strip()) > 0:
                all_sentences.append(clean_s)
                doc_metadata.append(("Central", stage))

    # 处理地方文件
    for y in all_years:
        if y in sc_dict:
            sents = split_sentences(sc_dict[y])
            for s in sents:
                clean_s = preprocess_for_bertopic(s)
                if len(clean_s.strip()) > 0:
                    all_sentences.append(clean_s)
                    doc_metadata.append(("Sichuan", y))

        if y in js_dict:
            sents = split_sentences(js_dict[y])
            for s in sents:
                clean_s = preprocess_for_bertopic(s)
                if len(clean_s.strip()) > 0:
                    all_sentences.append(clean_s)
                    doc_metadata.append(("Jiangsu", y))

    # ================= 4. 训练 BERTopic 模型 =================
    print(f"\n🧠 Training BERTopic on {len(all_sentences)} sentences...")
    embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')
    vectorizer_model = CountVectorizer(stop_words=list(stop_words))

    topic_model = BERTopic(embedding_model=embedding_model,
                           vectorizer_model=vectorizer_model,
                           language="multilingual",
                           calculate_probabilities=False)

    topics, _ = topic_model.fit_transform(all_sentences)
    num_topics = max(topics) + 2
    print(f"🏆 BERTopic generated {num_topics - 1} distinct valid topics (excluding noise topic -1).")

    # ================= 5. 构建分布与计算矩阵 =================
    print("\n📈 Constructing Topic Distributions and Similarity Matrices...")


    def get_topic_distribution(target_source, target_key):
        dist = np.zeros(num_topics)
        count = 0
        for i, (source, key) in enumerate(doc_metadata):
            if source == target_source and key == target_key:
                dist[topics[i] + 1] += 1
                count += 1
        if count > 0:
            dist = dist / count
        return dist


    central_dists = {stage: get_topic_distribution("Central", stage) for stage in original_stages}


    def build_bertopic_matrix(prov_name):
        matrix = np.full((len(all_years), len(original_stages)), np.nan)
        diags, offs = [], []

        for i, year in enumerate(all_years):
            prov_dist = get_topic_distribution(prov_name, year)
            if np.sum(prov_dist) == 0: continue

            for j, stage in enumerate(original_stages):
                cent_dist = central_dists[stage]
                jsd = jensenshannon(prov_dist, cent_dist)
                sim = 1 - jsd
                matrix[i, j] = sim

                stage_years = stages[stage]
                if min(stage_years) <= year <= max(stage_years):
                    diags.append(sim)
                else:
                    offs.append(sim)

        contrast = np.mean(diags) - np.mean(offs) if diags and offs else 0
        print(f"\n📊 [{prov_name}] BERTopic Matrix Stats:")
        print(f"   - Diagonal Mean (Responsiveness): {np.mean(diags):.4f}")
        print(f"   - Off-Diagonal Mean (Noise): {np.mean(offs):.4f}")
        print(f"   - Discriminant Contrast: {contrast:.4f}")
        return matrix, diags, offs


    sc_bt_mat, sc_diags, sc_offs = build_bertopic_matrix("Sichuan")
    js_bt_mat, js_diags, js_offs = build_bertopic_matrix("Jiangsu")

    # ================= 6. 计算结构性指数 (PDI & PEI) =================
    r_bt_all, p_bt_all, f_bt_all = [], [], []
    for prov_name, prov_mat in [("Sichuan", sc_bt_mat), ("Jiangsu", js_bt_mat)]:
        for i, year in enumerate(all_years):
            for j, stage in enumerate(original_stages):
                val = prov_mat[i, j]
                if np.isnan(val): continue

                stage_years = stages[stage]
                if year < min(stage_years):
                    f_bt_all.append(val)
                elif min(stage_years) <= year <= max(stage_years):
                    r_bt_all.append(val)
                else:
                    p_bt_all.append(val)

    mean_r_bt = np.mean(r_bt_all)
    mean_p_bt = np.mean(p_bt_all)
    mean_f_bt = np.mean(f_bt_all)

    pdi_bt = mean_p_bt / mean_r_bt if mean_r_bt > 0 else np.nan
    pei_bt = mean_f_bt / mean_r_bt if mean_r_bt > 0 else np.nan

    print("\n📉 Structural Indices (PDI & PEI) for BERTopic Baseline:")
    print(f"   - BERTopic Global Responsiveness (R): {mean_r_bt:.4f}")
    print(f"   - BERTopic Global Persistence (P):    {mean_p_bt:.4f}")
    print(f"   - BERTopic Global Foresight (F):      {mean_f_bt:.4f}")
    print(f"\n   🔥 BERTopic Path Dependence Index (PDI): {pdi_bt:.4f}")
    print(f"   🔥 BERTopic Policy Exploration Index (PEI): {pei_bt:.4f}")
    print("=======================================================================")

    # ================= 7. 绘制并保存 BERTopic 热力图 =================
    print("\n🎨 Drawing BERTopic Baseline Heatmaps...")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    n_rows, n_cols = len(all_years), len(original_stages)

    # ⚠️ 强制将比例尺统一为 0 到 1，与主模型图片保持完全一致
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


    def draw_heatmap(ax, matrix, title):
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


    draw_heatmap(ax_sc, sc_bt_mat, "BERTopic Baseline: Policy Responsiveness (Sichuan)")
    draw_heatmap(ax_js, js_bt_mat, "BERTopic Baseline: Policy Responsiveness (Jiangsu)")
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

    plt.suptitle("Baseline Comparison: BERTopic-based Semantic Consistency Matrix", fontsize=26, fontweight='bold',
                 y=0.96)

    save_path = os.path.join(output_dir, 'BERTopic_Baseline_Matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ BERTopic Baseline Heatmap successfully saved to: {save_path}")