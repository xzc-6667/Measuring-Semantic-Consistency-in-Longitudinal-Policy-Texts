import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import matplotlib

# 强制使用非交互式后端，只保存图片不弹窗，解决 PyCharm 或服务器环境报错
matplotlib.use('Agg')

# ================= 配置区域 =================
# 数据源目录 (请确保路径正确)
BASE_DIR = r"C:\Users\user\Desktop\word\四川省委一号文件"
# 图片保存目录
SAVE_DIR = r"C:\Users\user\Desktop\word\政策分析\修改版\图片"
# 测试的时间节点
YEARS = [2005, 2010, 2015, 2020, 2025]

# 设置中文字体 (防止图表中的减号或可能出现的中文字符报错)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ================= 数据读取函数 =================
def find_score_column(df, model_name):
    """智能查找包含分数的列名"""
    cols = df.columns.astype(str).tolist()
    priorities = ['max_similarity', 'similarity', 'score', 'normalized_score', '相似度', '得分', '归一化得分']
    for p in priorities:
        for c in cols:
            if p in c.lower(): return c
    if 'deepseek' in model_name.lower():
        for c in cols:
            if '评分' in c or 'score' in c: return c
    return cols[-1]


def load_all_data():
    """读取所有年份和模型的数据集"""
    all_data = []
    for year in YEARS:
        year_folder = os.path.join(BASE_DIR, f"{year}结果")
        print(f"📂 正在处理: {year_folder}")
        if not os.path.exists(year_folder):
            print(f"⚠️ 警告: 文件夹不存在 {year_folder}")
            continue

        file_rules = {
            'BERT': f"*{year}*bert关键词匹配结果*.xlsx",
            'BGE': f"*{year}*bge关键词匹配结果*.xlsx",
            'E5': f"*{year}*E5关键词匹配结果*.xlsx",
            'DeepSeek': "LLM_Ollama_deepseek_上下文评分_最终全文结果.xlsx"
        }

        for model_name, pattern in file_rules.items():
            search_path = os.path.join(year_folder, pattern)
            found_files = glob.glob(search_path)

            if model_name == 'DeepSeek':
                target_file = os.path.join(year_folder, pattern)
                found_files = [target_file] if os.path.exists(target_file) else []

            if not found_files:
                print(f"  ❌ 未找到 {model_name} 文件")
                continue

            try:
                df = pd.read_excel(found_files[0])
                score_col = find_score_column(df, model_name)
                scores = df[score_col].dropna().astype(float)

                # 特殊处理：如果 DeepSeek 是 0-10 分，归一化到 0-1
                if model_name == 'DeepSeek' and scores.max() > 1.0:
                    scores = scores / 10.0

                for score in scores:
                    all_data.append({'Year': year, 'Model': model_name, 'Score': score})
                print(f"  ✅ {model_name} 加载成功")
            except Exception as e:
                print(f"  ❌ 读取错误 {model_name}: {e}")

    return pd.DataFrame(all_data)


# ================= 核心绘图：1x2 复合组图 =================
print("🚀 开始读取数据...")
df_final = load_all_data()

if df_final.empty:
    print("❌ 没有加载到任何数据，请检查路径！")
else:
    print(f"\n📊 数据加载完毕，开始绘制复合组图...")
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"📂 已自动创建保存目录: {SAVE_DIR}")

    # 创建 1 行 2 列的画布，宽度 20，高度 8，确保图表大气清晰
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))


    # ------------------ 左图 (a)：非线性比例尺箱线图 ------------------
    # 定义非线性映射，放大高分段区域 (0.8 - 1.0)
    def forward(x):
        return x ** 4


    def inverse(x):
        return x ** (1 / 4)


    sns.boxplot(
        data=df_final, x="Year", y="Score", hue="Model",
        palette="Set2", width=0.7, linewidth=1.5,
        showfliers=True,
        flierprops={'marker': 'o', 'markerfacecolor': 'white', 'markersize': 3, 'markeredgecolor': 'grey'},
        ax=axes[0]
    )

    # 应用非线性比例尺与自定义刻度
    axes[0].set_yscale('function', functions=(forward, inverse))
    custom_ticks = [0, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    axes[0].set_yticks(custom_ticks)
    axes[0].set_yticklabels([str(t) for t in custom_ticks], fontsize=12)
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5, which='major')

    # 左图装饰与图例
    axes[0].set_title("(a) Cross-sectional Distributions Across Models", fontsize=16, weight='bold', pad=15)
    axes[0].set_xlabel("Year", fontsize=14)
    axes[0].set_ylabel("Similarity Score", fontsize=14)
    axes[0].legend(title='Model', loc='upper left', fontsize=11, title_fontsize=12)

    # ------------------ 右图 (b)：稳定性趋势折线图 (Mean ± SD) ------------------
    sns.lineplot(
        data=df_final, x="Year", y="Score", hue="Model", style="Model",
        markers=True, dashes=False, errorbar='sd', palette="Set1",
        linewidth=2.5, markersize=8, alpha=0.8,
        ax=axes[1]
    )

    # 右图装饰与图例
    axes[1].set_title("(b) Longitudinal Stability (Mean ± SD)", fontsize=16, weight='bold', pad=15)
    axes[1].set_xlabel("Year", fontsize=14)
    axes[1].set_ylabel("Similarity Score", fontsize=14)
    axes[1].set_xticks(YEARS)
    axes[1].set_ylim(0.4, 1.0)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(title='Model Type', loc='upper left', fontsize=11, title_fontsize=12)

    # ------------------ 整体保存设置 ------------------
    # 自动调整子图间距，防止坐标轴文字重叠
    plt.tight_layout()

    # 拼接最终的图片保存路径
    save_path = os.path.join(SAVE_DIR, "Figure2_Combined_Baseline_Comparison.png")

    # bbox_inches='tight' 确保保存时边缘不被裁剪
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 复合对比组图已成功保存至: {save_path}")