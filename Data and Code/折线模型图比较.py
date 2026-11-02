import matplotlib

matplotlib.use('Agg')  # 后台绘图模式

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# ================= 配置区域 =================
# 数据源目录
BASE_DIR = r"C:\Users\user\Desktop\word\四川省委一号文件"
# 图片保存目录
SAVE_DIR = r"C:\Users\user\Desktop\word\政策分析\修改版\图片"
YEARS = [2005, 2010, 2015, 2020, 2025]

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ================= 数据读取函数 (保持不变) =================
def find_score_column(df, model_name):
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
    all_data = []
    for year in YEARS:
        year_folder = os.path.join(BASE_DIR, f"{year}结果")
        if not os.path.exists(year_folder): continue

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

            if not found_files: continue

            try:
                df = pd.read_excel(found_files[0])
                score_col = find_score_column(df, model_name)
                scores = df[score_col].dropna().astype(float)
                if model_name == 'DeepSeek' and scores.max() > 1.0:
                    scores = scores / 10.0

                for score in scores:
                    all_data.append({'Year': year, 'Model': model_name, 'Score': score})
            except:
                pass
    return pd.DataFrame(all_data)


# ================= 核心绘图：稳定性趋势图 =================
print("🚀 正在读取数据...")
df_final = load_all_data()

if not df_final.empty:
    print(f"📊 数据加载完毕，开始绘制稳定性趋势图...")

    # 创建保存目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 设置画布
    plt.figure(figsize=(12, 8))

    # 【关键代码】绘制带误差带的折线图
    # errorbar='sd' 表示阴影区域显示“标准差”（Standard Deviation）
    # 这直接反映了数据的离散程度（即稳定性）
    sns.lineplot(
        data=df_final,
        x="Year",
        y="Score",
        hue="Model",
        style="Model",  # 不同模型用不同线型
        markers=True,  # 显示数据点
        dashes=False,  # 全部用实线
        errorbar='sd',  # 【核心】阴影代表标准差（稳定性）
        palette="Set1",  # 鲜艳的配色
        linewidth=2.5,
        markersize=8,
        alpha=0.8
    )

    # 图表装饰
    plt.title("Comparison of Semantic Consistency Stability (Mean ± SD)", fontsize=18, pad=20, weight='bold')
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Similarity Score", fontsize=14)

    # 设置刻度
    plt.xticks(YEARS)  # 强制只显示这5个年份
    plt.ylim(0.4, 1.0)  # 聚焦上半部分，看清差异

    # 网格线
    plt.grid(True, linestyle='--', alpha=0.5)

    # 图例
    plt.legend(title='Model Type', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)

    # 保存
    save_path = os.path.join(SAVE_DIR, "Model_Stability_Trend_Analysis.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 稳定性分析图已保存至: {save_path}")

else:
    print("❌ 未读到数据")