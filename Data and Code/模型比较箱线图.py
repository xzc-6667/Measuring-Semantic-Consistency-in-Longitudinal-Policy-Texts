import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import matplotlib

# 【核心修复】强制使用非交互式后端，只保存图片不弹窗，解决 PyCharm 报错
matplotlib.use('Agg')
# =================配置区域=================
# 您的根目录
BASE_DIR = r"C:\Users\user\Desktop\word\四川省委一号文件"
# 要处理的年份
YEARS = [2005, 2010, 2015, 2020, 2025]

# 设置中文字体 (Windows通常是SimHei或Microsoft YaHei)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =================核心处理函数=================
def find_score_column(df, model_name):
    """
    智能查找包含分数的列名
    """
    cols = df.columns.astype(str).tolist()

    # 1. 优先查找明确的列名
    priorities = ['max_similarity', 'similarity', 'score', 'normalized_score', '相似度', '得分', '归一化得分']
    for p in priorities:
        for c in cols:
            if p in c.lower():
                return c

    # 2. 如果是DeepSeek，找含有数字的列
    if 'deepseek' in model_name.lower():
        # 假设DeepSeek结果里有一列是最终评分
        for c in cols:
            if '评分' in c or 'score' in c:
                return c

    # 3. 实在找不到，返回最后一列（通常是计算结果）
    return cols[-1]


def load_all_data():
    all_data = []

    for year in YEARS:
        year_folder = os.path.join(BASE_DIR, f"{year}结果")
        print(f"📂 正在处理: {year_folder}")

        if not os.path.exists(year_folder):
            print(f"⚠️ 警告: 文件夹不存在 {year_folder}")
            continue

        # 定义该年份下的文件规则
        file_rules = {
            'BERT': f"*{year}*bert关键词匹配结果*.xlsx",
            'BGE': f"*{year}*bge关键词匹配结果*.xlsx",
            'E5': f"*{year}*E5关键词匹配结果*.xlsx",
            'DeepSeek': "LLM_Ollama_deepseek_上下文评分_最终全文结果.xlsx"  # 固定文件名
        }

        for model_name, pattern in file_rules.items():
            # 构建完整路径（支持文件名中可能存在的微小差异，使用glob搜索）
            search_path = os.path.join(year_folder, pattern)
            found_files = glob.glob(search_path)

            # DeepSeek文件名是固定的，不需要glob模糊搜索，直接拼接
            if model_name == 'DeepSeek':
                target_file = os.path.join(year_folder, pattern)
                if os.path.exists(target_file):
                    found_files = [target_file]
                else:
                    found_files = []

            if not found_files:
                print(f"  ❌ 未找到 {model_name} 文件: {pattern}")
                continue

            file_path = found_files[0]  # 取第一个匹配到的文件

            try:
                df = pd.read_excel(file_path)
                score_col = find_score_column(df, model_name)

                # 提取分数数据
                scores = df[score_col].dropna().astype(float)

                # 特殊处理：如果DeepSeek是0-10分，归一化到0-1
                if model_name == 'DeepSeek' and scores.max() > 1.0:
                    scores = scores / 10.0

                # 添加到总数据列表
                for score in scores:
                    all_data.append({
                        'Year': year,
                        'Model': model_name,
                        'Score': score
                    })
                print(f"  ✅ {model_name} 加载成功 (列名: {score_col}, 数量: {len(scores)})")

            except Exception as e:
                print(f"  ❌ 读取错误 {model_name}: {e}")

    return pd.DataFrame(all_data)


# =================主程序=================
print("🚀 开始读取数据...")
df_final = load_all_data()

if df_final.empty:
    print("❌ 没有加载到任何数据，请检查路径是否正确！")
else:
    print(f"\n📊 数据加载完毕，共 {len(df_final)} 条记录。开始绘图...")

    # 设置画布大小
    plt.figure(figsize=(14, 10))  # 高度稍微增加一点，给拉伸留空间


    # ================= 核心修改：定义非线性比例尺 =================
    # 原理：使用 x 的 4 次方进行映射。
    # 0.5 的 4 次方是 0.0625（只占底部的 6%）
    # 0.8 的 4 次方是 0.41（占 41%）
    # 这样 0.8-1.0 的高分区间就占据了屏幕上方约 60% 的空间！
    def forward(x):
        return x ** 4


    def inverse(x):
        return x ** (1 / 4)


    # 应用非线性比例尺
    plt.yscale('function', functions=(forward, inverse))
    # ==========================================================

    # 绘制分组箱线图
    sns.boxplot(
        data=df_final,
        x="Year",
        y="Score",
        hue="Model",
        palette="Set2",
        width=0.7,
        linewidth=1.5,
        showfliers=True,
        flierprops={'marker': 'o', 'markerfacecolor': 'white', 'markersize': 3, 'markeredgecolor': 'grey'},
        # showmeans=True,  # 【新增】显示平均分
        # meanprops={"marker": "^", "markerfacecolor": "white", "markeredgecolor": "black"}
        # 【新增】平均分用白色三角形表示
    )

    # 图表美化
    plt.title("Comparison of Policy Semantic Responsiveness Distributions Across Four Models", fontsize=20, pad=20, weight='bold')
    plt.xlabel("Year", fontsize=16)
    plt.ylabel("Similarity Score", fontsize=16)

    # ================= 核心修改：手动设置刻度 =================
    # 因为坐标轴被拉伸了，自动生成的刻度可能会很怪，我们需要手动指定
    # 这样读者能清楚看到 0.8-0.9 之间的间隔变宽了
    custom_ticks = [0, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    plt.yticks(custom_ticks, labels=[str(t) for t in custom_ticks], fontsize=12)

    # 开启网格线，一定要开启！这样审稿人能看到网格线是“上疏下密”的，说明你没有造假
    plt.grid(axis='y', linestyle='--', alpha=0.5, which='major')
    # ==========================================================

    # 锁定 Y 轴范围（稍微留点余量防止顶格）
    plt.ylim(0, 1.02)

    # 图例设置
    plt.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=12)

    # 保存与显示
    # ================= 修改保存路径 =================
    # 1. 定义新的保存目录
    SAVE_DIR = r"C:\Users\user\Desktop\word\政策分析\修改版\图片"

    # 2. 自动创建文件夹 (如果不存在的话)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"📂 已自动创建保存目录: {SAVE_DIR}")

    # 3. 拼接完整文件路径
    save_path = os.path.join(SAVE_DIR, "Model_Comparison_Nonlinear_Scale.png")
    # ===============================================

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 图片已保存至: {save_path}")
    plt.show()
