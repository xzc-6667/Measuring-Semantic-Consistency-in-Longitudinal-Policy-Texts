import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置字体 =================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 读取数据 =================
base_dir = r'C:\Users\user\Desktop\word\农业农村部一号文件\图片\省级响应'
matrix_save_path = os.path.join(base_dir, 'heatmap_matrices.npz')

print("📂 正在加载底层矩阵数据...")
data = np.load(matrix_save_path)
sc_f, sc_r, sc_p = data['sc_f'], data['sc_r'], data['sc_p']
js_f, js_r, js_p = data['js_f'], data['js_r'], data['js_p']

# ================= 2. 数据重构 (用于左侧箱线图) =================
records = []
metrics = [
    ("Responsiveness", sc_r, js_r),
    ("Persistence", sc_p, js_p),
    ("Foresight", sc_f, js_f)
]

# 将矩阵中非 NaN 的有效数值提取出来，转为一维长表
for metric_name, sc_mat, js_mat in metrics:
    for val in sc_mat[~np.isnan(sc_mat)]:
        records.append({"Score": val, "Dimension": metric_name, "Province": "Sichuan"})
    for val in js_mat[~np.isnan(js_mat)]:
        records.append({"Score": val, "Dimension": metric_name, "Province": "Jiangsu"})

df_box = pd.DataFrame(records)

# ================= 3. 数据重构 (用于右侧折线图) =================
stages_labels = ['Stage I\n(04-07)', 'Stage II\n(08-12)', 'Stage III\n(13-17)', 'Stage IV\n(18-22)', 'Stage V\n(23-26)']

# axis=0 表示按列(阶段)求平均，这样就得到了 5 个阶段的动态平均值！
sc_trend = np.nanmean(sc_r, axis=0)
js_trend = np.nanmean(js_r, axis=0)

# ================= 4. 开始双画板绘图 =================
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
palette = {"Sichuan": "#FF6B6B", "Jiangsu": "#4D96FF"} # 经典的红蓝对比配色

# --------- 左图：全局分布箱线图 ---------
sns.boxplot(x="Dimension", y="Score", hue="Province", data=df_box,
            palette=palette, ax=axes[0], width=0.5, fliersize=5, linewidth=1.5)
# 叠加上散点，让数据密度更直观
sns.stripplot(x="Dimension", y="Score", hue="Province", data=df_box,
              palette=palette, ax=axes[0], dodge=True, alpha=0.3, size=4, legend=False)

axes[0].set_title("(a) Global Distribution of Attention Configurations", fontsize=16, fontweight='bold', pad=15)
axes[0].set_ylabel("Semantic Similarity Score", fontsize=13)
axes[0].set_xlabel("")
axes[0].tick_params(axis='x', labelsize=13)
axes[0].legend(title="")
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

# --------- 右图：分阶段动态演化折线图 ---------
axes[1].plot(stages_labels, sc_trend, marker='o', markersize=9, linewidth=3, color="#FF6B6B", label="Sichuan")
axes[1].plot(stages_labels, js_trend, marker='s', markersize=9, linewidth=3, color="#4D96FF", label="Jiangsu")

axes[1].set_title("(b) Temporal Evolution of Policy Responsiveness", fontsize=16, fontweight='bold', pad=15)
axes[1].set_ylabel("Stage-Average Responsiveness", fontsize=13)
axes[1].tick_params(axis='x', labelsize=12)
axes[1].legend(fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

# ================= 5. 保存图表 =================
save_path = os.path.join(base_dir, "Attention_Metrics_Distribution_Trend.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"🖼️ 完美！双面板统计图已成功保存至: {save_path}")