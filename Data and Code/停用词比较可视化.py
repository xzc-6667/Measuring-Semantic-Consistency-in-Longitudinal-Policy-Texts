import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib

# 使用 Agg 后端
matplotlib.use('Agg')
# ================= 1. 配置参数与数据 =================
# 输出路径 (保持不变)
OUTPUT_DIR = r'C:\Users\user\Desktop\word\政策分析\修改版\图片'

# 确保文件夹存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 数据
data = {
    'Year': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021,
             2022, 2023, 2024, 2025],
    'No_Stop': [0.7365, 0.6676, 0.6967, 0.7467, 0.7529, 0.7133, 0.7142, 0.7101, 0.7469, 0.6822, 0.7089, 0.7177, 0.6824,
                0.7190, 0.7268, 0.7017, 0.6751, 0.7016, 0.6671, 0.7171, 0.7296, 0.6830],
    'With_Stop': [0.7203, 0.7063, 0.7040, 0.7848, 0.7374, 0.7146, 0.7314, 0.7937, 0.7484, 0.7049, 0.7186, 0.7413,
                  0.7212, 0.7091, 0.7184, 0.7319, 0.6867, 0.7150, 0.6770, 0.7512, 0.7404, 0.7057]
}
df = pd.DataFrame(data)
df['Diff'] = df['With_Stop'] - df['No_Stop']

# ================= 2. 学术绘图设置 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 颜色定义
color_ours = '#d62728'  # Red
color_base = '#1f77b4'  # Blue
color_pos = '#d62728'
color_neg = '#2ca02c'

# ================= 3. 开始绘图 =================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# --- 上图：趋势对比 ---
ax1.plot(df['Year'], df['With_Stop'], marker='o', markersize=5, linestyle='-',
         color=color_ours, label='With Policy Stopwords (Ours)', linewidth=2)
ax1.plot(df['Year'], df['No_Stop'], marker='s', markersize=5, linestyle='--',
         color=color_base, label='Without Policy Stopwords (Baseline)', linewidth=1.5, alpha=0.8)

# 填充差异
ax1.fill_between(df['Year'], df['No_Stop'], df['With_Stop'], where=(df['With_Stop'] > df['No_Stop']),
                 interpolate=True, color=color_ours, alpha=0.1)
ax1.fill_between(df['Year'], df['No_Stop'], df['With_Stop'], where=(df['With_Stop'] <= df['No_Stop']),
                 interpolate=True, color=color_base, alpha=0.1)

# 设置标签
ax1.set_ylabel('BERT Semantic Similarity', fontsize=14, fontname='Times New Roman')
ax1.set_title('Comparison of Keyword Extraction Performance (2004-2025)', fontsize=16, fontweight='bold', pad=15)

# 【修改点在这里】：图例移到右上角，loc='upper right'
ax1.legend(loc='upper right', frameon=True, fontsize=11, edgecolor='gray', framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.6)

# 【优化点】：适当增加Y轴上限，给右上角的图例留出“呼吸空间”，防止挡住最高点
y_max = max(df['No_Stop'].max(), df['With_Stop'].max()) + 0.05
y_min = min(df['No_Stop'].min(), df['With_Stop'].min()) - 0.02
ax1.set_ylim(y_min, y_max)

# --- 下图：差异柱状图 ---
bar_colors = [color_pos if x >= 0 else color_neg for x in df['Diff']]
ax2.bar(df['Year'], df['Diff'], color=bar_colors, alpha=0.8, width=0.6)
ax2.axhline(0, color='black', linewidth=0.8)

ax2.set_ylabel('Similarity Gain\n(With - Without)', fontsize=12, fontname='Times New Roman')
ax2.set_xlabel('Year', fontsize=14, fontname='Times New Roman')
ax2.grid(True, linestyle=':', alpha=0.6, axis='y')

plt.xticks(df['Year'], rotation=45)
plt.tight_layout()

# ================= 4. 保存 =================
save_name_png = os.path.join(OUTPUT_DIR, 'Figure_Stopword_Comparison_English_TR.png')
# save_name_pdf = os.path.join(OUTPUT_DIR, 'Figure_Stopword_Comparison_English_TR.pdf')

plt.savefig(save_name_png, dpi=300, bbox_inches='tight')
# plt.savefig(save_name_pdf, bbox_inches='tight')

print(f"✅ 图片已更新 (图例在右上角)！")
print(f"   PNG: {save_name_png}")
# print(f"   PDF: {save_name_pdf}")
