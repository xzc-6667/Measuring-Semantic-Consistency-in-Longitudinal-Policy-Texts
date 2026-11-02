import os
import re
import json
import numpy as np
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 基础配置与加载模型 ==========
print("🔍 Loading BERT model...")
model = SentenceTransformer('shibing624/text2vec-base-chinese')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

base_path = r'C:\Users\user\Desktop\word\农业农村部一号文件'
jiangsu_dir = r'C:\Users\user\Desktop\word\江苏省委一号文件'
sichuan_dir = r'C:\Users\user\Desktop\word\四川省委一号文件'
keywords_dir = os.path.join(base_path, '扩展语义簇')
output_dir = os.path.join(base_path, '图片', '省级响应')
os.makedirs(output_dir, exist_ok=True)

stages = {
    'Stage 1_2004-2007': list(range(2004, 2008)),
    'Stage 2_2008-2012': list(range(2008, 2013)),
    'Stage 3_2013-2017': list(range(2013, 2018)),
    'Stage 4_2018-2022': list(range(2018, 2023)),
    'Stage 5_2023-2026': list(range(2023, 2027)),
}
all_years = list(range(2004, 2027))
original_stages = list(stages.keys())

# ========== 2. 读取语义簇与文本 ==========
stage_clusters = {}
for stage in original_stages:
    with open(os.path.join(keywords_dir, f'{stage}_clusters.json'), 'r', encoding='utf-8') as f:
        stage_clusters[stage] = json.load(f)

stage_centroids = {}
for stage in original_stages:
    centroid_vecs = []
    for core_word, data in stage_clusters[stage].items():
        words = [core_word] + list(data['related'].keys())
        weights = [data['core_weight']] + list(data['related'].values())
        embs = model.encode(words, convert_to_tensor=True)
        wt = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)
        centroid = torch.sum(embs * wt, dim=0) / torch.sum(wt)
        centroid_vecs.append(centroid.cpu().numpy())
    stage_centroids[stage] = np.array(centroid_vecs)


def load_texts(directory, pattern):
    res = {}
    for fname in os.listdir(directory):
        match = re.match(pattern, fname)
        if match:
            with open(os.path.join(directory, fname), 'r', encoding='utf-8') as f:
                res[int(match.group(1))] = f.read().strip()
    return res


sc_dict = load_texts(sichuan_dir, r'(\d{4})四川省一号文件\.txt')
js_dict = load_texts(jiangsu_dir, r'(\d{4})江苏省委一号文件\.txt')


def split_sentences(text):
    return [s.strip() for s in re.sub(r'[。！？；?!;]', '。', text).split('。') if s.strip()]


# ========== 3. 极速核心：预计算所有句子的 Max Similarity ==========
# 我们只跑一次 BERT，把所有相似度存下来，之后无论测多少个阈值都是秒出结果！
print("🧠 Pre-computing all sentence similarities (This takes ~10 mins)...")


def precompute_max_sims(province_dict):
    # 结构: dict[year][stage] = [list of max_sim for each sentence]
    sim_data = {}
    for year in all_years:
        content = province_dict.get(year, "")
        sents = split_sentences(content)
        if not sents:
            sim_data[year] = None
            continue
        sent_vecs = model.encode(sents, convert_to_tensor=True)
        year_stage_sims = {}
        for stage in original_stages:
            centroids = stage_centroids[stage]
            # 计算该年所有句子与该阶段所有质心的相似度矩阵
            sim_matrix = cosine_similarity(sent_vecs.cpu().numpy(), centroids)
            # 取每个句子的最大相似度
            max_sims = np.max(sim_matrix, axis=1)
            year_stage_sims[stage] = max_sims
        sim_data[year] = year_stage_sims
    return sim_data


sc_sim_data = precompute_max_sims(sc_dict)
js_sim_data = precompute_max_sims(js_dict)

# ========== 4. 阈值寻优 (0.1秒即可完成) ==========
print("🚀 Optimizing threshold for maximum discriminant validity...")
thresholds_to_test = np.arange(0.45, 0.55, 0.002)  # 测试 0.48 到 0.58 之间所有值
results = []

for tau in thresholds_to_test:
    sc_diags, sc_offs = [], []

    # 计算四川的对角线与非对角线均值
    for i, year in enumerate(all_years):
        if not sc_sim_data[year]: continue
        for j, stage in enumerate(original_stages):
            max_sims = sc_sim_data[year][stage]
            match_ratio = np.mean(max_sims >= tau)  # 超过阈值的句子比例

            stage_years = stages[stage]
            if min(stage_years) <= year <= max(stage_years):
                sc_diags.append(match_ratio)
            else:
                sc_offs.append(match_ratio)

    mean_diag = np.mean(sc_diags)
    mean_off = np.mean(sc_offs)
    contrast = mean_diag - mean_off
    results.append(
        {'tau': tau, 'Diagonal (Responsiveness)': mean_diag, 'Off-Diagonal (Noise)': mean_off, 'Contrast': contrast})

df_res = pd.DataFrame(results)

# 找出最优阈值
optimal_row = df_res.loc[df_res['Contrast'].idxmax()]
best_tau = optimal_row['tau']
best_diag = optimal_row['Diagonal (Responsiveness)']

print(f"\n✅ Optimization Complete!")
print(f"🏆 Optimal Threshold (Tau): {best_tau:.3f}")
print(f"📈 Expected Responsiveness (Diagonal Mean): {best_diag:.3f}")
print(f"📉 Expected Off-Diagonal Mean: {optimal_row['Off-Diagonal (Noise)']:.3f}")
print(f"⭐ Max Contrast Achieved: {optimal_row['Contrast']:.3f}")

# ========== 5. 绘制寻优曲线图 (可以直接放入论文) ==========
plt.figure(figsize=(10, 6))
plt.plot(df_res['tau'], df_res['Diagonal (Responsiveness)'], label='Diagonal (Sync Responsiveness)', color='red',
         linewidth=2)
plt.plot(df_res['tau'], df_res['Off-Diagonal (Noise)'], label='Off-Diagonal (Foresight/Persistence)', color='blue',
         linewidth=2)
plt.plot(df_res['tau'], df_res['Contrast'], label='Discriminant Contrast (Difference)', color='green', linestyle='--',
         linewidth=2.5)

plt.axvline(x=best_tau, color='black', linestyle=':', linewidth=2)
plt.text(best_tau + 0.002, optimal_row['Contrast'] + 0.02,
         f"Optimal $\\tau$ = {best_tau:.3f}\nDiag Mean $\\approx$ {best_diag:.2f}", fontweight='bold')

plt.title("Threshold Optimization: Maximizing Discriminant Validity", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Similarity Threshold ($\\tau$)", fontsize=14)
plt.ylabel("Mean Ratio Score", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

save_plot = os.path.join(output_dir, 'Threshold_Optimization_Curve.png')
plt.savefig(save_plot, dpi=300, bbox_inches='tight')
print(f"🖼️ Curve saved to: {save_plot}")

# D:\anaconda3\envs\lda_env\python.exe "D:\python\PyCharm 2021.3.2\plugins\python\helpers\pydev\pydevconsole.py" --mode=client --port=26906
# import sys; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['H:\\pythonProject', 'H:/pythonProject'])
# PyDev console: starting.
# Python 3.9.21 (main, Dec 11 2024, 16:35:24) [MSC v.1929 64 bit (AMD64)] on win32
# runfile('H:/pythonProject/中央一号文件/基线模型比较/阈值设定.py', wdir='H:/pythonProject/中央一号文件/基线模型比较')
# 🔍 Loading BERT model...
# 🧠 Pre-computing all sentence similarities (This takes ~10 mins)...
# 🚀 Optimizing threshold for maximum discriminant validity...
# ✅ Optimization Complete!
# 🏆 Optimal Threshold (Tau): 0.548
# 📈 Expected Responsiveness (Diagonal Mean): 0.597
# 📉 Expected Off-Diagonal Mean: 0.565
# ⭐ Max Contrast Achieved: 0.032
# Traceback (most recent call last):
#   File "D:\anaconda3\envs\lda_env\lib\code.py", line 90, in runcode
#     exec(code, self.locals)
#   File "<input>", line 1, in <module>
#   File "D:\python\PyCharm 2021.3.2\plugins\python\helpers\pydev\_pydev_bundle\pydev_umd.py", line 198, in runfile
#     pydev_imports.execfile(filename, global_vars, local_vars)  # execute the script
#   File "D:\python\PyCharm 2021.3.2\plugins\python\helpers\pydev\_pydev_imps\_pydev_execfile.py", line 18, in execfile
#     exec(compile(contents+"\n", file, 'exec'), glob, loc)
#   File "H:/pythonProject/中央一号文件/基线模型比较/阈值设定.py", line 145, in <module>
#     plt.figure(figsize=(10, 6))
#   File "C:\Users\user\AppData\Roaming\Python\Python39\site-packages\matplotlib\pyplot.py", line 1027, in figure
#     manager = new_figure_manager(
#   File "C:\Users\user\AppData\Roaming\Python\Python39\site-packages\matplotlib\pyplot.py", line 549, in new_figure_manager
#     _warn_if_gui_out_of_main_thread()
#   File "C:\Users\user\AppData\Roaming\Python\Python39\site-packages\matplotlib\pyplot.py", line 526, in _warn_if_gui_out_of_main_thread
#     canvas_class = cast(type[FigureCanvasBase], _get_backend_mod().FigureCanvas)
#   File "C:\Users\user\AppData\Roaming\Python\Python39\site-packages\matplotlib\pyplot.py", line 358, in _get_backend_mod
#     switch_backend(rcParams._get("backend"))
#   File "C:\Users\user\AppData\Roaming\Python\Python39\site-packages\matplotlib\pyplot.py", line 416, in switch_backend
#     canvas_class = module.FigureCanvas
# AttributeError: module 'backend_interagg' has no attribute 'FigureCanvas'
# import matplotlib
# # 强制使用标准的无头后端，绕开 PyCharm 拦截
# matplotlib.use('Agg', force=True)
# import matplotlib.pyplot as plt
# import os
# # 重新绘制寻优曲线图
# plt.figure(figsize=(10, 6))
# plt.plot(df_res['tau'], df_res['Diagonal (Responsiveness)'], label='Diagonal (Sync Responsiveness)', color='red', linewidth=2)
# plt.plot(df_res['tau'], df_res['Off-Diagonal (Noise)'], label='Off-Diagonal (Foresight/Persistence)', color='blue', linewidth=2)
# plt.plot(df_res['tau'], df_res['Contrast'], label='Discriminant Contrast (Difference)', color='green', linestyle='--', linewidth=2.5)
# plt.axvline(x=best_tau, color='black', linestyle=':', linewidth=2)
# plt.text(best_tau + 0.002, optimal_row['Contrast'] + 0.02, f"Optimal $\\tau$ = {best_tau:.3f}\nDiag Mean $\\approx$ {best_diag:.2f}", fontweight='bold')
# plt.title("Threshold Optimization: Maximizing Discriminant Validity", fontsize=16, fontweight='bold', pad=15)
# plt.xlabel("Similarity Threshold ($\\tau$)", fontsize=14)
# plt.ylabel("Mean Ratio Score", fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.6)
# # 保存图片
# output_dir = r'C:\Users\user\Desktop\word\农业农村部一号文件\图片\省级响应'
# save_plot = os.path.join(output_dir, 'Threshold_Optimization_Curve.png')
# plt.savefig(save_plot, dpi=300, bbox_inches='tight')
# print(f"🖼️ 完美！寻优曲线图已成功保存至: {save_plot}")
# 🖼️ 完美！寻优曲线图已成功保存至: C:\Users\user\Desktop\word\农业农村部一号文件\图片\省级响应\Threshold_Optimization_Curve.png
# import matplotlib
# matplotlib.use('Agg', force=True)
# import matplotlib.pyplot as plt
# import os
# import numpy as np
# import pandas as pd
# # 1. 扩大阈值搜索区间：从 0.45 测到 0.75
# print("🚀 Re-optimizing threshold in an expanded range [0.45, 0.75]...")
# thresholds_to_test = np.arange(0.45, 0.75, 0.002)
# results = []
# for tau in thresholds_to_test:
#     sc_diags, sc_offs = [], []
#     for i, year in enumerate(all_years):
#         if not sc_sim_data[year]: continue
#         for j, stage in enumerate(original_stages):
#             max_sims = sc_sim_data[year][stage]
#             match_ratio = np.mean(max_sims >= tau)
#             stage_years = stages[stage]
#             if min(stage_years) <= year <= max(stage_years):
#                 sc_diags.append(match_ratio)
#             else:
#                 sc_offs.append(match_ratio)
#     mean_diag = np.mean(sc_diags)
#     mean_off = np.mean(sc_offs)
#     contrast = mean_diag - mean_off
#     results.append(
#         {'tau': tau, 'Diagonal (Responsiveness)': mean_diag, 'Off-Diagonal (Noise)': mean_off, 'Contrast': contrast})
# df_res = pd.DataFrame(results)
# # 2. 找出新的全局最优阈值
# optimal_row = df_res.loc[df_res['Contrast'].idxmax()]
# best_tau = optimal_row['tau']
# best_diag = optimal_row['Diagonal (Responsiveness)']
# print(f"\n✅ Optimization Complete!")
# print(f"🏆 True Optimal Threshold (Tau): {best_tau:.3f}")
# print(f"📈 Expected Responsiveness (Diagonal Mean): {best_diag:.3f}")
# print(f"⭐ Max Contrast Achieved: {optimal_row['Contrast']:.3f}")
# # 3. 重新绘制寻优曲线图
# plt.figure(figsize=(10, 6))
# plt.plot(df_res['tau'], df_res['Diagonal (Responsiveness)'], label='Diagonal (Sync Responsiveness)', color='red',
#          linewidth=2)
# plt.plot(df_res['tau'], df_res['Off-Diagonal (Noise)'], label='Off-Diagonal (Foresight/Persistence)', color='blue',
#          linewidth=2)
# plt.plot(df_res['tau'], df_res['Contrast'], label='Discriminant Contrast (Difference)', color='green', linestyle='--',
#          linewidth=2.5)
# plt.axvline(x=best_tau, color='black', linestyle=':', linewidth=2)
# plt.text(best_tau + 0.005, optimal_row['Contrast'] + 0.02,
#          f"Optimal $\\tau$ = {best_tau:.3f}\nDiag Mean $\\approx$ {best_diag:.2f}", fontweight='bold')
# plt.title("Threshold Optimization: Maximizing Discriminant Validity", fontsize=16, fontweight='bold', pad=15)
# plt.xlabel("Similarity Threshold ($\\tau$)", fontsize=14)
# plt.ylabel("Mean Ratio Score", fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.6)
# # 保存图片
# save_plot = os.path.join(output_dir, 'Threshold_Optimization_Curve_Expanded.png')
# plt.savefig(save_plot, dpi=300, bbox_inches='tight')
# print(f"🖼️ 完美的寻优曲线图已成功保存至: {save_plot}")
# 🚀 Re-optimizing threshold in an expanded range [0.45, 0.75]...
# ✅ Optimization Complete!
# 🏆 True Optimal Threshold (Tau): 0.558
# 📈 Expected Responsiveness (Diagonal Mean): 0.512
# ⭐ Max Contrast Achieved: 0.033
# 🖼️ 完美的寻优曲线图已成功保存至: C:\Users\user\Desktop\word\农业农村部一号文件\图片\省级响应\Threshold_Optimization_Curve_Expanded.png
# 进程已结束,退出代码0