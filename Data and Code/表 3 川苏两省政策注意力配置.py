import numpy as np
import pandas as pd
import os

# ================= 1. 读取数据 =================
base_dir = r'C:\Users\user\Desktop\word\农业农村部一号文件\图片\省级响应'
# 这里的 data_path 是字符串路径
data_path = os.path.join(base_dir, 'heatmap_matrices.npz')

# 💡 关键修正：使用 np.load() 加载文件内容
data = np.load(data_path)

# 现在可以正常提取矩阵了
sc_f, sc_r, sc_p = data['sc_f'], data['sc_r'], data['sc_p']
js_f, js_r, js_p = data['js_f'], data['js_r'], data['js_p']

# 定义阶段时间跨度
stages_info = [
    ("Stage I", "2004-2007", range(0, 4)),
    ("Stage II", "2008-2012", range(4, 9)),
    ("Stage III", "2013-2017", range(9, 14)),
    ("Stage IV", "2018-2022", range(14, 19)),
    ("Stage V", "2023-2026", range(19, 23)),
]

def calculate_indices(r_mat, p_mat, f_mat):
    results = []
    for i, (name, period, rows) in enumerate(stages_info):
        # 1. 计算当期响应均值 R
        r_mean = np.nanmean(r_mat[rows, i])

        # 2. 计算持续性均值 P (当前行在之前阶段的均值)
        p_vals = p_mat[np.ix_(rows, range(0, i))] if i > 0 else np.array([])
        p_mean = np.nanmean(p_vals) if p_vals.size > 0 else np.nan

        # 3. 计算前瞻性均值 F (当前行在之后阶段的均值)
        f_vals = f_mat[np.ix_(rows, range(i + 1, 5))] if i < 4 else np.array([])
        f_mean = np.nanmean(f_vals) if f_vals.size > 0 else np.nan

        # 构建指数
        results.append({
            "政策阶段": f"{name} ({period})",
            "响应性 (R)": r_mean,
            "路径依赖指数 (P/R)": p_mean / r_mean if not np.isnan(p_mean) else np.nan,
            "政策探索指数 (F/R)": f_mean / r_mean if not np.isnan(f_mean) else np.nan
        })
    return results

# 分省计算
sc_res = calculate_indices(sc_r, sc_p, sc_f)
js_res = calculate_indices(js_r, js_p, js_f)

# 数据整合
final_list = []
for i in range(5):
    row_sc = sc_res[i].copy()
    row_sc["省份"] = "四川"
    row_js = js_res[i].copy()
    row_js["省份"] = "江苏"
    final_list.extend([row_sc, row_js])

df = pd.DataFrame(final_list)

# 数值格式化
df["响应性 (R)"] = df["响应性 (R)"].map(lambda x: f"{x:.3f}")
df["路径依赖指数 (P/R)"] = df["路径依赖指数 (P/R)"].map(lambda x: f"{x:.3f}" if not np.isnan(x) else "-")
df["政策探索指数 (F/R)"] = df["政策探索指数 (F/R)"].map(lambda x: f"{x:.3f}" if not np.isnan(x) else "-")

# 调整列顺序并保存
df = df[["政策阶段", "省份", "响应性 (R)", "路径依赖指数 (P/R)", "政策探索指数 (F/R)"]]
# 保存到 base_dir 目录下
output_path = os.path.join(base_dir, "Policy_Attention_Indices_Table.csv")
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"✅ 表 3 已生成并保存为: {output_path}")
print(df.to_markdown(index=False))