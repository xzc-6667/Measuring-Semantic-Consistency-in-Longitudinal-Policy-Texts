import akshare as ak
import pandas as pd
import os
from datetime import datetime

# =====================================================================
# 2025年 9–11 月 玉米期货结构化交易数据采集脚本（完整独立版）
# =====================================================================

# -------------------------- 核心配置 --------------------------
# 合约代码（11月到期，覆盖9–11月）
CORN_CONTRACT = "C2511"
TARGET_VARIETY = "玉米"

# 采集时间范围
START_DATE = datetime(2025, 9, 1)
END_DATE = datetime(2025, 11, 30)

# 数据存储路径
BASE_SAVE_DIR = "./2025年9-11月玉米期货结构化数据"
MONTHLY_SAVE_DIR = f"{BASE_SAVE_DIR}/按月份拆分数据"
MERGED_SAVE_DIR = f"{BASE_SAVE_DIR}/合并版数据"

os.makedirs(MONTHLY_SAVE_DIR, exist_ok=True)
os.makedirs(MERGED_SAVE_DIR, exist_ok=True)

# -------------------------- 数据预处理函数 --------------------------
def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    数据预处理：
    1. 补充品种、合约、日期、月份字段
    2. 计算上日结算价、涨跌额、涨跌幅
    3. 整理为最终14字段结构
    """
    df = raw_df.copy()

    # 基础字段
    df["品种"] = TARGET_VARIETY
    df["合约代码"] = CORN_CONTRACT
    df["日期"] = df["date"].dt.strftime("%Y-%m-%d")
    df["月份"] = df["date"].dt.strftime("%Y%m")

    # 衍生指标
    df["上日结算价"] = df["settle"].shift(1)
    df["涨跌1"] = df["close"] - df["上日结算价"]
    df["涨跌2"] = (df["涨跌1"] / df["上日结算价"] * 100).round(2)

    # 空值处理
    df["上日结算价"].fillna(df["settle"].iloc[0], inplace=True)
    df["涨跌1"].fillna(0, inplace=True)
    df["涨跌2"].fillna(0, inplace=True)

    # 字段重命名与顺序整理
    final_df = df.rename(columns={
        "open": "开盘价",
        "high": "最高价",
        "low": "最低价",
        "close": "收盘价",
        "settle": "结算价",
        "volume": "成交量",
        "hold": "持仓量"
    })[
        [
            "日期", "月份", "品种", "合约代码",
            "开盘价", "最高价", "最低价",
            "收盘价", "结算价", "上日结算价",
            "涨跌1", "涨跌2",
            "成交量", "持仓量"
        ]
    ]

    return final_df


# -------------------------- 保存函数 --------------------------
def split_and_save_by_month(processed_df: pd.DataFrame):
    """按月份拆分保存 + 合并版保存"""

    # 合并版
    merged_path = f"{MERGED_SAVE_DIR}/玉米期货_2025年9-11月合并数据.csv"
    processed_df.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"✅ 合并数据保存完成：{os.path.abspath(merged_path)}")

    # 按月份拆分
    for month in sorted(processed_df["月份"].unique()):
        month_df = processed_df[processed_df["月份"] == month].reset_index(drop=True)
        month_path = f"{MONTHLY_SAVE_DIR}/玉米期货_2025年{month}_数据.csv"
        month_df.to_csv(month_path, index=False, encoding="utf-8-sig")
        print(f"✅ {month} 月数据保存完成（{len(month_df)} 条）")


# -------------------------- 核心采集函数 --------------------------
def collect_9_11_month_data():
    try:
        print(f"🔍 正在获取 {CORN_CONTRACT} 合约历史行情数据……")
        raw_df = ak.futures_zh_daily_sina(symbol=CORN_CONTRACT)
        raw_df["date"] = pd.to_datetime(raw_df["date"])

        print(
            f"📥 接口数据范围：{raw_df['date'].min().strftime('%Y-%m-%d')} "
            f"至 {raw_df['date'].max().strftime('%Y-%m-%d')}"
        )

        # 时间筛选
        mask = (raw_df["date"] >= START_DATE) & (raw_df["date"] <= END_DATE)
        filtered_df = raw_df.loc[mask].copy()

        if filtered_df.empty:
            print("⚠️ 未获取到 2025年9–11月 数据，可能尚未更新")
            return

        # 数据处理
        print("🔧 正在进行结构化处理……")
        processed_df = preprocess_data(filtered_df)

        # 保存
        print("💾 正在保存数据……")
        split_and_save_by_month(processed_df)

        # 总结
        print("\n🎉 2025年9–11月玉米期货数据采集完成！")
        print(f"📊 共 {len(processed_df)} 条交易日记录")
        print(f"📁 数据目录：{os.path.abspath(BASE_SAVE_DIR)}")

    except Exception as e:
        print(f"\n❌ 采集失败：{e}")
        print("💡 若合约失效，可尝试将 CORN_CONTRACT 改为 'C0'（连续合约）")


# -------------------------- 主程序入口 --------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("📊 玉米期货历史数据采集程序（2025年9–11月专用）")
    print(f"🔍 合约代码：{CORN_CONTRACT} | 品种：{TARGET_VARIETY}")
    print(f"🗓️ 时间范围：{START_DATE.strftime('%Y-%m-%d')} 至 {END_DATE.strftime('%Y-%m-%d')}")
    print("=" * 70)

    collect_9_11_month_data()
