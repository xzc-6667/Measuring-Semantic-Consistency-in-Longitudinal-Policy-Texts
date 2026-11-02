import pandas as pd
import numpy as np
import requests
import os
import warnings
import re

warnings.filterwarnings('ignore')

# ========== 核心配置 ==========
OLLAMA_MODEL_NAME = "deepseek-r1:14b"
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"

print(f"🔍 正在使用的Ollama API URL：{OLLAMA_API_URL}")

# 文件路径 (请确保这些路径是正确的)
KEYWORDS_FILE = r'D:\models\农业农村部一号文件\2005关键词.txt'
SENTENCE_EXCEL = r'D:\models\四川省委一号文件\2005四川省一号文件_分句结果.xlsx'
LLM_SCORE_SAVE_PATH = r'D:\models\四川省委一号文件\2005结果\LLM_Ollama_deepseek_上下文评分_最终全文结果.xlsx'

# Excel列名
SENTENCE_ID_COL = '句子编号'
SENTENCE_CONTENT_COL = '句子内容'
CONTEXT_CONTENT_COL = '上下文内容'
SCORE_COL = 'deepseek_相关度评分'
THINKING_COL = 'deepseek_思考过程'


# ========== 工具函数：净化关键词（避免无效字符） ==========
def clean_keyword(kw):
    """净化关键词：去除空格、控制字符，保留纯中文"""
    kw = re.sub(r'[\x00-\x1F\x7F]', '', kw)  # 去除控制字符
    kw = kw.strip().replace('　', '').replace(' ', '')  # 去除全角/半角空格
    kw = re.sub(r'[^\u4e00-\u9fa5]', '', kw)  # 只保留中文
    return kw


# ========== 1. 测试Ollama服务 ==========
def test_ollama_service():
    try:
        response = requests.get("http://127.0.0.1:11434", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama服务正常运行！")
            return True
        else:
            print(f"❌ 服务返回异常：状态码 {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接Ollama服务！请确保执行了 `ollama serve`")
        return False


if not test_ollama_service():
    exit(1)


# ========== 2. 加载关键词（彻底修复：按逗号分割+净化） ==========
def load_keywords(file_path):
    try:
        with open(file_path, 'r', encoding='gbk') as f:
            content = f.read().strip()  # 读取所有内容
            raw_keywords = [kw.strip() for kw in content.split(',') if kw.strip()]  # 按逗号分割
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gb2312') as f:
            content = f.read().strip()
            raw_keywords = [kw.strip() for kw in content.split(',') if kw.strip()]

    # 净化关键词，过滤无效字符
    cleaned_keywords = []
    for kw in raw_keywords:
        clean_kw = clean_keyword(kw)
        if clean_kw:
            cleaned_keywords.append(clean_kw)

    return cleaned_keywords


KEYWORDS = load_keywords(KEYWORDS_FILE)
if len(KEYWORDS) == 0:
    raise ValueError("❌ 未加载到有效关键词！请检查关键词文件格式（逗号分隔的中文词语）")
print(f"✅ 加载关键词 {len(KEYWORDS)} 个：{KEYWORDS[:5]}...")  # 显示前5个，确认拆分成功


# ========== 3. 加载句子、创建上下文 & 断点续传逻辑 ==========
def prepare_data_for_scoring(source_excel_path, save_excel_path):
    # 原始数据加载和清洗
    sentence_df = pd.read_excel(source_excel_path)[[SENTENCE_ID_COL, SENTENCE_CONTENT_COL]].dropna()
    sentence_df[SENTENCE_ID_COL] = pd.to_numeric(sentence_df[SENTENCE_ID_COL], errors='coerce').astype(int)
    sentence_df = sentence_df[sentence_df[SENTENCE_CONTENT_COL].str.strip() != ''].reset_index(drop=True)

    # 创建上下文
    sentences = sentence_df[SENTENCE_CONTENT_COL].tolist()
    contextual_sentences = []
    n = len(sentences)

    for i in range(n):
        prev2 = sentences[i - 2] if i >= 2 else ""
        prev1 = sentences[i - 1] if i >= 1 else ""
        curr = sentences[i]
        next1 = sentences[i + 1] if i < n - 1 else ""
        next2 = sentences[i + 2] if i < n - 2 else ""
        context_str = (
            f"前两句: {prev2}\n"
            f"前一句: {prev1}\n"
            f"本句(待评分): {curr}\n"
            f"后一句: {next1}\n"
            f"后两句: {next2}"
        )
        contextual_sentences.append(context_str)

    sentence_df[CONTEXT_CONTENT_COL] = contextual_sentences

    # 断点续传逻辑
    if os.path.exists(save_excel_path):
        try:
            completed_df = pd.read_excel(save_excel_path)
            # 过滤已完成有效评分的句子
            completed_ids = completed_df[
                completed_df[SCORE_COL].notna() & (completed_df[SCORE_COL] > 0)
                ][SENTENCE_ID_COL].tolist()

            # 待处理的句子
            remaining_df = sentence_df[~sentence_df[SENTENCE_ID_COL].isin(completed_ids)].reset_index(drop=True)

            # 已完成的句子（合并原始列和评分列）
            finished_df = sentence_df[sentence_df[SENTENCE_ID_COL].isin(completed_ids)].merge(
                completed_df[[SENTENCE_ID_COL, SCORE_COL, THINKING_COL]],  # 只合并需要的列
                on=SENTENCE_ID_COL,
                how='left'
            )

            print(f"🔄 发现上次进度文件：已完成 {len(completed_ids)} 句评分。")
            print(f"🚀 将从第 {len(completed_ids) + 1} 句开始继续。")
            return finished_df, remaining_df

        except Exception as e:
            print(f"❌ 进度文件加载失败：{str(e)}，将从头开始处理。")
            return pd.DataFrame(), sentence_df
    else:
        print("🆕 未发现上次进度文件，将从头开始处理所有句子。")
        return pd.DataFrame(), sentence_df


# 调用数据准备函数
finished_df, remaining_df = prepare_data_for_scoring(SENTENCE_EXCEL, LLM_SCORE_SAVE_PATH)

# ========== 核心修复：初始化缺失列（解决KeyError） ==========
deepseek_score_df = pd.concat([finished_df, remaining_df], ignore_index=True)

# 检查并创建评分列和思考过程列（首次运行时缺失）
if SCORE_COL not in deepseek_score_df.columns:
    deepseek_score_df[SCORE_COL] = 0.0  # 初始化为0.0
if THINKING_COL not in deepseek_score_df.columns:
    deepseek_score_df[THINKING_COL] = ""  # 初始化为空字符串

# 填充NaN值（确保无缺失）
deepseek_score_df[SCORE_COL] = deepseek_score_df[SCORE_COL].fillna(0.0)
deepseek_score_df[THINKING_COL] = deepseek_score_df[THINKING_COL].fillna("")


# ========== 4. 评分函数 (适配词语类关键词) ==========
def get_deepseek_score(context_sentence, keywords):
    prompt = f"""
    任务：请根据“本句（待评分）及其前后上下文”与所提供的 {len(keywords)} 个独立政策关键词之间的关联程度，给出 0–10 分的语义关联度评分（允许小数，保留 1 位或 2 位小数）。

    -------------------------------
    【评分核心机制：关键词匹配数量 × 关联强度】
    请同时判断：
    - 句子是否直接包含关键词或其同义表达；
    - 该关键词是否是句子的核心语义；
    - 一个关键词可以是“弱相关 / 中度相关 / 强相关”，分数不同；
    - 分数可以是任意小数，如 8.8、9.1、9.3，不需要 0.5 递增。
    -------------------------------

    【评分标准（请严格执行）】

    （A）9.5 分及以上（极高相关，仅在极少数情况下使用）
    - 句子与 **5 个及以上关键词** 高度相关（直接或核心同义表达）
    - 且句子主旨基本由这些关键词构成（几乎完全吻合中央政策主题）
    - 禁止给出 >10 分。

    （B）8.5 – 9.4 分（强相关）
    - 出现 **2–4 个关键词** 的强相关（直接出现或核心动作/对象）
    - 或虽然没有多个关键词，但 **1 个关键词的关联度极高且是句子主题核心**
    - 分数请细化（如 9.1、9.3、9.6），不必以 0.5 为步长。

    （C）7.0 – 8.4 分（中度相关）
    - 出现 **1 个关键词的强相关**；或  
    - 出现 **2 个关键词的弱到中度相关**  
    - 如语义贴合但没有关键动作/主体，也应落在该区间。
    - 可给小数（如 7.4、8.2）。

    （D）5.0 – 6.9 分（弱相关）
    - 文本与 **至少 1 个关键词存在弱或间接关联**  
    （如“帮扶”→“补贴”，“农村地区”→“乡村”）
    - 也可用于出现政策外围相关但非核心语义的情况（如机制、制度、改革等泛政策词）。

    （E）0 – 4.9 分（无关或极弱相关）
    - 文本中未出现任何关键词或同义表达；
    - 语义与关键词集合无直接关系。

    -------------------------------

    【关键词说明】
    1. 所有关键词均作为独立单词单独判断，不合并主题；
    2. 允许识别政策常用场景中的同义/近义表达；
    3. 不得引入关键词列表之外的其他主题；
    4. 允许输出任意小数，如 8.7、9.1、9.45 等。

    -------------------------------

    输出要求：仅输出数字，不要解释。

    上下文信息（含目标句及前后关联句子）：
    {context_sentence}

    政策关键词列表：{', '.join(keywords)}
    """
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.0,  # 0温度确保评分一致
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        raw_response = result.get("response", "").strip()


        think_match = re.search(r'<think>(.*?)</think>', raw_response, flags=re.DOTALL)
        thinking_process = think_match.group(1).strip() if think_match else ""

        clean_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
        score_match = re.search(r'(\d+(\.\d+)?)', clean_response)

        if score_match:
            score = float(score_match.group(1))
            final_score = round(max(0.0, min(10.0, score)), 1)
            return final_score, thinking_process
        else:
            return 0.0, f"评分解析失败，模型原始输出：{raw_response}"
    except Exception as e:
        error_info = f"请求失败: {str(e)[:60]}"
        return 0.0, error_info


# ========== 5. 批量评分 (仅对 remaining_df 进行) ==========

total_remaining = len(remaining_df)
total_finished = len(finished_df)
print(
    f"\n🚀 开始评分，待处理 {total_remaining} 句，从 ID: {remaining_df[SENTENCE_ID_COL].iloc[0] if total_remaining > 0 else 'N/A'} 开始。")

for i, row in enumerate(remaining_df.itertuples()):
    # 使用 itertuples() 避免 loc/iloc 的性能损失，但需要获取原始索引用于更新 deepseek_score_df
    original_idx = row.Index
    context_input = getattr(row, CONTEXT_CONTENT_COL)

    current_global_index = total_finished + i + 1

    print(
        f"🔄 [{current_global_index}/{total_finished + total_remaining}] 正在评分 (ID: {getattr(row, SENTENCE_ID_COL)})...")

    score, thinking_process = get_deepseek_score(context_input, KEYWORDS)

    # 将分数和思考过程写入到 total_df 的对应位置 (通过原始索引)
    deepseek_score_df.loc[original_idx, 'deepseek_相关度评分'] = score
    deepseek_score_df.loc[original_idx, 'deepseek_思考过程'] = thinking_process

    print(f"   -> 评分：{score}")

    # 每5句保存一次 (使用全局索引 i+1)
    if (i + 1) % 5 == 0:
        save_cols = [SENTENCE_ID_COL, SENTENCE_CONTENT_COL, 'deepseek_相关度评分', 'deepseek_思考过程']
        # 注意：这里保存的是完整的 deepseek_score_df
        deepseek_score_df[save_cols].to_excel(LLM_SCORE_SAVE_PATH, index=False, engine='openpyxl')
        print(f"💾 [进度保存] 已保存进度至：{LLM_SCORE_SAVE_PATH}")

# ========== 6. 最终保存与统计 ==========
# 确保最后不足5句的数据也被保存
save_cols = [SENTENCE_ID_COL, SENTENCE_CONTENT_COL, 'deepseek_相关度评分', 'deepseek_思考过程']
deepseek_score_df[save_cols].to_excel(LLM_SCORE_SAVE_PATH, index=False, engine='openpyxl')
print(f"\n✅ 全文评分完成！最终结果已保存至：{LLM_SCORE_SAVE_PATH}")

# 简单统计
valid_scores = deepseek_score_df[deepseek_score_df['deepseek_相关度评分'] > 0]
print("\n📊 结果统计：")
print(f"总处理句数：{len(deepseek_score_df)}")
print(f"有关联句子数（>0分）：{len(valid_scores)}")
if len(deepseek_score_df) > 0:
    print(f"平均评分：{deepseek_score_df['deepseek_相关度评分'].mean():.2f} 分")