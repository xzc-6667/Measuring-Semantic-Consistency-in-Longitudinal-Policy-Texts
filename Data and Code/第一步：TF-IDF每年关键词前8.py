import os
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba


class PolicyKeywordExtractor:
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.stopwords = self.load_stopwords()

    def load_stopwords(self):
        stopwords = set()
        # 加载哈工大停用词
        with open("../6.24/chinese_stopwords.txt", "r", encoding="utf-8") as f:
            for line in f:
                stopwords.add(line.strip())

        # 添加领域内高频但无分析价值的词
        domain_stopwords = {
            # 常规动作词
            '继续', '推进', '加强', '加大', '开展', '完善', '建立', '健全', '实施', '发展',
            '有效', '提高', '加快', '落实', '深化', '规范', '扩大', '积极', '稳步',
            '全面', '发展', '牵头', '提高', '做好',
            "支持", "增加", "加强", "进一步", "推动", "推进", "提升", "促进",
            "保障", "开展", "实施", "深化", "搞好", "持续", "深入", "强化",
            "必须", "加快", "优化", "着力", "确保", "推动落实", "不断", "建设",

            # 政策术语与结构词
            '工程', '项目', '行动', '工作', '服务', '机制', '体系', '制度', '改革',
            '政策', '措施', '管理', '重点', '问题', '方面', '任务', '目标', '内容',
            '成效', '基础', '能力', '水平', '路径', '作用', '要求', '建设', '治理',
            # 组织与政府单位
            '国家', '社会', '政府', '地方', '各级', '相关', '组织', '部门', '机构', '单位',
            # 文本分析中无意义词
            '文件', '一号', '中共中央', '党中央', '分析', '句子', '清洗', '原始', '综合',
            '数总', '后前', '两句', '投入', '坚持', '有关', '保护', '力度', '加工', '发挥', '小型'
            , '切实', '推广', '条件', '稳定', '鼓励', '一批',
            # # 高频农业通用词（如需过滤主题背景）
            '农业', '农村', '农民', '社会主义'

        }
        stopwords.update(domain_stopwords)
        return stopwords

    def clean_text(self, text):
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)  # 仅保留中文
        return text

    def preprocess(self, text):
        text = self.clean_text(text)
        words = jieba.cut(text)
        filtered = [w for w in words if w and w not in self.stopwords and len(w) > 1]
        return ' '.join(filtered)

    def extract_top_keywords_tfidf(self, processed_sentences, topk=8):
        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x.split(),
            stop_words=list(self.stopwords),
            max_features=1000
        )
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        feature_array = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().sum(axis=0)

        top_indices = scores.argsort()[::-1][:topk]
        return [feature_array[i] for i in top_indices]

    def extract_keywords_by_year(self):
        year_keywords = defaultdict(list)
        for filename in sorted(os.listdir(self.file_dir)):
            if filename.endswith(".txt"):
                year = filename[:4]
                with open(os.path.join(self.file_dir, filename), "r", encoding="utf-8") as f:
                    text = f.read()
                sentences = re.split(r'[。！？]', text)
                processed_sentences = [self.preprocess(s) for s in sentences if len(s) > 5]
                keywords = self.extract_top_keywords_tfidf(processed_sentences)
                year_keywords[year] = keywords

                # 输出关键词到控制台
                print(f"{year} 年关键词（TF-IDF 提取）: {', '.join(keywords)}")

                # 保存到 txt 文件（可选）
                # out_path = f"C:/Users/user/Desktop/word/农业农村部一号文件/关键词/{year}_keywords.txt"
                # with open(out_path, "w", encoding="utf-8") as out_f:
                #     out_f.write("\n".join(keywords))

        return year_keywords


# 使用示例
extractor = PolicyKeywordExtractor("C:/Users/user/Desktop/word/农业农村部一号文件")
year_keywords = extractor.extract_keywords_by_year()


# 阶段划分函数
def group_years_by_stage(year_keywords):
    stage_keywords = {
        "阶段一（2004–2007）": set(),
        "阶段二（2008–2012）": set(),
        "阶段三（2013–2017）": set(),
        "阶段四（2018–2022）": set(),
        "阶段五（2023–2026）": set()
    }

    for year, keywords in year_keywords.items():
        y = int(year)
        if 2004 <= y <= 2007:
            stage_keywords["阶段一（2004–2007）"].update(keywords)
        elif 2008 <= y <= 2012:
            stage_keywords["阶段二（2008–2012）"].update(keywords)
        elif 2013 <= y <= 2017:
            stage_keywords["阶段三（2013–2017）"].update(keywords)
        elif 2018 <= y <= 2022:
            stage_keywords["阶段四（2018–2022）"].update(keywords)
        elif 2023 <= y <= 2026:
            stage_keywords["阶段五（2023–2026）"].update(keywords)

    return stage_keywords


# 输出并保存每个阶段关键词
output_dir = "C:/Users/user/Desktop/word/农业农村部一号文件/关键词修改"
os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

stage_keywords = group_years_by_stage(year_keywords)

for stage, keywords in stage_keywords.items():
    sorted_keywords = sorted(keywords)
    print(f"\n{stage} 的关键词集合（共 {len(keywords)} 个）:")
    print(", ".join(sorted_keywords))

    # 保存为 txt 文件
    stage_filename = stage.replace("（", "_").replace("）", "").replace("–", "-") + "_keywords.txt"
    output_path = os.path.join(output_dir, stage_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted_keywords))
