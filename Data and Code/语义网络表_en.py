import os
import re
import jieba
import networkx as nx
import pandas as pd
from collections import Counter, defaultdict
from networkx.algorithms.community import greedy_modularity_communities, modularity

# ===================== 1. 基础配置 =====================
base_path = r'C:\Users\user\Desktop\word\农业农村部一号文件'
folder_path = base_path
keyword_path = os.path.join(base_path, '关键词')
stopword_path = os.path.join(base_path, '停用词', 'chinese_stopwords.txt')

stages = {
    'Stage 1 (2004-2007)': list(range(2004, 2008)),
    'Stage 2 (2008-2012)': list(range(2008, 2013)),
    'Stage 3 (2013-2017)': list(range(2013, 2018)),
    'Stage 4 (2018-2022)': list(range(2018, 2023)),
    'Stage 5 (2023-2026)': list(range(2023, 2027))
}

# 完整英文映射词典（与画图脚本保持绝对一致）
english_mapping = {
    "三农": "San Nong", "乡村": "Rural Areas", "振兴": "Revitalization", "创新": "Innovation",
    "粮食": "Grain", "耕地": "Arable Land", "生产": "Production", "农产品": "Agri-Products",
    "试点": "Pilot", "技术": "Technology", "企业": "Enterprise", "基础设施": "Infrastructure",
    "工程": "Project", "补贴": "Subsidy", "资金": "Funds", "经济": "Economy",
    "科技": "Sci-Tech", "现代农业": "Modern Agri", "主产区": "Major Areas",
    "脱贫": "Poverty Allev", "质量": "Quality", "用地": "Land Use", "品牌": "Brand",
    "帮扶": "Assistance", "资产": "Assets", "整治": "Remediation", "农民": "Farmers",
    "增收": "Income Incr", "市场": "Market", "文化": "Culture", "培训": "Training",
    "建设": "Constr", "设施": "Facility", "治理": "Gov", "扩大": "Expand",
    "保险": "Insurance", "标准": "Standard", "规范": "Reg", "适度": "Moderate",
    "科学": "Science", "机收": "Mech Harvest", "加工": "Processing", "减损": "Loss Reduct",
    "贴息": "Interest Disc", "奖补": "Award", "组建": "Establish", "夯实": "Consolidate",
    "水平": "Level", "鼓励": "Encourage", "学校": "School", "投资": "Investment",
    "联合体": "Consortium", "职业": "Vocational", "人口": "Pop", "新增": "New Added",
    "转移": "Transfer", "指标": "Indicator", "支付": "Payment", "继续": "Continue",
    "行动": "Action", "促进": "Promote", "就近": "Nearby", "富农": "Enrich Farmers",
    "农惠农": "Agri-benefit", "县级": "County Lvl", "稻谷": "Rice", "承担": "Undertake",
    "保费": "Premium", "比例": "Ratio", "运用": "Utilize", "经验": "Exp",
    "学习": "Learn", "制度": "System", "健全": "Improve", "产粮": "Grain Prod",
    "活动": "Activity", "进口": "Import", "壮大": "Expand", "引导": "Guide",
    "扶持": "Support", "群众性": "Mass", "健康": "Healthy", "乡土": "Rural",
    "惠民": "Benefit Ppl", "高素质": "High Qual", "打造": "Build", "队伍": "Team",
    "适应": "Adapt", "数字": "Digital", "县域": "County", "富民": "Enrich Ppl",
    "推进": "Promote", "推动": "Promote", "问题": "Issue", "人才": "Talent",
    "条件": "Condition", "提供": "Provide", "城市": "City", "逐步": "Step by Step",
    "基本": "Basic", "公共服务": "Public Svc", "改革": "Reform", "探索": "Explore",
    "实践": "Practice", "登记": "Register", "制定": "Formulate", "办法": "Measure",
    "坚持不懈": "Persevere", "关键": "Key", "重中之重": "Top Priority", "作为": "As",
    "工作": "Work", "全党": "Whole Party", "必须": "Must", "解决": "Solve",
    "紧密团结": "Unite", "同志": "Comrade", "合理": "Reasonable", "干部": "Cadre",
    "动力": "Motion", "积极性": "Enthusiasm", "激发": "Stimulate", "群众": "Masses",
    "项目": "Project", "编制": "Plan", "全面": "Comp", "攻坚": "Tackle",
    "投入": "Input", "管理机制": "Mgmt Mech", "形成": "Form", "长效": "Long-term",
    "全过程": "Whole Proc", "监管": "Supervision", "着力": "Focus", "国家": "Country",
    "多渠道": "Multi-chnl", "易地": "Relocate", "倾斜": "Tilt", "搬迁": "Move",
    "安置": "Resettle", "深化": "Deepen", "持续": "Sustain", "参与": "Participate",
    "村级": "Village Lvl", "村庄": "Village", "扶贫": "Poverty Allev", "繁荣": "Prosperity",
    "生活": "Life", "基础": "Foundation", "支持": "Support", "规划": "Planning",
    "地区": "Region", "更加": "More", "不得": "No", "实施": "Implement",
    "机制": "Mechanism", "协调": "Coordinate", "完善": "Perfect", "综合": "Comprehensive",
    "贸易": "Trade", "施策": "Policy", "各类": "Various", "严厉打击": "Crack Down",
    "体系": "System", "非法": "Illegal", "范围": "Scope", "矛盾": "Conflict",
    "化解": "Resolve", "延长": "Extend", "整省": "Province", "需求": "Demand",
    "调整": "Adjust", "优化": "Optimize", "培育": "Cultivate", "导向": "Guide",
    "涉农": "Agri-related", "管理": "Manage", "特色": "Feature",
    "农垦": "Farms", "体制": "System", "公司": "Company", "推广": "Promote",
    "节水": "Water Save", "灌溉": "Irrigation", "台账": "Ledger", "医生": "Doctor",
    "稳定": "Stable", "计划": "Plan", "专项": "Special", "融合": "Integrate",
    "城乡": "Urban-Rural", "加强": "Strengthen", "用水": "Water Use", "强化": "Reinforce",
    "发展": "Develop", "农村": "Rural", "海洋": "Marine", "建设工程": "Constr Proj",
    "防草料": "Forage", "河道": "River", "加大": "Increase", "基地": "Base",
    "水资源": "Water Res", "强度": "Intensity", "红线": "Red Line", "总量": "Total",
    "实行": "Execute", "约束": "Constrain", "安全": "Safety", "领域": "Field",
    "风险": "Risk", "排查": "Inspect", "源头": "Source", "消防": "Fire",
    "管控": "Control", "紧密": "Close", "联合": "Joint", "家庭": "Family",
    "农场": "Farm", "建立": "Establish", "承包": "Contract", "有偿": "Paid",
    "土地": "Land", "使用权": "Use Right", "自愿": "Voluntary", "收益分配": "Inc Dist",
    "集体": "Collective", "各地": "Regions", "到户": "To House", "验收": "Accept",
    "政府": "Govt", "活力": "Vitality", "更加积极性": "More Enthus", "协同": "Collab",
    "公益性": "Pub Welfare", "补助": "Subsidy", "生态": "Ecology", "产业": "Industry",
    "劳务": "Labor", "兴农": "Promote Agri", "新型": "New Type", "经营": "Operation",
    "社会化": "Social", "主体": "Entity", "提高": "Improve", "协作": "Coop",
    "推介": "Promote", "实效": "Effect", "一体化": "Integrate", "要求": "Req",
    "农业": "Agri", "加快": "Speed Up", "出口": "Export", "优势": "Advantage",
    "深入展开": "Deepen", "综合治理": "Comp Gov", "深入": "Deep",
    "倡导": "Advocate", "实用": "Practical", "技能": "Skill", "服务": "Service",
    "就业": "Job", "支撑": "Support", "惠农": "Benefit", "扎实": "Solid",
    "农田水利": "Irrigation", "书记": "Secretary", "落实": "Implement",
    "增加": "Add", "有效": "Effective", "支农": "Support Agri", "超采": "Over Mine",
    "财政": "Finance", "财政收入": "Revenue", "地下水": "Groundwater", "重大": "Major",
    "一批": "Batch", "开展": "Launch", "启动": "Start", "抓紧": "Hurry",
    "配套": "Match", "农机": "Machine", "基层": "Grassroot",
    "绿色通道": "Green Chnl", "防治": "Prevent", "水利建设": "Water Constr",
    "饲草料": "Forage", "黑土": "Black Soil", "具备": "Have", "建成": "Built",
    "合作社": "Co-op", "农户": "Farmer", "宅基地": "Homestead", "一体": "One Body",
    "能力": "Ability", "做好": "Do Well", "管好": "Manage Well",
    "资源": "Resource", "执法": "Enforce", "进村": "To Village", "入户": "To Home",
    "检疫": "Quarantine", "重点": "Focus", "模式": "Mode", "金融": "Finance",
    "改厕": "Toilet", "畜禽": "Livestock", "污染": "Pollution",
    "处理": "Treat", "水产": "Aquatic", "资源化": "Res Utilization",
    "粪污": "Waste", "养殖": "Breeding", "深入开展": "Deepen",
    "水利": "Water", "水利工程": "Water Proj", "财政投入": "Fiscal Input",
    "法律": "Law", "利用": "Use", "农技": "Agrotechnology", "激励": "Incentive",
    "贷款": "Loan",
    "利益": "Interest", "补偿": "Compensate", "政策": "Policy", "补充": "Supplement",
    "财力": "Finances", "投放": "Allocate", "依法": "Lawfully", "走私": "Smuggle",
    "提升": "Enhance", "各级": "All Lvls", "带动": "Drive", "致富": "Prosper",
    "力度": "Intensity", "医疗卫生": "Healthcare", "医院": "Hospital", "收益": "Profit",
    "保障机制": "Safeguard", "监测": "Monitor", "预警": "Early Warning", "信息": "Info",
    "文明": "Civilization", "规模经营": "Scale Op", "有机": "Organic", "充分发挥": "Fully Utilize",
    "技术改造": "Tech Upgrade", "区域": "Region", "布局": "Layout",
    "中小企业": "SMEs", "特色产业": "Char Ind", "购置": "Purchase", "应用": "Apply",
    "防范": "Prevent", "本领": "Skill", "科研院所": "Research Inst", "坚持": "Persist",
    "并举": "Co-develop", "挪用": "Embezzle", "要素": "Factor",
    "双向": "Two-way", "流动": "Flow", "渔船": "Fishing Boat", "共享": "Share",
    "绿色": "Green", "成果": "Result", "草原": "Grassland", "保护": "Protect",
    "奖励": "Reward", "到期": "Expire", "妥善解决": "Resolve", "延包": "Extend Contract",
    "公共": "Public", "预算": "Budget", "效能": "Efficiency", "锚定": "Anchor",
    "目标": "Target", "取得": "Achieve", "创业": "Startup", "带头人": "Leader",
    "地方": "Local", "效率": "Efficiency", "期货市场": "Futures Mkt", "债务": "Debt",
    "供销": "Supply", "精深": "Intensive", "精品": "Premium", "产业链": "Ind Chain",
    "建立健全": "Establish", "发挥": "Exert", "现代": "Modern",
    "有序": "Orderly", "盘活": "Revitalize", "维护": "Maintain", "进城": "Urbanize",
    "生态环境": "Eco-env", "彩礼": "Bride Price", "省际": "Inter-prov", "联动": "Synergy",
    "树立": "Establish", "农民工": "Migrant Worker", "关爱": "Care", "工资": "Wage",
    "现代化": "Modernization", "面源": "Non-point", "作风": "Conduct", "培养": "Cultivate",
    "储备": "Reserve", "议事": "Discuss", "协商": "Consult", "重金属": "Heavy Metal",
    "动态": "Dynamic", "评估": "Evaluate", "村干部": "Village Cadre", "报酬": "Pay",
    "保障": "Guarantee", "因地制宜": "Suit Local", "新一轮": "New Round", "高校": "University",
    "毕业生": "Graduate", "更好": "Better", "党委": "Party Comm", "作用": "Role",
    "资源管理": "Res Mgmt", "突出": "Highlight", "统筹": "Coordinate",
    "农村基层": "Rural Grassroots", "服务中心": "Service Ctr", "监督管理": "Supervision",
    "农技推广": "AgTech Promo", "承包地": "Contracted Land", "党组织": "Party Org",
    "村党组织": "Village Party", "村庄规划": "Village Plan", "防返贫": "Anti-Relapse",
    "农业科技": "Agri-Tech", "农机装备": "Agri-Equip", "高质量": "High Quality",
    "现代化建设": "Modern Constr", "执法监管": "Law Enforcement", "经营主体": "Biz Entity",
    "社会化服务": "Social Service", "新型农村": "New Rural", "人居环境": "Living Env",
    "统筹协调": "Coordination", "防灾减灾": "Disaster Relief", "科技创新": "Tech Innov",
    "12": "12", "24": "24", "26": "26", "十九": "19", "十二": "12", "二十四": "24", "二十六": "26",
    "正确": "Correct", "结合实际": "Practical", "发布": "Release",
    "衔接": "Connect", "拓宽": "Broaden", "秸秆": "Straw",
    "统筹规划": "Overall Plan", "开发": "Develop", "效益": "Benefit",
    "中心": "Center", "状况": "Status", "落户": "Settle",
    "集群": "Cluster", "法治": "Rule of Law", "程序": "Procedure",
    "积极参与": "Participate Actively"
}


# ===================== 2. 数据处理 =====================
def load_data():
    stopwords = set()
    if os.path.exists(stopword_path):
        with open(stopword_path, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f)
    stopwords.update(['的', '和', '在', '要', '等', '了', '是', '对', '也', '有', '不', '为', '上'])

    domain_stopwords = {
        '继续', '推进', '加强', '加大', '开展', '完善', '建立', '健全', '实施', '发展',
        '有效', '提高', '加快', '落实', '深化', '规范', '扩大', '积极', '稳步',
        '全面', '牵头', '支持', '增加', '进一步', '推动', '提升', '促进', '做好',
        '保障', '搞好', '持续', '深入', '强化', '必须', '优化', '着力', '确保',
        '推动落实', '不断', '建设', '工程', '项目', '行动', '工作', '服务',
        '机制', '体系', '制度', '改革', '政策', '措施', '管理', '重点', '问题',
        '方面', '任务', '目标', '内容', '成效', '基础', '能力', '水平', '路径',
        '作用', '要求', '治理', '国家', '社会', '政府', '地方', '各级', '相关',
        '组织', '部门', '机构', '单位', '文件', '一号', '中共中央', '党中央',
        '分析', '句子', '清洗', '原始', '综合', '数总', '后前', '两句', '投入',
        '坚持', '有关', '保护', '力度', '加工', '发挥', '小型', '切实', '推广',
        '条件', '稳定', '鼓励', '一批', '农业', '农村', '农民', '社会主义'
    }
    stopwords.update(domain_stopwords)

    all_text = ""
    pattern = re.compile(r'(\d{4})中央一号文件\.txt')
    if not os.path.exists(folder_path): return {}, "", [], Counter()

    for f_name in os.listdir(folder_path):
        if pattern.match(f_name):
            with open(os.path.join(folder_path, f_name), 'r', encoding='utf-8') as f:
                all_text += f.read()

    all_words = [w for w in jieba.lcut(all_text) if w not in stopwords and len(w) > 1]
    global_vocab = [w for w, c in Counter(all_words).items() if c > 5]

    stage_keywords = {}
    if os.path.exists(keyword_path):
        for fname in os.listdir(keyword_path):
            if fname.endswith('_keywords.txt'):
                name = fname.replace('_keywords.txt', '')
                if '2004' in name:
                    k = 'Stage 1 (2004-2007)'
                elif '2008' in name:
                    k = 'Stage 2 (2008-2012)'
                elif '2013' in name:
                    k = 'Stage 3 (2013-2017)'
                elif '2018' in name:
                    k = 'Stage 4 (2018-2022)'
                elif '2023' in name:
                    k = 'Stage 5 (2023-2026)'
                else:
                    continue
                with open(os.path.join(keyword_path, fname), 'r', encoding='utf-8') as f:
                    stage_keywords[k] = [l.strip() for l in f if l.strip()]
    return stage_keywords, all_text, global_vocab


def build_concentric_data(stage_kws, global_vocab, text, top_k=7):
    cooc = defaultdict(int)
    words = [w for w in jieba.lcut(text) if len(w) > 1]
    vocab_set = set(global_vocab)
    w_idx = {w: i for i, w in enumerate(words)}

    for kw in stage_kws:
        if kw not in w_idx: continue
        idx = w_idx[kw]
        start, end = max(0, idx - 5), min(len(words), idx + 6)
        for cw in words[start:end]:
            if cw != kw and cw in vocab_set:
                pair = tuple(sorted((kw, cw)))
                cooc[pair] += 1

    rel_dict = defaultdict(list)
    for kw in stage_kws:
        related = []
        for (u, v), count in cooc.items():
            if kw in (u, v):
                other = v if u == kw else u
                related.append((other, count))
        related.sort(key=lambda x: x[1], reverse=True)
        rel_dict[kw] = [x for x in related if x[0] not in stage_kws][:top_k]
    return rel_dict


# ===================== 3. 计算网络拓扑参数 (全英文) =====================
def calculate_network_metrics(stage_kws, rel_dict):
    G = nx.Graph()
    for core, rels in rel_dict.items():
        for rel, weight in rels:
            G.add_edge(core, rel, weight=weight)
    if len(G.nodes) == 0: return None

    # 计算参数
    nodes_count = G.number_of_nodes()
    edges_count = G.number_of_edges()
    density = nx.density(G)

    deg_cent = nx.degree_centrality(G)
    top_cores = sorted(deg_cent, key=deg_cent.get, reverse=True)
    top_cores_chinese = [w for w in top_cores if w in stage_kws][:5]
    if not top_cores_chinese: top_cores_chinese = top_cores[:5]

    # 转换为英文并拼接
    top_cores_english = [english_mapping.get(w, w) for w in top_cores_chinese]
    top_cores_str = ", ".join(top_cores_english)

    if nx.is_connected(G):
        avg_path = nx.average_shortest_path_length(G)
    else:
        subgraph = G.subgraph(max(nx.connected_components(G), key=len))
        avg_path = nx.average_shortest_path_length(subgraph) if len(subgraph) > 1 else 0

    try:
        communities = greedy_modularity_communities(G, weight='weight')
        mod_score = modularity(G, communities, weight='weight')
    except:
        mod_score = 0.0

    return {
        "Stage": "",
        "Nodes": nodes_count,
        "Edges": edges_count,
        "Density": round(density, 4),
        "Modularity": round(mod_score, 4),
        "Avg. Path Length": round(avg_path, 4),
        "Core Hub Nodes (Degree Top 5)": top_cores_str
    }


# ===================== 4. 主程序 =====================
def main():
    stage_keywords, all_text, global_vocab = load_data()
    if not stage_keywords: return

    results = []
    for stage in sorted(stage_keywords.keys()):
        keywords = stage_keywords[stage]
        if not keywords: continue
        rel_dict = build_concentric_data(keywords, global_vocab, all_text, top_k=7)
        metrics = calculate_network_metrics(keywords, rel_dict)
        if metrics:
            metrics["Stage"] = stage
            results.append(metrics)

    df = pd.DataFrame(results)
    cols = ["Stage", "Nodes", "Edges", "Density", "Modularity", "Avg. Path Length", "Core Hub Nodes (Degree Top 5)"]
    df = df[cols]

    print("\n" + "=" * 80)
    print(df.to_markdown(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()