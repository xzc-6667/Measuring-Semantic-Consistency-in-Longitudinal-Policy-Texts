import os
import re
import json
import jieba
import networkx as nx
from collections import Counter, defaultdict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects
import math

# 使用 Agg 后端
matplotlib.use('Agg')

# ===================== 1. 基础配置 =====================
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 路径 (请确认你的路径)
base_path = r'C:\Users\user\Desktop\word\农业农村部一号文件'
folder_path = base_path
keyword_path = os.path.join(base_path, '关键词')
stopword_path = os.path.join(base_path, '停用词', 'chinese_stopwords.txt')
output_path = os.path.join(base_path, '图片', '语义网络图', 'english_versions')
os.makedirs(output_path, exist_ok=True)

# 新增：JSON 扩展语义簇保存路径
expanded_output_path = os.path.join(base_path, '扩展语义簇')
os.makedirs(expanded_output_path, exist_ok=True)

# 阶段划分
stages = {
    'Stage 1 (2004-2007)': list(range(2004, 2008)),
    'Stage 2 (2008-2012)': list(range(2008, 2013)),
    'Stage 3 (2013-2017)': list(range(2013, 2018)),
    'Stage 4 (2018-2022)': list(range(2018, 2023)),
    'Stage 5 (2023-2026)': list(range(2023, 2027))
}

# ===================== 补充与扩容的英文映射 =====================
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

# 全局变量用于收集缺失的翻译词汇
missing_translations = set()


# ===================== 2. 数据处理 =====================
def preprocess_text(text):
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
    words = jieba.lcut(text)
    return [w for w in words if len(w) > 1]


def load_data():
    global stopwords
    stopwords = set()
    if os.path.exists(stopword_path):
        with open(stopword_path, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f)

    # 基础通用词
    stopwords.update(['的', '和', '在', '要', '等', '了', '是', '对', '也', '有', '不', '为', '上'])

    # ================= 核心修改：新增领域专用停用词 =================
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
    # ================================================================

    all_text = ""
    pattern = re.compile(r'(\d{4})中央一号文件\.txt')

    if not os.path.exists(folder_path):
        print(f"错误: 找不到文件夹 {folder_path}")
        return {}, "", [], Counter()

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
    return stage_keywords, all_text, global_vocab, Counter(all_words)


# ===================== 3. 数据构建 =====================

def build_concentric_data(stage_kws, global_vocab, text, word_freq, top_k=8):
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

    core_list = []
    rel_dict = defaultdict(list)

    for kw in stage_kws:
        core_list.append(kw)
        related = []
        for (u, v), count in cooc.items():
            if kw in (u, v):
                other = v if u == kw else u
                related.append((other, count))
        related.sort(key=lambda x: x[1], reverse=True)
        filtered_related = [x for x in related if x[0] not in stage_kws]
        rel_dict[kw] = filtered_related[:top_k]

    return core_list, rel_dict


# ===================== 4. 绘图：完美版布局（支持中文修复版） =====================

def check_collision(x, y, existing_positions, min_dist=0.25):
    for ex, ey in existing_positions:
        dist = np.sqrt((x - ex) ** 2 + (y - ey) ** 2)
        if dist < min_dist:
            return True
    return False


def draw_final_polished(ax, core_list, rel_dict, word_freq, title):
    if not core_list:
        ax.axis('off');
        return

    num_cores = len(core_list)
    radius_inner_1 = 0.70
    radius_inner_2 = 1.15
    core_positions = {}

    for i, core in enumerate(core_list):
        angle = 2 * np.pi * i / num_cores
        r = radius_inner_1 if i % 2 == 0 else radius_inner_2
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        core_positions[core] = (x, y)

    min_dist_base = 1.20
    max_dist_base = 2.40
    rel_positions_dict = {}
    edges = []
    existing_positions = []

    for i, core in enumerate(core_list):
        rels = rel_dict.get(core, [])
        if not rels: continue

        weights = [w for _, w in rels]
        max_w = max(weights) if weights else 1
        min_w = min(weights) if weights else 0

        start_angle = 2 * np.pi * (i - 0.4) / num_cores
        end_angle = 2 * np.pi * (i + 0.4) / num_cores

        for j, (r_node, weight) in enumerate(rels):
            if len(rels) == 1:
                sub_angle = 2 * np.pi * i / num_cores
            else:
                sub_angle = start_angle + (end_angle - start_angle) * (j / (len(rels) - 1))

            norm_w = 0.5
            if max_w != min_w:
                norm_w = (weight - min_w) / (max_w - min_w)

            target_radius = max_dist_base - (norm_w * (max_dist_base - min_dist_base))
            final_x, final_y = 0, 0
            placed = False

            for push in np.linspace(0, 0.8, 20):
                test_r = target_radius + push
                test_x = test_r * np.cos(sub_angle)
                test_y = test_r * np.sin(sub_angle)
                if not check_collision(test_x, test_y, existing_positions, min_dist=0.25):
                    final_x, final_y = test_x, test_y
                    placed = True
                    break

            if not placed:
                final_x = (target_radius + 0.3) * np.cos(sub_angle + 0.1)
                final_y = (target_radius + 0.3) * np.sin(sub_angle + 0.1)

            existing_positions.append((final_x, final_y))
            rel_positions_dict[(core, r_node)] = (final_x, final_y)
            edges.append((core_positions[core], (final_x, final_y), weight))

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)

    max_edge_w = max([e[2] for e in edges]) if edges else 1
    for start, end, w in edges:
        lw = 0.5 + (w / max_edge_w) * 2.5
        ax.plot([start[0], end[0]], [start[1], end[1]], color='gray', alpha=0.3, linewidth=lw, zorder=1)

    for (core_node, r_node), (Rx, Ry) in rel_positions_dict.items():
        eng_name = english_mapping.get(r_node, r_node)
        if eng_name == r_node:
            curr_font = 'SimHei'
            missing_translations.add(r_node)
        else:
            curr_font = 'Arial'

        ax.scatter(Rx, Ry, s=150, color='#4D96FF', edgecolors='white', linewidth=1.5, zorder=2)

        angle_deg = np.degrees(np.arctan2(Ry, Rx))
        if -90 <= angle_deg <= 90:
            rotation = angle_deg;
            ha = 'left'
            tx = Rx + 0.08 * np.cos(np.radians(angle_deg))
            ty = Ry + 0.08 * np.sin(np.radians(angle_deg))
        else:
            rotation = angle_deg + 180;
            ha = 'right'
            tx = Rx + 0.08 * np.cos(np.radians(angle_deg))
            ty = Ry + 0.08 * np.sin(np.radians(angle_deg))

        t = ax.text(tx, ty, eng_name, fontsize=13, fontfamily=curr_font,
                    ha=ha, va='center', rotation=rotation, rotation_mode='anchor', zorder=3)
        t.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground='white')])

    for c_node, (Cx, Cy) in core_positions.items():
        eng_name = english_mapping.get(c_node, c_node)
        if eng_name == c_node:
            curr_font = 'SimHei'
            missing_translations.add(c_node)
        else:
            curr_font = 'Arial'

        ax.scatter(Cx, Cy, s=700, color='#FF6B6B', edgecolors='white', linewidth=2, zorder=4)
        t = ax.text(Cx, Cy, eng_name, fontsize=15, fontweight='bold', fontfamily=curr_font,
                    ha='center', va='center', color='white', zorder=5)
        t.set_path_effects([path_effects.withStroke(linewidth=2, foreground='#333333')])

    ax.set_title(title, fontsize=26, fontweight='bold', y=1.12, pad=30)


# ===================== 5. 主程序 =====================

def main():
    print("Loading data...")
    stage_keywords, all_text, global_vocab, word_freq = load_data()

    fig, axes = plt.subplots(2, 3, figsize=(26, 24), dpi=300)
    axes = axes.flatten()
    sorted_stages = sorted(stage_keywords.keys())

    print("Generating Final Polished Layouts & Saving Semantic Clusters...")

    for i, stage in enumerate(sorted_stages):
        if i >= 5: break
        print(f"Processing {stage}...")
        keywords = stage_keywords[stage]
        if not keywords: continue

        # 1. 构建网络图数据
        core_list, rel_dict = build_concentric_data(keywords, global_vocab, all_text, word_freq, top_k=7)

        # 2. 画图
        draw_final_polished(axes[i], core_list, rel_dict, word_freq, stage)

        # 3. ================= 提取并保存带权重的中文语义簇 (JSON格式) =================
        stage_cluster_data = {}
        for core in core_list:
            rels = rel_dict.get(core, [])
            related_data = {}
            if rels:
                # 获取最大关联共现频次，用于归一化
                max_w = max([w for _, w in rels]) if rels else 1
                for rel_word, weight in rels:
                    norm_w = 0.1 + 0.7 * (weight / max_w) if max_w > 0 else 0.5
                    related_data[rel_word] = round(norm_w, 4)

            stage_cluster_data[core] = {
                "core_weight": 1.0,
                "related": related_data
            }

        safe_stage_name = stage.replace(" (", "_").replace(")", "")
        save_json_file = os.path.join(expanded_output_path, f"{safe_stage_name}_clusters.json")

        with open(save_json_file, 'w', encoding='utf-8') as f:
            json.dump(stage_cluster_data, f, ensure_ascii=False, indent=4)
        print(f"   💾 JSON 扩展语义簇已保存至: {save_json_file}")

    # 图例设置
    ax_legend = axes[5]
    ax_legend.axis('off')
    legend_elements = [
        Patch(facecolor='#FF6B6B', edgecolor='white', label='Core Keywords'),
        Patch(facecolor='#4D96FF', edgecolor='white', label='Related Nodes'),
        plt.Line2D([0], [0], color='gray', lw=2, alpha=0.5, label='Semantic Link')
    ]
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=26, frameon=False, title="Legend",
                     title_fontsize=30)

    # 全局排版
    plt.suptitle("Semantic Structure Evolution: Weighted Distance (2004-2026)", fontsize=36, fontweight='bold', y=0.98)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.05, wspace=0.1, hspace=0.1)

    save_file = os.path.join(output_path, "Final_Ultimate_Layout_FixedChinese_2026.jpg")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved final layout to: {save_file}")

    if missing_translations:
        print("\n⚠️ 发现未翻译的中文词汇，请补充到 english_mapping 中：")
        print("、".join(missing_translations))


if __name__ == "__main__":
    main()