import re


SCHOOL_ZH_MAP = {
    # UK
    "University of Cambridge": "剑桥大学",
    "University of Oxford": "牛津大学",
    "Imperial College London": "帝国理工学院",
    "LSE": "伦敦政治经济学院",
    "UCL": "伦敦大学学院",
    "University College London": "伦敦大学学院",
    "Durham University": "杜伦大学",
    "University of Warwick": "华威大学",
    "University of Edinburgh": "爱丁堡大学",
    "University of Bristol": "布里斯托大学",
    "University of Manchester": "曼彻斯特大学",
    "University of Sheffield": "谢菲尔德大学",
    "University of Leeds": "利兹大学",
    "Loughborough University": "拉夫堡大学",
    "University of Glasgow": "格拉斯哥大学",
    "University of Leicester": "莱斯特大学",
    "University of Surrey": "萨里大学",
    "Nottingham Trent": "诺丁汉特伦特大学",
    "Nottingham Trent University": "诺丁汉特伦特大学",

    # US
    "MIT": "麻省理工学院",
    "Massachusetts Institute of Technology": "麻省理工学院",
    "Harvard University": "哈佛大学",
    "Columbia University": "哥伦比亚大学",
    "Johns Hopkins": "约翰霍普金斯大学",
    "Johns Hopkins University": "约翰霍普金斯大学",
    "USC": "南加州大学",
    "University of Southern California": "南加州大学",
    "University of Michigan": "密歇根大学",
    "NYU Stern": "纽约大学斯特恩商学院",
    "New York University Stern": "纽约大学斯特恩商学院",
    "UIUC": "伊利诺伊大学厄巴纳-香槟分校",
    "University of Illinois Urbana-Champaign": "伊利诺伊大学厄巴纳-香槟分校",
    "Boston University": "波士顿大学",
    "University of Pittsburgh": "匹兹堡大学",
    "UC Davis": "加州大学戴维斯分校",
    "University of California, Davis": "加州大学戴维斯分校",
    "Penn State": "宾夕法尼亚州立大学",
    "Pennsylvania State University": "宾夕法尼亚州立大学",
    "Rutgers University": "罗格斯大学",

    # Canada
    "University of Toronto": "多伦多大学",
    "McGill University": "麦吉尔大学",
    "UBC": "不列颠哥伦比亚大学",
    "University of British Columbia": "不列颠哥伦比亚大学",
    "University of Waterloo": "滑铁卢大学",
    "McMaster University": "麦克马斯特大学",

    # Australia
    "University of Melbourne": "墨尔本大学",
    "University of Sydney": "悉尼大学",
    "Australian National University": "澳大利亚国立大学",
    "UNSW Sydney": "新南威尔士大学悉尼",
    "University of New South Wales": "新南威尔士大学",
    "University of Queensland": "昆士兰大学",
    "Monash University": "莫纳什大学",
}

TAG_ZH_MAP = {
    "favourable for hukou/talent-visa return": ["有利于回国落户", "人才签证", "留学生落户"],
    "startup/incubation friendly": ["创业孵化友好", "适合创业"],
    "elite alumni network": ["精英校友网络", "校友资源强"],
    "strong sports programmes": ["体育项目强", "体育资源好"],
}

CLUSTER_ZH_MAP = {
    "business": "商科",
    "economics": "经济",
    "finance": "金融",
    "social science": "社会科学",
    "policy": "公共政策",
    "engineering": "工程",
    "computer science": "计算机",
    "data science": "数据科学",
    "law": "法律",
    "medicine": "医学",
}


COUNTRY_CODE_MAP = {
    "UK": "UK",
    "US": "US",
    "Canada": "CA",
    "Australia": "AU",
    "加拿大": "CA",
    "澳大利亚": "AU",
    "英国": "UK",
    "美国": "US",
}


def parse_country_heading(line):
    match = re.match(r"^##\s+(.+?)\s+([A-Za-z]+)", line.strip())
    if not match:
        return None

    country = match.group(1).strip()
    country_code_raw = match.group(2).strip()

    return {
        "country": country,
        "country_code": COUNTRY_CODE_MAP.get(country_code_raw, country_code_raw),
    }


def parse_school_name(line):
    match = re.match(r"^###\s+(.+)$", line.strip())
    if not match:
        return ""

    return match.group(1).strip()


def extract_bullet_value(block, keyword):
    for line in block.splitlines():
        line = line.strip()

        if not line.startswith("-"):
            continue

        if keyword not in line:
            continue

        match = re.search(r"\*\*.*?\*\*:\s*(.+)$", line)
        if match:
            return match.group(1).strip()

        return line.split(":", 1)[1].strip().lstrip("*").strip()

    return ""


def split_list_value(value):
    if not value:
        return []

    parts = re.split(r"，|；|;|\),\s*", value)
    cleaned = []

    for item in parts:
        item = item.strip()
        if not item:
            continue

        if item.endswith(")"):
            cleaned.append(item)
        else:
            cleaned.append(item + ")" if "(" in item else item)

    return cleaned


def enrich_terms(values, mapping):
    terms = []

    for value in values:
        terms.append(value)
        normalized = value.lower()

        for key, zh_value in mapping.items():
            if key.lower() == normalized:
                if isinstance(zh_value, list):
                    terms.extend(zh_value)
                else:
                    terms.append(zh_value)

    return list(dict.fromkeys(terms))


def build_school_content(record):
    lines = [
        f"国家：{record['country']} {record['country_code']}",
        f"学校：{record['school_name_en']} {record['school_name_zh']}",
        f"代表项目方向：{record['programme']}",
        f"学术门槛：{record['academic_bar']}",
        f"语言要求：雅思 IELTS {record['ielts']}",
        f"额外要求：{record['extra_requirements']}",
        f"选拔热度：{record['selectivity']}",
        f"参考费用：{record['cost']}",
        f"适配专业集群：{'；'.join(record['clusters_enriched'])}",
        f"价值点标签：{'；'.join(record['value_tags_enriched'])}",
    ]

    return "\n".join([line for line in lines if line.strip()])


def split_school_markdown(text, source_file):
    chunks = []
    current_country = {"country": "", "country_code": ""}
    current_school = ""
    current_lines = []

    def flush_school():
        if not current_school or not current_lines:
            return

        raw_content = "\n".join(current_lines).strip()

        programme = extract_bullet_value(raw_content, "Programme")
        academic_bar = extract_bullet_value(raw_content, "Academic bar")
        ielts = extract_bullet_value(raw_content, "IELTS")
        extra_requirements = extract_bullet_value(raw_content, "Extra requirements")
        selectivity = extract_bullet_value(raw_content, "Selectivity")
        cost = extract_bullet_value(raw_content, "Approx. cost")
        clusters = split_list_value(extract_bullet_value(raw_content, "Suited clusters"))
        value_tags = split_list_value(extract_bullet_value(raw_content, "Value-point tags"))

        school_zh = SCHOOL_ZH_MAP.get(current_school, "")
        clusters_enriched = enrich_terms(clusters, CLUSTER_ZH_MAP)
        value_tags_enriched = enrich_terms(value_tags, TAG_ZH_MAP)

        record = {
            "country": current_country.get("country", ""),
            "country_code": current_country.get("country_code", ""),
            "school_name_en": current_school,
            "school_name_zh": school_zh,
            "programme": programme,
            "academic_bar": academic_bar,
            "ielts": ielts,
            "extra_requirements": extra_requirements,
            "selectivity": selectivity,
            "cost": cost,
            "clusters": clusters,
            "clusters_enriched": clusters_enriched,
            "value_tags": value_tags,
            "value_tags_enriched": value_tags_enriched,
        }

        chunks.append(
            {
                "content": build_school_content(record),
                "raw_content": raw_content,
                "section_title": current_school,
                "source_ref": f"{source_file}#{current_school.replace(' ', '-')}",
                "metadata": {
                    "doc_type": "school_profile",
                    **record,
                },
            }
        )

    for line in text.splitlines():
        country = parse_country_heading(line)
        if country:
            flush_school()
            current_country = country
            current_school = ""
            current_lines = []
            continue

        school_name = parse_school_name(line)
        if school_name:
            flush_school()
            current_school = school_name
            current_lines = [line]
            continue

        if current_school:
            current_lines.append(line)

    flush_school()
    return chunks