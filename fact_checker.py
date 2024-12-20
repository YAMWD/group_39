import re
import requests
import math
import nltk
from nltk.corpus import stopwords

# 下载并加载停用词
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# Wikidata API地址
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


def extract_keywords(question, answer, question_type="general"):
    """
    提取关系三元组关键词 (主体, 谓词, 客体)。

    参数：
        question (str): 问题文本。
        answer (str): 回答文本。
        question_type (str): 问题类型，"general" 或 "yes_no"。

    返回：
        list: 包含三个关键词的列表。
    """
    question_clean = question.strip()
    answer_clean = answer.strip()

    if question_type == "yes_no":
        # 提取是非题的答案
        answer_match = re.search(r'\b(yes|no)\b', answer_clean, re.IGNORECASE)
        extracted_answer = answer_match.group(1).lower() if answer_match else "unknown"

        # 从问题中提取主体和客体
        match = re.search(r"Is (.+?) the (\w+) of (.+?)\??", question_clean, re.IGNORECASE)
        if match:
            subject = match.group(1).strip()
            predicate = match.group(2).strip()
            obj = match.group(3).strip()
            return [subject, predicate, obj, extracted_answer]
    else:
        # 实体问题
        # 示例：The capital of France is...
        match = re.search(r"The capital of (.+?) is (.+)", question_clean, re.IGNORECASE)
        if match:
            obj = match.group(1).strip()
            subject = match.group(2).strip()
            return [subject, "capital", obj, extracted_answer]

        # 其他实体问题格式可在此扩展
        # 默认返回unknown关键词
        return ["unknown1", "unknown2", "unknown3", "unknown"]

    # 如果无法匹配，返回未知关键词
    return ["unknown1", "unknown2", "unknown3", "unknown"]


def wikipedia_search(query):
    """
    使用Wikipedia搜索API查找与query最相关的页面标题。

    参数：
        query (str): 搜索查询。

    返回：
        str or None: 页面标题，如果未找到则返回None。
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1
    }
    try:
        response = requests.get(WIKIPEDIA_API_URL, params=params, timeout=5)
        data = response.json()
        search_results = data.get("query", {}).get("search", [])
        if search_results:
            return search_results[0]["title"]
    except requests.RequestException as e:
        print(f"Error during Wikipedia search: {e}")
    return None


def wikipedia_extract(title):
    """
    获取Wikipedia页面的摘要文本。

    参数：
        title (str): 页面标题。

    返回：
        str: 页面摘要文本。
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "format": "json",
        "titles": title
    }
    try:
        response = requests.get(WIKIPEDIA_API_URL, params=params, timeout=5)
        data = response.json()
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        return page.get("extract", "")
    except requests.RequestException as e:
        print(f"Error during Wikipedia extract: {e}")
    return ""


def build_word_index_map(text):
    """
    清洗文本、去除停用词，并建立词典映射。

    参数：
        text (str): 要处理的文本。

    返回：
        tuple: (words列表, word2id字典)
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    # 去停用词
    filtered = [w for w in tokens if w not in STOPWORDS and w.strip() != ""]
    unique_words = list(set(filtered))
    word2id = {w: i for i, w in enumerate(unique_words)}
    return filtered, word2id


def find_positions(words, word2id, keyword):
    """
    找出关键词在文本中的所有出现位置。

    参数：
        words (list): 文本中的单词列表。
        word2id (dict): 单词到ID的映射。
        keyword (str): 要查找的关键词。

    返回：
        list: 关键词出现的位置列表。
    """
    positions = []
    k_lower = keyword.lower()
    for idx, w in enumerate(words):
        if w == k_lower:
            positions.append(idx)
    return positions


def compute_score(words, word2id, keywords):
    """
    计算关键词之间的距离得分。

    参数：
        words (list): 文本中的单词列表。
        word2id (dict): 单词到ID的映射。
        keywords (list): 三个关键词列表。

    返回：
        float: 计算得到的得分。
    """
    pos = [find_positions(words, word2id, kw) for kw in keywords]

    # 如果有关键词不存在，则得分为0
    if any(len(p) == 0 for p in pos):
        return 0.0

    best_score = 0.0
    # 遍历所有组合，计算距离得分
    for p1 in pos[0]:
        for p2 in pos[1]:
            for p3 in pos[2]:
                distance = max(p1, p2, p3) - min(p1, p2, p3)
                score = math.exp(-distance)
                if score > best_score:
                    best_score = score
    return best_score


def wikidata_query(entity, property_id="P36"):
    """
    使用Wikidata API查询实体的特定属性。

    参数：
        entity (str): 要查询的实体名称。
        property_id (str): Wikidata属性ID（默认"P36"表示首都）。

    返回：
        str or None: 属性值的名称，如果查询失败或无结果则返回None。
    """
    # Step 1: 搜索实体对应的QID
    search_params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": entity
    }
    try:
        response = requests.get(WIKIDATA_API_URL, params=search_params, timeout=5)
        data = response.json()
        search_results = data.get("search", [])
        if not search_results:
            return None
        qid = search_results[0]["id"]
    except requests.RequestException as e:
        print(f"Error during Wikidata search: {e}")
        return None

    # Step 2: 使用SPARQL查询特定属性值
    sparql_url = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT ?itemLabel WHERE {{
      wd:{qid} wdt:{property_id} ?item.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    headers = {"Accept": "application/sparql-results+json"}
    try:
        response = requests.get(sparql_url, params={"query": query}, headers=headers, timeout=5)
        res_data = response.json()
        bindings = res_data["results"]["bindings"]
        if bindings:
            return bindings[0]["itemLabel"]["value"]
        return None
    except requests.RequestException as e:
        print(f"Error during Wikidata SPARQL query: {e}")
        return None


def wikidata_query_for_fact(keywords):
    """
    根据关系三元组关键词使用Wikidata查询事实。

    参数：
        keywords (list): 四个关键词列表 [主体, 谓词, 客体, extracted_answer]。

    返回：
        bool or None: 如果能够验证事实，返回True或False；否则返回None。
    """
    if len(keywords) == 4:
        subject = keywords[0]
        predicate = keywords[1].lower()
        obj = keywords[2]
        extracted_answer = keywords[3]

        if predicate == "capital":
            # 验证 subject 是否为 obj 的首都
            correct_capital = wikidata_query(obj, "P36")  # P36表示首都
            if correct_capital is None:
                return None
            return correct_capital.lower() == subject.lower()
    return None


def fact_check(question, answer, question_type="general"):
    """
    进行事实核验，判断回答是否正确。

    参数：
        question (str): 问题文本。
        answer (str): 回答文本。
        question_type (str): 问题类型，"general" 或 "yes_no"。

    返回：
        str: "correct", "incorrect", 或 "unknown"。
    """
    # Step 1: 提取关键词
    keywords = extract_keywords(question, answer, question_type)
    print(f"Extracted Keywords: {keywords}")

    # Step 2: 使用Wikidata查询验证
    wd_result = wikidata_query_for_fact(keywords)
    if wd_result is not None:
        return "correct" if wd_result else "incorrect"

    # Step 3: 若Wikidata无法直接验证，则进行关键词距离检查
    # 以客体关键词为主实体搜索Wikipedia文本
    entity_to_search = keywords[2]
    title = wikipedia_search(entity_to_search)
    if title is None:
        return "unknown"
    text = wikipedia_extract(title)
    if not text.strip():
        return "unknown"

    # Step 4: 清洗文本并建立词典
    words, word2id = build_word_index_map(text)

    # Step 5: 计算关键词距离得分
    score = compute_score(words, word2id, keywords[:3])  # 不包括 extracted_answer
    print(f"Distance Score: {score}")

    # Step 6: 判断结果
    threshold = 0.1
    if score >= threshold:
        return "correct"
    else:
        return "incorrect"


# 主程序示例
if __name__ == "__main__":
    # 示例1：实体问题
    question1 = "The capital of France is..."
    answer1 = "The capital of France is Paris."
    result1 = fact_check(question1, answer1, "general")
    print(f"Question: {question1}\nAnswer: {answer1}\nResult: {result1}\n")

    # 示例2：yes/no问题
    question2 = "Is Tokyo the capital of Japan?"
    answer2 = "Yes, Tokyo is indeed the capital of Japan."
    result2 = fact_check(question2, answer2, "yes_no")
    print(f"Question: {question2}\nAnswer: {answer2}\nResult: {result2}\n")

    # 示例3：无法验证的问题
    question3 = "Is Pluto still considered a planet?"
    answer3 = "No, Pluto was reclassified as a dwarf planet."
    result3 = fact_check(question3, answer3, "yes_no")
    print(f"Question: {question3}\nAnswer: {answer3}\nResult: {result3}\n")

    # 示例4：特殊问题类型
    question4 = "Is Beijing the capital of China?"
    answer4 = "Yes, Beijing is the capital of China."
    result4 = fact_check(question4, answer4, "yes_no")
    print(f"Question: {question4}\nAnswer: {answer4}\nResult: {result4}\n")