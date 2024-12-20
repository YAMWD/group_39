import re
import requests
import math
import nltk
from nltk.corpus import stopwords

# Download and load stopwords
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# Wikipedia API URLs
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

def extract_keywords(question, answer):

    question_clean = question.strip()
    answer_clean = answer.strip()

    # Try to match Yes/No style questions
    match = re.search(r"Is\s+(.+?)\s+the\s+(\w+)\s+of\s+(.+?)\??", question_clean, re.IGNORECASE)
    if match:
        subject = match.group(1).strip()
        predicate = match.group(2).strip()
        obj = match.group(3).strip()
        return [subject, predicate, obj]

    # Try to match Entity style questions
    match = re.search(r"The\s+(\w+)\s+of\s+(.+?)\s+is\s+(.+)", question_clean, re.IGNORECASE)
    if match:
        predicate = match.group(1).strip()
        obj = match.group(2).strip()
        subject = match.group(3).strip().rstrip('.').strip()
        return [subject, predicate, obj]

    # Fallback to using answer if patterns fail
    if answer_clean:
        return [answer_clean, "unknown", question_clean]

    return ["unknown1", "unknown2", "unknown3"]

def wikipedia_search(query):
    """
    Search for a Wikipedia page title using the Wikipedia API.
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
    Retrieve the summary text of a Wikipedia page.
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
    Build a word-to-index mapping for the text.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    filtered = [w for w in tokens if w not in STOPWORDS and w.strip() != ""]
    unique_words = list(set(filtered))
    word2id = {w: i for i, w in enumerate(unique_words)}
    return filtered, word2id

def find_positions(words, keyword):
    """
    Find all positions of a keyword in the text.
    """
    positions = [idx for idx, w in enumerate(words) if w == keyword.lower()]
    return positions

def compute_score(words, keywords, decay=1.0):
    """
    Compute the distance-based score for the keywords in the text.
    """
    positions = [find_positions(words, kw) for kw in keywords]

    # If any keyword is not found, return a score of 0
    if any(len(pos) == 0 for pos in positions):
        return 0.0

    best_score = 0.0
    # Iterate through all combinations of keyword positions
    for p1 in positions[0]:
        for p2 in positions[1]:
            for p3 in positions[2]:
                distance = max(p1, p2, p3) - min(p1, p2, p3)
                score = math.exp(-decay * distance)
                best_score = max(best_score, score)
    return best_score

def fact_check(question, answer):
    """
    Perform fact-checking for a question and answer pair.
    """
    keywords = extract_keywords(question, answer)
    print(f"Extracted Keywords: {keywords}")

    # Search Wikipedia for the object (e.g., country or entity)
    entity_to_search = keywords[2]
    title = wikipedia_search(entity_to_search)
    if title is None:
        return "incorrect"
    text = wikipedia_extract(title)
    if not text.strip():
        return "incorrect"

    # Build the word index map and compute the distance score
    words, word2id = build_word_index_map(text)
    score = compute_score(words, keywords)
    print(f"Distance Score: {score}")

    # Threshold for determining correctness
    threshold = 0.1
    return "correct" if score >= threshold else "incorrect"