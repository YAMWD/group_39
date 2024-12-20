# entity_linker.py

from importlib import import_module
import numpy as np
import spacy.cli as spacy_cli
import spacy.language as spacy_lang
import requests
import torch
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

sentence_model = SentenceTransformer('all-mpnet-base-v2')

def download_and_init_nlp(model_name: str, **kwargs) -> spacy_lang.Language:
    try:
        model_module = import_module(model_name)
    except ModuleNotFoundError:
        spacy_cli.download(model_name)
        model_module = import_module(model_name)
    return model_module.load(**kwargs)

def get_wiki_candidates(entity: str, max_results: int = 5) -> list[str]:
    """
    Fetch potential Wikipedia page titles for a given entity.
    Add filters to ensure only relevant results are returned.
    """
    search_url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": entity,
        "srlimit": max_results,
    }

    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        candidates = [
            result['title']
            for result in data['query']['search']
            if "disambiguation" not in result['title'].lower()  # Filter disambiguation pages
        ]
        return candidates
    except requests.RequestException as e:
        print(f"Error fetching Wikipedia candidates for '{entity}': {e}")
        return []

def get_wiki_content(title: str) -> str:
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        page = next(iter(data['query']['pages'].values()))
        return page.get('extract', '')
    except requests.RequestException:
        return ""


embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def select_final_entity(entities: dict[str, str], context: str) -> tuple[str, str]:
    """
    Select the most relevant entity using semantic similarity.
    :param entities: Dictionary of entities and their Wikipedia URLs.
    :param context: Context text for disambiguation.
    :return: Tuple containing the best entity and its URL.
    """
    if not entities:
        return "No relevant entity found", ""

    # Prepare text pairs for similarity comparison
    entity_names = list(entities.keys())
    context_embeddings = sentence_model.encode([context])
    entity_embeddings = sentence_model.encode(entity_names)

    # Compute cosine similarity
    similarities = np.dot(context_embeddings, entity_embeddings.T).flatten()

    # Select the best matching entity
    if similarities.size > 0:
        best_idx = int(np.argmax(similarities))
        best_entity = entity_names[best_idx]
        best_url = entities[best_entity]
        return best_entity, best_url

    return "No relevant entity found", ""
    # 将上下文与实体内容转化为向量
    embeddings = embedding_model.encode([context] + entity_contents, convert_to_tensor=True)
    context_emb = embeddings[0].unsqueeze(0)
    candidate_embs = embeddings[1:]

    # 使用余弦相似度计算相似度
    similarities = util.pytorch_cos_sim(context_emb, candidate_embs).flatten()

    best_idx = int(torch.argmax(similarities))
    best_entity = entity_names[best_idx]
    best_url = entities[best_entity]

    return best_entity, best_url

class EntityLinker:
    def __init__(self):
        self.nlp = download_and_init_nlp("en_core_web_sm")

    def extract_entities(self, text: str) -> list[str]:
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'LOC', 'NORP']:
                entities.append(ent.text)
        return list(set(entities))

    def disambiguate_entity(self, entity: str, context: str) -> tuple[str, str]:
        """
        根据原有逻辑获取候选项，然后交给 select_final_entity 进行最终选择。
        """
        candidates = get_wiki_candidates(entity)
        if not candidates:
            return "", ""

        # 构造一个临时 entities 字典供 select_final_entity 使用
        temp_entities = {candidate: f"https://en.wikipedia.org/wiki/{candidate.replace(' ', '_')}" for candidate in candidates}

        # 使用 select_final_entity 从 candidate 中选出最合适的实体
        best_entity, best_url = select_final_entity(temp_entities, context)
        if best_entity == "No relevant entity found":
            return "", ""
        return best_entity, best_url

    def process_text(self, input_text: str) -> dict[str, str]:
        entities = self.extract_entities(input_text)
        entity_links = {}
        for entity in entities:
            _, url = self.disambiguate_entity(entity, input_text)
            if url:
                entity_links[entity] = url
        return entity_links

def main():
    linker = EntityLinker()
    text = """Is Rome the capital of Italy? surely it is but many don't know this fact that Italy was not always called as Italy. 
    Before Italy came into being in 1861, it had several names including Italian Kingdom, Roman Empire and the Republic of Italy among others. 
    If we start the chronicle back in time, then Rome was the first name to which Romans were giving credit. 
    Later this city became known as "Caput Mundi" or the capital of the world..."""
    entity_links = linker.process_text(text)
    for entity, url in sorted(entity_links.items()):
        print(f"{entity} ⇒ {url}")

if __name__ == "__main__":
    main()