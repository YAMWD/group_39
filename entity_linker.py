from importlib import import_module
import numpy as np
import spacy
import spacy.cli as spacy_cli
import spacy.language as spacy_lang
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def download_and_init_nlp(model_name: str, **kwargs) -> spacy_lang.Language:
    """
    Load a spaCy model and download it if it is not installed.
    :param model_name: Name of the spaCy model, e.g., en_core_web_sm.
    :param kwargs: Additional options to pass to the spaCy loader, such as component exclusion.
    :return: An initialized spaCy Language object.
    """
    try:
        model_module = import_module(model_name)
    except ModuleNotFoundError:
        spacy_cli.download(model_name)
        model_module = import_module(model_name)

    return model_module.load(**kwargs)


def get_wiki_candidates(entity: str, max_results: int = 5) -> list[str]:
    """
    Fetch potential Wikipedia page titles for a given entity.
    :param entity: The entity name to search.
    :param max_results: Maximum number of results to return.
    :return: A list of Wikipedia page titles.
    """
    search_url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": entity,
        "srlimit": max_results
    }

    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        return [result['title'] for result in data['query']['search']]
    except requests.RequestException:
        return []


def get_wiki_content(title: str) -> str:
    """
    Retrieve the content of a Wikipedia page for a given title.
    :param title: Title of the Wikipedia page.
    :return: Extracted plain text content of the page.
    """
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


class EntityLinker:
    def __init__(self):
        """
        Initialize the Entity Linker with required NLP models and tools.
        """
        self.nlp = download_and_init_nlp("en_core_web_sm")
        self.vectorizer = CountVectorizer(stop_words='english')

    def extract_entities(self, text: str) -> list[str]:
        """
        Extract named entities from the input text.
        :param text: Input text to process.
        :return: A list of unique named entities.
        """
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'LOC', 'NORP']:
                entities.append(ent.text)
        return list(set(entities))

    def disambiguate_entity(self, entity: str, context: str) -> tuple[str, str]:
        """
        Disambiguate an entity using the Bag of Words approach.
        :param entity: The entity to disambiguate.
        :param context: The context text to compare against candidates.
        :return: A tuple containing the best matching Wikipedia title and its URL.
        """
        # Retrieve candidate pages from Wikipedia
        candidates = get_wiki_candidates(entity)
        if not candidates:
            return "", ""

        # Fetch content for each candidate
        candidate_contents = [get_wiki_content(title) for title in candidates]

        # Filter valid contents and corresponding titles
        valid_contents = []
        valid_titles = []
        for title, content in zip(candidates, candidate_contents):
            if content.strip():
                valid_contents.append(content)
                valid_titles.append(title)

        if not valid_contents:
            return "", ""

        # Create a document matrix with the context and candidate contents
        all_docs = [context] + valid_contents
        try:
            X = self.vectorizer.fit_transform(all_docs)
        except:
            return "", ""

        # Compute similarity between context and candidates
        similarities = cosine_similarity(X[0:1], X[1:]).flatten()

        # Get the best matching candidate
        if len(similarities) > 0:
            best_idx = np.argmax(similarities)
            best_title = valid_titles[best_idx]
            url = f"https://en.wikipedia.org/wiki/{best_title.replace(' ', '_')}"
            return best_title, url

        return "", ""

    def process_text(self, input_text: str) -> dict[str, str]:
        """
        Process the input text and link entities to Wikipedia pages.
        :param input_text: Text containing entities.
        :return: A dictionary mapping entities to Wikipedia URLs.
        """
        # Extract entities from the text
        entities = self.extract_entities(input_text)

        # Disambiguate and map each entity to a Wikipedia page
        entity_links = {}
        for entity in entities:
            _, url = self.disambiguate_entity(entity, input_text)
            if url:
                entity_links[entity] = url

        return entity_links


def main():
    """
    Example usage of the EntityLinker.
    """
    # Initialize the Entity Linker
    linker = EntityLinker()

    # Example input text
    text = """Is Rome the capital of Italy? surely it is but many don't know this fact that Italy was not always called as Italy. 
    Before Italy came into being in 1861, it had several names including Italian Kingdom, Roman Empire and the Republic of Italy among others. 
    If we start the chronicle back in time, then Rome was the first name to which Romans were giving credit. 
    Later this city became known as "Caput Mundi" or the capital of the world..."""

    # Process the text and get entity links
    entity_links = linker.process_text(text)

    # Print the results
    for entity, url in sorted(entity_links.items()):
        print(f"{entity} ⇒ {url}")


if __name__ == "__main__":
    main()