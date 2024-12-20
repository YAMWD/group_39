from importlib import import_module
import numpy as np
import spacy.cli as spacy_cli
import spacy.language as spacy_lang
import requests
import torch
from sentence_transformers import SentenceTransformer, util

# Initialize a pre-trained sentence transformer model for semantic similarity
sentence_model = SentenceTransformer('all-mpnet-base-v2')

def download_and_init_nlp(model_name: str, **kwargs) -> spacy_lang.Language:
    """
    Download and initialize a spaCy NLP model.
    If the model is not available locally, it will be downloaded.
    :param model_name: Name of the spaCy model (e.g., "en_core_web_sm").
    :param kwargs: Additional arguments for model initialization.
    :return: Loaded spaCy NLP model.
    """
    try:
        model_module = import_module(model_name)
    except ModuleNotFoundError:
        spacy_cli.download(model_name)  # Download the model if not found
        model_module = import_module(model_name)
    return model_module.load(**kwargs)

def get_wiki_candidates(entity: str, max_results: int = 5) -> list[str]:
    """
    Fetch potential Wikipedia page titles for a given entity.
    Uses the Wikipedia API to retrieve search results.
    :param entity: The entity name to search for.
    :param max_results: Maximum number of search results to return.
    :return: A list of Wikipedia page titles.
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
        response = requests.get(search_url, params=params)  # Send a GET request to the API
        response.raise_for_status()
        data = response.json()
        # Filter and return relevant page titles (excluding disambiguation pages)
        candidates = [
            result['title']
            for result in data['query']['search']
            if "disambiguation" not in result['title'].lower()
        ]
        return candidates
    except requests.RequestException as e:
        print(f"Error fetching Wikipedia candidates for '{entity}': {e}")
        return []

def get_wiki_content(title: str) -> str:
    """
    Fetch the introductory content of a Wikipedia page.
    :param title: Title of the Wikipedia page.
    :return: Introductory text of the page or an empty string if unavailable.
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
        page = next(iter(data['query']['pages'].values()))  # Extract page data
        return page.get('extract', '')  # Return the introduction text
    except requests.RequestException:
        return ""

# Initialize another sentence transformer for additional processing
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def select_final_entity(entities: dict[str, str], context: str) -> tuple[str, str]:
    """
    Select the most relevant entity from a list of candidates using semantic similarity.
    :param entities: Dictionary of entities and their corresponding Wikipedia URLs.
    :param context: Context text to help disambiguate the entity.
    :return: The best matching entity and its Wikipedia URL.
    """
    if not entities:
        return "No relevant entity found", ""

    # Prepare embeddings for context and entity names
    entity_names = list(entities.keys())
    context_embeddings = sentence_model.encode([context])  # Encode the context
    entity_embeddings = sentence_model.encode(entity_names)  # Encode entity names

    # Compute cosine similarity between context and entity embeddings
    similarities = np.dot(context_embeddings, entity_embeddings.T).flatten()

    # Select the best matching entity
    if similarities.size > 0:
        best_idx = int(np.argmax(similarities))  # Get the index of the best match
        best_entity = entity_names[best_idx]
        best_url = entities[best_entity]
        return best_entity, best_url

    return "No relevant entity found", ""

class EntityLinker:
    def __init__(self):
        # Load the spaCy model for named entity recognition (NER)
        self.nlp = download_and_init_nlp("en_core_web_sm")

    def extract_entities(self, text: str) -> list[str]:
        """
        Extract named entities from the input text using spaCy.
        :param text: Input text.
        :return: A list of extracted entities.
        """
        doc = self.nlp(text)  # Process text with spaCy
        entities = []
        # Filter entities by specific types (e.g., ORG, PERSON, etc.)
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'LOC', 'NORP']:
                entities.append(ent.text)
        return list(set(entities))  # Remove duplicate entities

    def disambiguate_entity(self, entity: str, context: str) -> tuple[str, str]:
        """
        Disambiguate an entity by finding the best match from Wikipedia candidates.
        :param entity: The entity to disambiguate.
        :param context: Context text to assist in disambiguation.
        :return: The best matching entity and its Wikipedia URL.
        """
        candidates = get_wiki_candidates(entity)  # Fetch candidates from Wikipedia
        if not candidates:
            return "", ""

        # Create a temporary dictionary of candidates and their URLs
        temp_entities = {candidate: f"https://en.wikipedia.org/wiki/{candidate.replace(' ', '_')}" for candidate in candidates}

        # Select the most relevant entity using context
        best_entity, best_url = select_final_entity(temp_entities, context)
        if best_entity == "No relevant entity found":
            return "", ""
        return best_entity, best_url

    def process_text(self, input_text: str) -> dict[str, str]:
        """
        Process the input text to extract and link entities to Wikipedia pages.
        :param input_text: The input text to process.
        :return: A dictionary of entities and their Wikipedia URLs.
        """
        entities = self.extract_entities(input_text)  # Extract named entities
        entity_links = {}
        # Disambiguate and link each entity
        for entity in entities:
            _, url = self.disambiguate_entity(entity, input_text)
            if url:
                entity_links[entity] = url
        return entity_links

def main():
    """
    Main function to demonstrate entity linking.
    """
    linker = EntityLinker()  # Initialize the EntityLinker
    text = """Is Rome the capital of Italy? surely it is but many don't know this fact that Italy was not always called as Italy. 
    Before Italy came into being in 1861, it had several names including Italian Kingdom, Roman Empire and the Republic of Italy among others. 
    If we start the chronicle back in time, then Rome was the first name to which Romans were giving credit. 
    Later this city became known as "Caput Mundi" or the capital of the world..."""
    entity_links = linker.process_text(text)  # Extract and link entities
    # Print the results
    for entity, url in sorted(entity_links.items()):
        print(f"{entity} ⇒ {url}")

if __name__ == "__main__":
    main()