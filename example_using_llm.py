from importlib import import_module
import requests
from llama_cpp import Llama
import spacy
from spacy.cli import download
from spacy.language import Language

def download_and_init_nlp(model_name: str, **kwargs) -> Language:
    """
    Load a spaCy model and download it if it is not already installed.
    :param model_name: Name of the spaCy model, e.g., en_core_web_sm.
    :param kwargs: Additional arguments passed to the spaCy loader.
    :return: An initialized spaCy Language object.
    """
    try:
        model_module = import_module(model_name)
    except ModuleNotFoundError:
        download(model_name)
        model_module = import_module(model_name)

    return model_module.load(**kwargs)

def get_answer():
    """
    Use a Llama model to answer a predefined question.
    The model path and question are hardcoded for demonstration.
    """
    model_path = "models/llama-2-7b.Q4_K_M.gguf"
    # Model source: https://huggingface.co/TheBloke.

    question = "What is the capital of Italy?"
    llm = Llama(model_path=model_path, verbose=False)
    print(f"Asking the question \"{question}\" to {model_path} (wait, it can take some time...)")
    output = llm(
        question,  # Prompt
        max_tokens=128,  # Generate up to 128 tokens
        stop=["Q:", "\n"],  # Stop at specific tokens to prevent generating unrelated questions
        echo=True  # Echo the input question in the output
    )
    print("Here is the output")
    print(output['choices'])

def get_NER(text: str):
    """
    Perform Named Entity Recognition (NER) on the given text using spaCy.
    :param text: Input text for NER.
    :return: A list of named entities detected in the text.
    """
    # Specify the spaCy model to use
    NER_model_name = "en_core_web_sm"
    nlp = download_and_init_nlp(NER_model_name)

    # Process the input text to extract named entities
    doc = nlp(text)
    ents = doc.ents
    return ents

def get_candidates_wikipedia(mention: str):
    """
    Fetch candidate Wikipedia titles for a given mention using the Wikipedia API.
    :param mention: The mention or entity to search for.
    :return: A list of Wikipedia page titles.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": mention,
        "format": "json",
        "srlimit": 5  # Limit the number of candidate results
    }
    response = requests.get(url, params=params)
    results = response.json().get("query", {}).get("search", [])
    candidates = [result["title"] for result in results]
    return candidates

if __name__ == "__main__":
    # Example usage: Extract named entities from text
    ents = get_NER("Tennis champion Emerson was expected to win Wimbledon.")
    print(f"Named Entities: {[ent.text for ent in ents]}")

    # Fetch Wikipedia candidates for a specific mention
    candidates = get_candidates_wikipedia("Emerson")
    print(f"Wikipedia Candidates: {candidates}")