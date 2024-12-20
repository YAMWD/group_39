from importlib import import_module

import requests

from llama_cpp import Llama
import spacy
from spacy.cli import download
from spacy.language import Language

def download_and_init_nlp(model_name: str, **kwargs) -> Language:
    """Load a spaCy model, download it if it has not been installed yet.
    :param model_name: the model name, e.g., en_core_web_sm
    :param kwargs: options passed to the spaCy loader, such as component exclusion
    :return: an initialized spaCy Language
    """
    try:
        model_module = import_module(model_name)
    except ModuleNotFoundError:
        download(model_name)
        model_module = import_module(model_name)

    return model_module.load(**kwargs)

def get_answer():
    model_path = "models/llama-2-7b.Q4_K_M.gguf"
    # models: https://huggingface.co/TheBloke.

    question = "What is the capital of Italy? "
    llm = Llama(model_path=model_path, verbose=False)
    print("Asking the question \"%s\" to %s (wait, it can take some time...)" % (question, model_path))
    output = llm(
          question, # Prompt
          max_tokens=128, # Generate up to 32 tokens
          stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
          echo=True # Echo the prompt back in the output
    )
    print("Here is the output")
    print(output['choices'])

def get_NER(text: str):
    # Specify the model name you want to download, e.g., "en_core_web_sm"
    NER_model_name = "en_core_web_sm"
    nlp = download_and_init_nlp(NER_model_name)
    # ents = NER(output['choices'][0]['text']).ents
    doc = nlp(text)
    ents = doc.ents
    return ents

def get_candidates_wikipedia(mention: str):
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": mention,
        "format": "json",
        "srlimit": 5  # Limit the number of candidates
    }
    response = requests.get(url, params=params)
    results = response.json().get("query", {}).get("search", [])
    candidates = [result["title"] for result in results]
    return candidates

if __name__ == "__main__":
    ents = get_NER("Tennis champion Emerson was expected to win Wimbledon.")
    get_candidates_wikipedia("Emerson")