from llama_cpp import Llama
import spacy
from spacy.cli import download
from spacy.language import Language
from importlib import import_module

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

model_path = "models/llama-2-7b.Q4_K_M.gguf"

# If you want to use larger models...
#model_path = "models/llama-2-13b.Q4_K_M.gguf"
# All models are available at https://huggingface.co/TheBloke. Make sure you download the ones in the GGUF format

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

# Specify the model name you want to download, e.g., "en_core_web_sm"
NER_model_name = "en_core_web_sm"
NER = download_and_init_nlp(NER_model_name)
ents = NER(output['choices'][0]['text']).ents
for ent in ents:
      print(ent.text, ent.label_)