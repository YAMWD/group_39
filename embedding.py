from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import torch.nn as nn

def compute_embedding(texts):
    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    # Tokenize the document
    inputs = tokenizer(texts, return_tensors = "pt", padding = True, truncation = True)

    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)

    return embeddings

def compute_cos(embeddings):
    
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    candidates = torch.unsqueeze(embeddings[1:], 0) if embeddings[1:].dim() == 1 else embeddings[1:]

    context = torch.unsqueeze(embeddings[0], 0).expand(candidates.shape[0], -1)

    return cos(context, candidates)

def disambiguate(texts):

    embeddings = compute_embedding(texts)

    cos_sim = compute_cos(embeddings)

    return torch.argmax(cos_sim)

if __name__ == '__main__':
    texts = ["The capital of Italy.", "Rome (Italian and Latin: Roma), is the capital city of Italy.", "artificial intelligence is dedicated to help human being"]
    best = disambiguate(texts)
    print(best.item(), texts[1:][best])