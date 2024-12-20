# extract_answer.py
import re

import numpy as np
import torch
from torch import cosine_similarity
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from entity_linker import get_wiki_content

# Define keywords for "Yes" and "No"
YES_KEYWORDS = ["yes", "yeah", "yep", "of course", "definitely", "sure", "absolutely", "certainly"]
NO_KEYWORDS = ["no", "nope", "not really", "never", "nah", "absolutely not", "certainly not"]

# Model path for the Yes/No classifier
YESNO_MODEL_PATH = "./yesno_classifier"

yesno_tokenizer = AutoTokenizer.from_pretrained(YESNO_MODEL_PATH)
yesno_model = AutoModelForSequenceClassification.from_pretrained(YESNO_MODEL_PATH)
yesno_model.eval()
yesno_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yesno_model.to(yesno_device)

# Initialize the NER model
# Use a pre-trained model from Hugging Face
NER_MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")


from sentence_transformers import SentenceTransformer, util
import torch

# 全局初始化句向量模型
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def select_final_entity(entities: dict[str, str], context: str) -> tuple[str, str]:
    if not entities:
        return "No relevant entity found", ""

    # Step 1: Extract context-based features
    def extract_keywords(text):
        # words with more than 3 characters
        return [word.lower() for word in re.findall(r'\b\w{4,}\b', text)]

    context_keywords = set(extract_keywords(context))

    # Step 2: Rule-based filtering
    def filter_entities(entities, keywords):
        filtered_entities = {}
        for entity, url in entities.items():
            # Keep entities that overlap with context keywords
            entity_keywords = set(extract_keywords(entity))
            if entity_keywords & keywords:
                filtered_entities[entity] = url
        return filtered_entities

    # Apply rule filtering
    filtered_entities = filter_entities(entities, context_keywords)
    if not filtered_entities:  # Fallback to original entities if none remain
        filtered_entities = entities

    # Step 3: Prepare candidate texts for similarity scoring
    vectorizer = CountVectorizer(stop_words="english")
    candidate_texts = []
    entity_names = list(filtered_entities.keys())

    for entity in entity_names:
        # Combine entity name, Wikipedia content, and context for comparison
        wikipedia_content = get_wiki_content(entity)
        candidate_text = f"{entity} {wikipedia_content} {context}"
        candidate_texts.append(candidate_text)

    # Prepare input for vectorization
    all_texts = [context] + candidate_texts  # Include context as the first text
    try:
        X = vectorizer.fit_transform(all_texts)
    except Exception as e:
        print(f"Vectorization failed: {e}")
        return "No relevant entity found", ""

    # Step 4: Compute cosine similarity
    context_vector = X[0]  # First row is the context
    candidate_vectors = X[1:]  # Remaining rows are candidates
    similarities = cosine_similarity(context_vector, candidate_vectors).flatten()

    # Step 5: Select the best entity based on similarity
    best_idx = int(np.argmax(similarities))
    best_entity = entity_names[best_idx]
    best_url = filtered_entities[best_entity]

    return best_entity, best_url


def classify_yes_no_answer(raw_answer: str) -> str:
    inputs = yesno_tokenizer(raw_answer, return_tensors="pt", truncation=True, padding=True).to(yesno_device)
    with torch.no_grad():
        outputs = yesno_model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    predicted_class = torch.argmax(logits, dim=-1).item()
    model_label = yesno_model.config.id2label[predicted_class]
    model_confidence = probabilities[predicted_class]

    ans_lower = raw_answer.lower()
    yes_count = sum(kw in ans_lower for kw in YES_KEYWORDS)
    no_count = sum(kw in ans_lower for kw in NO_KEYWORDS)

    CONFIDENCE_THRESHOLD = 0.6

    if model_confidence < CONFIDENCE_THRESHOLD:
        if yes_count > no_count:
            final_label = "Yes"
        elif no_count > yes_count:
            final_label = "No"
        else:
            final_label = "Uncertain"
    else:
        final_label = model_label

    return final_label


def extract_entity_answer(question_text: str, raw_answer: str, entities: dict[str, str]) -> str:
    if not entities:
        return "No entity found"

    # Infer desired entity type from the question
    if question_text.lower().startswith("who"):
        desired_entity = "PERSON"
    elif question_text.lower().startswith("where"):
        desired_entity = "LOC"
    elif question_text.lower().startswith("when"):
        desired_entity = "DATE"
    else:
        desired_entity = None  # No filtering by default

    # Use NER to refine the candidates based on the desired type
    ner_results = ner_pipeline(raw_answer)
    filtered_entities = entities  # Default to all entities if no filtering is needed

    if desired_entity is not None:
        # Extract NER results matching the desired entity type
        ner_matched = set(
            ent["word"] for ent in ner_results if ent["entity_group"] == desired_entity
        )

        # Filter entities based on NER-matched names
        filtered_entities = {
            name: url for name, url in entities.items() if any(name.lower() in ner.lower() for ner in ner_matched)
        }

        # Fallback to all entities if no type-based match is found
        if not filtered_entities:
            filtered_entities = entities

    # Use similarity-based selection to find the most relevant entity
    best_entity, best_url = select_final_entity(filtered_entities, raw_answer)

    # If no relevant entity is found, return a default response
    if best_entity == "No relevant entity found":
        return "No entity found"

    # Return the selected entity name
    return best_entity


def extract_answer(question) -> str:
    if not hasattr(question, 'label') or not hasattr(question, 'raw_answer'):
        raise AttributeError("Question object must have 'label' and 'raw_answer' attributes.")

    if question.label == "yes_no":
        return classify_yes_no_answer(question.raw_answer)
    elif question.label == "entity":
        return extract_entity_answer(question.text, question.raw_answer, question.entities)
    else:
        return "Unknown question type."
