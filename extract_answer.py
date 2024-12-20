from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Define keywords for "Yes" and "No"
YES_KEYWORDS = ["yes", "yeah", "yep", "of course", "definitely", "sure", "absolutely", "certainly"]
NO_KEYWORDS = ["no", "nope", "not really", "never", "nah", "absolutely not", "certainly not"]

# Initialize the NER model
# Use a pre-trained model from Hugging Face
NER_MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")


def extract_yes_no_answer(raw_answer: str) -> str:
    ans_lower = raw_answer.lower()
    yes_count = sum(kw in ans_lower for kw in YES_KEYWORDS)
    no_count = sum(kw in ans_lower for kw in NO_KEYWORDS)

    if yes_count > no_count:
        return "Yes"
    elif no_count > yes_count:
        return "No"
    else:
        return "Uncertain"


def extract_entity_answer(question_text: str, raw_answer: str, entities: dict) -> str:
    if not entities:
        # If no entity linking results are provided, fall back to NER
        ner_results = ner_pipeline(raw_answer)
        if not ner_results:
            return "No entity found in the answer."

        # Infer the desired entity type based on the question type
        if question_text.lower().startswith("who"):
            desired_entity = "PERSON"
        elif question_text.lower().startswith("where"):
            desired_entity = "LOC"
        elif question_text.lower().startswith("when"):
            desired_entity = "DATE"
        else:
            desired_entity = None  # No filtering by default

        # Filter entities based on the desired type
        for entity in ner_results:
            if desired_entity is None or entity["entity_group"] == desired_entity:
                return entity["word"]

        return ner_results[0]["word"]  # Return the first entity if no type matches
    else:
        # Infer the desired entity type based on the question type
        if question_text.lower().startswith("who"):
            desired_entity = "PERSON"
        elif question_text.lower().startswith("where"):
            desired_entity = "LOC"
        elif question_text.lower().startswith("when"):
            desired_entity = "DATE"
        else:
            desired_entity = None  # No filtering by default

        # Filter and return the appropriate entity
        for entity, url in entities.items():
            # Assuming entity type can be inferred from the URL or EntityLinker results
            # Simply return the first entity for now
            return entity

        return next(iter(entities.keys()), "No entity found")


def extract_answer(question) -> str:
    if not hasattr(question, 'label') or not hasattr(question, 'raw_answer'):
        raise AttributeError("Question object must have 'label' and 'raw_answer' attributes.")

    if question.label == "yes_no":
        return extract_yes_no_answer(question.raw_answer)
    elif question.label == "entity":
        return extract_entity_answer(question.text, question.raw_answer, question.entities)
    else:
        return "Unknown question type."