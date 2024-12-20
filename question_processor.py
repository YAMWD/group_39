from classifier import QuestionClassifier
from extract_answer import extract_answer
from entity_linker import EntityLinker
from llm_client import get_raw_answer
import logging


class QuestionProcessor:
    def __init__(self, classifier_model_path):
        self.logger = logging.getLogger(__name__)
        # Initialize the question classifier
        self.classifier = QuestionClassifier(classifier_model_path)
        # Initialize the entity linker
        self.entity_linker = EntityLinker()

    def process_question(self, question):
        try:
            # Step 1: Classify the question (e.g., "yes_no" or "entity")
            question.label = self.classifier.classify(question.text)

            # Step 2: Get the raw answer from the language model
            question.raw_answer = get_raw_answer(question.text)

            # Step 3: Perform entity linking (only relevant for "entity" type questions)
            if question.label == "entity":
                question.entities = self.entity_linker.process_text(question.raw_answer)
            else:
                question.entities = {}

            # Step 4: Extract the final answer based on the question type
            question.extracted_answer = extract_answer(question)

            self.logger.info(f"Question ID {question.id} processed successfully.")
        except Exception as e:
            # Log and handle any errors that occur during processing
            self.logger.error(f"Error processing Question ID {question.id}: {e}")
            question.extracted_answer = "Error extracting answer."