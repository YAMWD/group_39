# classifier.py

from transformers import pipeline
import logging


class QuestionClassifier:
    def __init__(self, model_path):
        self.logger = logging.getLogger(__name__)
        try:
            # Load the text classification pipeline with the specified model and tokenizer
            self.classifier = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=model_path,
                return_all_scores=True
            )
            self.logger.info("Question classifier has been successfully loaded.")
        except Exception as e:
            self.logger.error(f"Failed to load the classifier: {e}")
            raise e

    def classify(self, question_text):
        try:
            result = self.classifier(question_text)
            if not result:
                return "unknown"

            # Assuming the classifier returns a list of predictions, sort by probability
            sorted_result = sorted(result[0], key=lambda x: x['score'], reverse=True)
            top_label = sorted_result[0]['label'].lower()

            # Map the top label to predefined categories
            if top_label in ["yes_no", "yes/no", "yesno"]:
                return "yes_no"
            elif top_label in ["entity", "entities"]:
                return "entity"
            else:
                return "unknown"
        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            return "unknown"