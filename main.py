from question import read_questions, write_output
from question_processor import QuestionProcessor
import logging

def main(input_file_path, output_file_path):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Initialize the question processor with the specified classifier model path
    classifier_model_path = "./distilbert_classifier"
    question_processor = QuestionProcessor(classifier_model_path)
    logger.info("Question processor initialized.")

    # Read questions from the input file
    questions = read_questions(input_file_path)
    logger.info(f"Loaded {len(questions)} questions from {input_file_path}.")

    # Process each question
    for question in questions:
        question_processor.process_question(question)

    # Write the processed results to the output file
    write_output(output_file_path, questions)
    logger.info(f"All questions processed and saved to {output_file_path}.")

if __name__ == '__main__':
    # Run the main function with example file paths
    main("example_input.txt", "output.txt")