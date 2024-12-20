# main.py
from question import read_questions, write_output
from question_processor import QuestionProcessor


def main(input_file_path, output_file_path):
    # Initialize the question processor with the specified classifier model path
    classifier_model_path = "./question_type_classifier"
    question_processor = QuestionProcessor(classifier_model_path)

    # Read questions from the input file
    questions = read_questions(input_file_path)
    for question in questions:
        question_processor.process_question(question)

    # Write the processed results to the output file
    write_output(output_file_path, questions)


if __name__ == '__main__':
    # Run the main function with example file paths
    main("example_input.txt", "example_output_2.txt")
