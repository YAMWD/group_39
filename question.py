import pandas as pd


class Question:
    def __init__(self, id, text):
        """
        Initialize a Question object.
        :param id: Unique identifier for the question.
        :param text: Text content of the question.
        """
        self.id = id
        self.text = text
        self.label = None  # To be set during classification
        self.raw_answer = ""  # Raw answer generated by the language model
        self.entities = {}  # Dictionary of entities linked to their corresponding data
        self.extracted_answer = ""  # Extracted answer from the raw answer


def read_questions(input_file_path):
    """
    Read questions from an input file and return a list of Question objects.
    Input file format: Tab-separated file containing "id" and "text" columns.
    :param input_file_path: Path to the input file.
    :return: List of Question objects.
    """
    df = pd.read_csv(input_file_path, sep='\t', names=["id", "text"])
    return [Question(row['id'], row['text']) for idx, row in df.iterrows()]


def write_output(output_file_path, questions):
    """
    Write the processed Question objects to an output file.
    Output file format: Tab-separated file containing six columns:
        "id", "text", "label", "raw_answer", "entities", "extracted_answer".
    :param output_file_path: Path to the output file.
    :param questions: List of processed Question objects.
    """
    # Prepare the data for writing
    data = [{
        "id": q.id,
        "text": q.text,
        "label": q.label,
        "raw_answer": q.raw_answer,
        "entities": q.entities,
        "extracted_answer": q.extracted_answer
    } for q in questions]
    df = pd.DataFrame(data)

    # Save the data to a tab-separated file
    df.to_csv(output_file_path, sep='\t', index=False, header=False)

    """
    Expected Output Format:
    The output should include the answer in the following format:

    <ID question><TAB>[R,A,C,E]<answer>

    Where:
    - "R" indicates the raw text produced by the language model.
    - "A" is the extracted answer.
    - "C" is a correctness tag ("correct"/"incorrect").
    - "E" contains the extracted entities along with their associated information.

    Example Output:
    ```
    question-001<TAB>R"Most people think Managua is the capital of Nicaragua.
    However, Managua is not the capital of Nicaragua.
    The capital of Nicaragua is Managua. Managua is the capital of Nicaragua.
    The capital of Nicaragua is Managua."<newline>
    question-001<TAB>A"no"<newline>
    question-001<TAB>C"incorrect"<newline>
    question-001<TAB>E"Nicaragua"<TAB>"https://en.wikipedia.org/wiki/Nicaragua"<newline>
    question-001<TAB>E"Managua"<TAB>"https://en.wikipedia.org/wiki/Managua"<newline>
    ```
    """

    """
    Input File Line Format:
    Each line in the input file should follow this format:
    <ID question><TAB>text of the question/completion<newline>

    Example Input:
    ```
    question-001<TAB>Is Managua the capital of Nicaragua?<newline>
    ```
    """