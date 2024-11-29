from typing import Optional

class Question:
    def __init__(self, question_line: str) -> None:
        """
        Question line format:
        <ID question><TAB>text of the question/completion<newline>

        Examples:
        ```plaintext
        question-001<TAB>Is Managua the capital of Nicaragua?<newline>
        ```
        """
        question_id, text = question_line.strip().split('\t')
        self.question_id: str = question_id
        self.text: str = text
        self.raw_answer: Optional[str] = None
        self.extracted_answer: Optional[str] = None
        self.is_correct: Optional[bool] = None
        self.entities: Optional[dict[str, str]] = None  # key: entity, value: URL

    def format_output(self) -> str:
        """
        Your output should be a file where the answer are in the following format

        <ID question><TAB>[R,A,C,E]<answer> where:
        - "R" indicates the raw text produced by the language model,
        - "A" is the extracted answer,
        - "C" is the tag correct/incorrect
        - "E" are the entities extracted.

        For instance, for the example above the output should be:

        ```plaintext
        question-001<TAB>R"Most people think Managua is the capital of Nicaragua.
        However, Managua is not the capital of Nicaragua.
        The capital of Nicaragua is Managua.
        The capital of Nicaragua is Managua. Managua is the capital of Nicaragua.
        The capital"<newline>
        question-001<TAB>A"no"<newline>
        question-001<TAB>C"incorrect"
        question-001<TAB>E"Nicaragua"<TAB>"https://en.wikipedia.org/wiki/Nicaragua"
        question-001<TAB>E"Managua"<TAB>"https://en.wikipedia.org/wiki/Managua"
        ```
        """
        raw_answer = f"{self.question_id}\tR\"{self.raw_answer}\""
        extracted_answer = f"{self.question_id}\tA\"{self.extracted_answer}\""
        is_correct = f"{self.question_id}\tC\"{'correct' if self.is_correct else 'incorrect'}\""
        entities: list[str] = [f"{self.question_id}\tE\"{entity}\"\t\"{url}\"" for entity, url in self.entities.items()]
        return '\n'.join([raw_answer, extracted_answer, is_correct] + entities)


def read_questions(file_path: str) -> list[Question]:
    with open(file_path, 'r') as file:
        return [Question(line) for line in file.readlines()]

def write_output(file_path: str, questions: list[Question]) -> None:
    with open(file_path, 'w') as file:
        for question in questions:
            file.write(question.format_output() + '\n')
