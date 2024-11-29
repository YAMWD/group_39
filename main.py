from llama_cpp import Llama
from entity_linker import EntityLinker
from question import Question, read_questions, write_output

model_path = "models/llama-2-7b.Q4_K_M.gguf"
llm = Llama(model_path=model_path, verbose=False)

def get_raw_answer(question_text: str) -> str:
    """Get raw answer text from the language model."""
    print(f"Question: {question_text}")
    prompt = f"Question: {question_text}\nAnswer: "
    output = llm(
          prompt, # Prompt
          max_tokens=50, # Generate up to 32 tokens
          stop=["\n"], # Stop generating just before the model would generate a new question
          echo=False # Echo the prompt back in the output
    )
    raw_answer = output['choices'][0]['text']
    print(f"Raw answer: {raw_answer}")
    return raw_answer

def main(input_file_path, output_file_path):
    questions = read_questions(input_file_path)
    entity_linker = EntityLinker()
    for question in questions:
        question.raw_answer = get_raw_answer(question.text)
        question.entities = entity_linker.process_text(question.raw_answer)
    write_output(output_file_path, questions)

if __name__ == '__main__':
    main("example_input.txt", "output.txt")
