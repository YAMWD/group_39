from llama_cpp import Llama

model_path = "models/llama-2-7b.Q4_K_M.gguf"
llm = Llama(model_path=model_path, verbose=False)

def get_raw_answer(question_text: str) -> str:

    print(f"Question: {question_text}")

    prompt = f"Question: {question_text}\nAnswer: "

    # Generate the answer using the LLaMA model
    output = llm(
        prompt,          # The question prompt
        max_tokens=50,   # Limit the length of the generated answer
        stop=["\n"],     # Stop generation when a newline character is encountered
        echo=False       # Do not include the prompt in the output
    )

    # Extract the raw answer from the model's output
    raw_answer = output['choices'][0]['text'].strip()

    print(f"Raw answer: {raw_answer}")

    return raw_answer

def main():
    # Example question to ask the model
    question = "Is Rome the capital of Italy?"

    # Get the raw answer from the model
    answer = get_raw_answer(question)

    # Print the question and the corresponding answer
    print(f"Question: {question}\nAnswer: {answer}")

if __name__ == "__main__":
    main()