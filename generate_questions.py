from llama_cpp import Llama

def generate_questions(model_path, prompt, output_file, num_questions=10):
    llm = Llama(model_path=model_path)

    # Adjust the prompt to ensure the correct format for generating questions
    complete_prompt = prompt  # Modify the prompt if needed for customization

    # Adjust temperature and top_p to increase output diversity
    output = llm(
        prompt=complete_prompt,
        max_tokens=256,  # Limit the generation length to avoid redundant output
        temperature=0.8,  # Increase diversity in generated questions
        top_p=0.95,  # Balance between quality and randomness
        echo=True  # Include the prompt in the output
    )


    generated_text = output["choices"][0]["text"]

    # Print the generated questions to the console
    print("Generated Questions:")
    print(generated_text)

    # Save the generated questions to the specified file
    with open(output_file, "w") as f:
        f.write(generated_text)


def main():
    model_path = "./models/llama-2-7b.Q4_K_M.gguf"  # Path to the pre-trained LLaMA model
    prompt = """
    Below is an instruction for generating training questions for a classification task. You will produce a list of 10 distinct questions. Each question should be formatted as:

    question-XXX<TAB><question text>

    Where:
    - `XXX` is a three-digit question number starting from 001 and incrementing by 1 for each question.
    - <TAB> represents a tab character.

    Requirements:
    1. Generate exactly 10 questions, numbered from 001 to 010.
    2. The questions should be diverse and cover various domains: geography, history, science, culture, etc.
    3. At least half should be yes/no questions (e.g., "Is X the capital of Y?", "Did X happen before Y?", "Is Z known for...?").
    4. The others should be entity-type questions requiring a completion of knowledge (e.g., "The currency used in X is...", "The leader of X in year Y was...", "The largest city in Z is...").
    5. Vary the style of wording. For example:
       - Yes/No sample: "Is Managua the capital of Nicaragua?"
       - Entity completion sample: "The currency used in Japan is..."
    6. Keep each question grammatically correct and meaningful. Avoid repetition of the same entities.

    Now, generate the list:

    question-001<TAB>
    """

    generate_questions(
        model_path=model_path,
        prompt=prompt,
        output_file="generated_questions_batch.txt",
        num_questions=10
    )


if __name__ == "__main__":
    main()