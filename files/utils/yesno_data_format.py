# yesno_data_format.py
import re

# Define input and output file paths
input_file = '../question_augmented.csv'
output_file = '../question_augmented_format.csv'

# Define delimiters
input_delimiter = '\t'
output_delimiter = '\t'

# Regular expression to clean brackets and single quotes
bracket_pattern = re.compile(r"[\[\]']")

# List to store processed data
processed_data = []

# Open the input file and process it line by line
with open(input_file, 'r', encoding='utf-8', errors='replace') as infile:
    for line_number, line in enumerate(infile, 1):
        # Strip whitespace from the line
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        print(f"Processing line {line_number}: {line}")

        # Split the line into at most three parts
        parts = line.split(input_delimiter, maxsplit=2)

        # Skip the line if it doesn't have exactly 3 parts
        if len(parts) != 3:
            print(f"Warning: Line {line_number} does not have 3 fields. Skipping. Content: {line}")
            continue

        # Extract fields
        _, question, label = parts

        # Clean the question field
        original_question = question
        question = bracket_pattern.sub('', question).strip()

        print(f"Original question: {original_question} -> Cleaned: {question}")

        # Add the cleaned question and label to the processed data
        processed_data.append([question, label.strip()])

# Write the processed data to the output file
with open(output_file, 'w', encoding='utf-8') as outfile:
    for index, (question, label) in enumerate(processed_data, 1):
        new_line = f"{index}{output_delimiter}{question}{output_delimiter}{label}\n"
        outfile.write(new_line)

# Print summary information
print(f"Data successfully processed and saved to '{output_file}'.")
print(f"Total number of processed entries: {len(processed_data)}")