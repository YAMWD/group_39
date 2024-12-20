import re

# Define input and output file paths
input_file = '../question_augmented.csv'  # Path to the input file
output_file = '../question_augmented_format.csv'  # Path to the output file

# Define delimiters for input and output files
input_delimiter = '\t'  # Delimiter for reading the input file (tab-separated)
output_delimiter = '\t'  # Delimiter for writing the output file (tab-separated)

# Initialize a list to store processed data
processed_data = []

# Compile a regular expression to remove square brackets and single quotes
bracket_pattern = re.compile(r"[\[\]']")

# Open the input file for reading
with open(input_file, 'r', encoding='utf-8', errors='replace') as infile:
    for line_number, line in enumerate(infile, 1):  # Iterate through each line with a line number
        line = line.strip()  # Remove leading/trailing whitespace
        if not line:  # Skip empty lines
            continue

        print(f"Processing line {line_number}: {line}")  # Debug: Print the current line being processed

        # Split the line into parts using the input delimiter
        parts = line.split(input_delimiter, maxsplit=2)

        # Check if the line contains exactly 3 fields
        if len(parts) != 3:
            print(f"Warning: Line {line_number} does not have 3 fields, skipping. Content: {line}")
            continue

        _, question, label = parts  # Extract the fields (ignoring the first column)

        # Store the original question for debugging purposes
        original_question = question
        # Remove brackets and single quotes from the question and strip whitespace
        question = bracket_pattern.sub('', question).strip()

        # Debug: Print before and after cleaning the question
        print(f"Original question: {original_question} -> Cleaned: {question}")

        # Add the cleaned question and label to the processed data list
        processed_data.append([question, label.strip()])

# Open the output file for writing
with open(output_file, 'w', encoding='utf-8') as outfile:
    for index, (question, label) in enumerate(processed_data, 1):  # Write each processed line
        # Format the output line with an index, question, and label
        new_line = f"{index}{output_delimiter}{question}{output_delimiter}{label}\n"
        outfile.write(new_line)  # Write the formatted line to the output file

# Print a summary of the processing
print(f"Data successfully processed and saved to '{output_file}'.")
print(f"Total number of processed lines: {len(processed_data)}")