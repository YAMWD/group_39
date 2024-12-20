# data_format.py
import re

# 定义输入和输出文件路径
input_file = '../question_augmented.csv'
output_file = '../question_augmented_format.csv'

# 定义分隔符
input_delimiter = '\t'
output_delimiter = '\t'

# 定义分隔符
input_delimiter = '\t'
output_delimiter = '\t'


processed_data = []

bracket_pattern = re.compile(r"[\[\]']")

with open(input_file, 'r', encoding='utf-8', errors='replace') as infile:
    for line_number, line in enumerate(infile, 1):

        line = line.strip()
        if not line:
            continue


        print(f"处理第 {line_number} 行: {line}")


        parts = line.split(input_delimiter, maxsplit=2)


        if len(parts) != 3:
            print(f"警告：第 {line_number} 行字段数量不足 3，跳过。内容：{line}")
            continue


        _, question, label = parts


        original_question = question
        question = bracket_pattern.sub('', question).strip()


        print(f"原问题: {original_question} -> 清理后: {question}")


        processed_data.append([question, label.strip()])


with open(output_file, 'w', encoding='utf-8') as outfile:
    for index, (question, label) in enumerate(processed_data, 1):
        new_line = f"{index}{output_delimiter}{question}{output_delimiter}{label}\n"
        outfile.write(new_line)


print(f"数据已成功处理并保存到 '{output_file}' 文件中。")
print(f"总处理数据量: {len(processed_data)} 条")