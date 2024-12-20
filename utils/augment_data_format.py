import re

# 定义输入和输出文件路径
input_file = '../question_augmented.csv'  # 原始数据文件
output_file = '../question_augmented_format.csv'  # 处理后输出的文件

# 定义分隔符
input_delimiter = '\t'  # 原始数据分隔符为制表符
output_delimiter = '\t'  # 输出文件分隔符为制表符

# 定义分隔符
input_delimiter = '\t'  # 原始数据分隔符为制表符
output_delimiter = '\t'  # 输出文件分隔符为制表符

# 存储处理后的数据
processed_data = []

# 正则表达式模式，去除所有中括号和单引号
bracket_pattern = re.compile(r"[\[\]']")

# 打开输入文件并逐行处理
with open(input_file, 'r', encoding='utf-8', errors='replace') as infile:
    for line_number, line in enumerate(infile, 1):
        # 去除行首行尾空白符
        line = line.strip()
        if not line:
            continue  # 跳过空行

        # 调试信息：打印当前行内容
        print(f"处理第 {line_number} 行: {line}")

        # 按指定分隔符分割数据，确保问题字段的分隔符不会影响处理
        parts = line.split(input_delimiter, maxsplit=2)

        # 检查字段数目是否符合预期
        if len(parts) != 3:
            print(f"警告：第 {line_number} 行字段数量不足 3，跳过。内容：{line}")
            continue

        # 提取字段
        _, question, label = parts

        # 清理问题内容，去除中括号和单引号
        original_question = question
        question = bracket_pattern.sub('', question).strip()

        # 调试信息：打印清理前后的问题内容
        print(f"原问题: {original_question} -> 清理后: {question}")

        # 添加到处理后的数据列表
        processed_data.append([question, label.strip()])

# 写入输出文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    for index, (question, label) in enumerate(processed_data, 1):
        # 构造输出行：序号<分隔符>问题内容<分隔符>标签
        new_line = f"{index}{output_delimiter}{question}{output_delimiter}{label}\n"
        outfile.write(new_line)

# 打印处理完成信息
print(f"数据已成功处理并保存到 '{output_file}' 文件中。")
print(f"总处理数据量: {len(processed_data)} 条")