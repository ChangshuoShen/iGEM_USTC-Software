import csv
import random

# 定义要随机填充的属性
attributes = ['Easy', 'Medium', 'Hard']

# 读取CSV文件并添加随机字段
def add_random_attribute(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as infile, \
         open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # 读取CSV的头部并在最后添加一个新的字段名
        header = next(reader)
        header.append('Difficulty')
        writer.writerow(header)
        
        # 逐行读取数据，并在最后添加一个随机属性
        for row in reader:
            row.append(random.choice(attributes))
            writer.writerow(row)

# 输入CSV文件路径和输出CSV文件路径
input_csv = 'data/riddles.csv'
output_csv = 'data/riddles1.csv'

# 调用函数处理CSV文件
add_random_attribute(input_csv, output_csv)