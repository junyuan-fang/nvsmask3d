import os
import re

# 定义要替换的旧路径和新路径
old_path_pattern = r"- /\n      - scratch\n      - project_2003267\n      - junyuan"
new_path_replacement = "- nvsmask3d\n      - data\n      - ScannetPP"

# 根目录（包含所有 config.yml 文件的目录）
target_directory = "outputs"

def replace_in_file(file_path, pattern, replacement):
    """读取文件并替换指定的多行字符串"""
    with open(file_path, 'r') as file:
        content = file.read()
    
    # 使用正则表达式替换多行路径
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # 如果内容有变化，则写回文件
    if new_content != content:
        with open(file_path, 'w') as file:
            file.write(new_content)
        print(f"Updated {file_path}")
    else:
        print(f"No changes made to {file_path}")

# 遍历指定目录查找 config.yml 文件并替换路径
for root, _, files in os.walk(target_directory):
    for file in files:
        if file == "config.yml":
            file_path = os.path.join(root, file)
            replace_in_file(file_path, old_path_pattern, new_path_replacement)

print("Path replacement complete.")
