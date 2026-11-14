import re

with open("raw_markdown.md", 'r', encoding="utf-8") as fl:
    lines = fl.readlines()

new_lines = "\n".join(lines)
new_lines = new_lines.replace(r"\(", "$").replace(r"\)", "$")

with open("markdown.md", 'w', encoding="utf-8") as fl:
    fl.write(new_lines)
