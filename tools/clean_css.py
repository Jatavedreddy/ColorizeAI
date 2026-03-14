import re

with open('frontend/css/style.css', 'r') as f:
    css = f.read()

# Remove flow-img styles
css = re.sub(r'\.flow-img\s*\{[^}]+\}', '', css)
css = re.sub(r'\.flow-img\.left\s*\{[^}]+\}', '', css)
css = re.sub(r'\.flow-img\.right\s*\{[^}]+\}', '', css)

with open('frontend/css/style.css', 'w') as f:
    f.write(css)
