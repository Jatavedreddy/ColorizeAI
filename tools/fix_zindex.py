import re

with open('frontend/css/style.css', 'r') as f:
    css = f.read()

# Fix #app-workspace z-index
css = re.sub(r'(#app-workspace\s*\{[^}]*?z-index:\s*)\d+', r'\g<1>999', css)

# Make sure hero-sticky has lower z-index and fix overflow
css = re.sub(r'(#hero-sticky\s*\{[^}]*?)overflow:\s*hidden;', r'\1overflow-x: hidden;\n    z-index: 10;', css)

# Fix #app-background-wrapper overflow
css = re.sub(r'(#app-background-wrapper\s*\{[^}]*?)overflow:\s*hidden;', r'\1overflow-x: hidden;\n    overflow-y: visible;', css)

# Fix .flow-img
css = re.sub(r'(\.flow-img\s*\{[^}]*?opacity:\s*)[\d\.]+;', r'\g<1>0.15;', css)
css = re.sub(r'(\.flow-img\s*\{[^}]*?z-index:\s*)\d+', r'\g<1>1', css)

# Fix .flow-img.left
css = re.sub(r'(\.flow-img\.left\s*\{[^}]*?left:\s*)[^;]+;', r'\g<1>-5%;', css)

# Fix .flow-img.right
css = re.sub(r'(\.flow-img\.right\s*\{[^}]*?right:\s*)[^;]+;', r'\g<1>-5%;', css)

# Fix #pixel-canvas z-index
css = re.sub(r'(#pixel-canvas\s*\{[^}]*?z-index:\s*)\d+', r'\g<1>2', css)

with open('frontend/css/style.css', 'w') as f:
    f.write(css)
