import re

with open('frontend/css/style.css', 'r') as f:
    css = f.read()

# Remove old #app-workspace
css = re.sub(r'#app-workspace\s*\{[^}]+\}', '', css)
# Remove old #pixel-canvas
css = re.sub(r'/\*\s*Pixel Canvas Background\s*\*/\s*#pixel-canvas\s*\{[^}]+\}', '', css)
css = re.sub(r'#pixel-canvas\s*\{[^}]+\}', '', css)
css = re.sub(r'/\*\s*Pixel Canvas Background\s*\*/', '', css)

new_styles = '''
#app-background-wrapper {
    position: relative;
    width: 100%;
    min-height: 800px;
    background: #F8F9FA;
    padding-bottom: 50px;
    display: flex;
    justify-content: center;
    overflow: hidden;
}

#app-workspace {
    position: relative;
    z-index: 50;
    margin-top: -80px;
    background: white;
    border-radius: 12px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.1);
    width: 90%;
    max-width: 1200px;
}

#pixel-canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 1;
    pointer-events: none;
}

.flow-img {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    width: 200px;
    height: auto;
    border-radius: 8px;
    opacity: 0.4;
    z-index: 2;
    object-fit: cover;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.flow-img.left {
    left: 2%;
    filter: grayscale(100%);
}

.flow-img.right {
    right: 2%;
    filter: grayscale(0%);
}
'''

# Clean trailing whitespace and append
css = css.rstrip() + "\n\n" + new_styles.strip() + "\n"

with open('frontend/css/style.css', 'w') as f:
    f.write(css)

# NOW we modify app.js
with open('frontend/js/app.js', 'r') as f:
    js = f.read()

# Replace the particle size
js = re.sub(r'this\.size = Math\.random\(\) \* 2 \+ 1;\s*//[^\\n]*', 'this.size = Math.random() * 8 + 4;', js)

# Replace particle x spawn logic inside reset
js = re.sub(r'reset\(\) \{\s*this\.x = 0;', 'reset() {\n            this.x = Math.random() * 100;', js)

# Replace the speedX logic
js = re.sub(r'this\.speedX = Math\.random\(\) \* 1\.5 \+ 0\.5;\s*//[^\\n]*', 'this.speedX = Math.random() * 2 + 0.5;', js)

with open('frontend/js/app.js', 'w') as f:
    f.write(js)
