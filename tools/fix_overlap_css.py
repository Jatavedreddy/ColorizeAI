import re

with open('frontend/css/style.css', 'r') as f:
    css = f.read()

# Make hero-sticky z-index explicitly 10 and remove any trace of overflow
css = re.sub(r'#hero-sticky\s*\{[^}]+\}', '''#hero-sticky {
    height: 100vh;
    position: sticky;
    top: 0;
    z-index: 10;
    display: flex;
    justify-content: center;
    align-items: center;
}''', css)

# Make scroll wrapper z-index explicitly lower too
css = re.sub(r'#scroll-wrapper\s*\{[^}]+\}', '''#scroll-wrapper {
    height: 300vh;
    position: relative;
    background-color: #000;
    z-index: 5;
}''', css)

# Fix #app-workspace strict rules
new_ws = '''#app-workspace {
    position: relative;
    z-index: 9999 !important;
    margin-top: -100px !important;
    background: white;
    border-radius: 16px !important;
    box-shadow: 0px -10px 40px rgba(0,0,0,0.15);
    width: 90%;
    max-width: 1200px;
    padding: 30px;
    padding-top: 20px;
    overflow: visible !important;
}'''
css = re.sub(r'#app-workspace\s*\{[^}]+\}', new_ws, css)

# Fix #app-background-wrapper strict rules
new_bg = '''#app-background-wrapper {
    position: relative;
    width: 100%;
    min-height: 800px;
    background: #F8F9FA;
    padding-bottom: 50px;
    display: flex;
    justify-content: center;
    overflow: visible !important;
}'''
css = re.sub(r'#app-background-wrapper\s*\{[^}]+\}', new_bg, css)

# Fix tabs visibility
new_tab = '''
.nav-tabs {
    position: relative;
    z-index: 10000;
    display: flex;
    background: transparent;
    border-bottom: 1px solid #e9ecef;
    margin-bottom: 20px;
}
'''
css = re.sub(r'\.nav-tabs\s*\{\s*\n\s*border-bottom:[^}]+\}', new_tab, css)

# Double check that no other rogue overflow: hidden exists globally outside .output-container
css = re.sub(r'overflow:\s*hidden;', '/* override hiding here */', css) 
# Wait, we only want to ensure output containers have hidden so we don't break that
css = css.replace('.output-container {\n    background-color: #fff;\n    min-height: 250px;\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    padding: 10px;\n    /* override hiding here */\n}', '.output-container {\n    background-color: #fff;\n    min-height: 250px;\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    padding: 10px;\n    overflow: hidden;\n}')

with open('frontend/css/style.css', 'w') as f:
    f.write(css)

