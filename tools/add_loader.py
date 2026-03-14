import re

# Update HTML
with open('frontend/index.html', 'r') as f:
    html = f.read()

for target_id in ['singleSpinner', 'batchSpinner', 'videoSpinner', 'metricsSpinner']:
    pattern = r'<div class="spinner-border[^>]+id="' + target_id + r'"[^>]*></div>'
    replacement = f'<div class="pixel-loader-container" id="{target_id}" style="display:none;"><div class="pixel-loader-text">Colorizing...</div><div class="pixel-loader-track"><div class="pixel-loader-fill"></div></div></div>'
    html = re.sub(pattern, replacement, html)

with open('frontend/index.html', 'w') as f:
    f.write(html)

# Update JS
with open('frontend/js/app.js', 'r') as f:
    js = f.read()

for target_id in ['singleSpinner', 'batchSpinner', 'videoSpinner', 'metricsSpinner']:
    js = js.replace(f"document.getElementById('{target_id}').style.display = 'inline-block';", f"document.getElementById('{target_id}').style.display = 'flex';")

with open('frontend/js/app.js', 'w') as f:
    f.write(js)

# Update CSS
with open('frontend/css/style.css', 'r') as f:
    css = f.read()

if 'pixel-loader-container' not in css:
    css += '''

/* Pixel Loader Styles */
.pixel-loader-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    padding: 40px 0;
    gap: 12px;
}

.pixel-loader-text {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: #555;
    animation: pulse 1.5s infinite;
}

.pixel-loader-track {
    width: 250px;
    height: 16px;
    background: repeating-linear-gradient(to right, #333 0, #333 8px, #555 8px, #555 16px);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
}

.pixel-loader-fill {
    height: 100%;
    width: 0%;
    background: repeating-linear-gradient(to right, #00C9FF 0, #00C9FF 8px, #92FE9D 8px, #92FE9D 16px, #FFDD00 16px, #FFDD00 24px, #ff007f 24px, #ff007f 32px);
    position: absolute;
    top: 0;
    left: 0;
    animation: pixel-sweep 2s linear infinite;
}

@keyframes pixel-sweep {
    0% { width: 0%; left: 0; right: auto; }
    50% { width: 100%; left: 0; right: auto; }
    50.001% { width: 100%; right: 0; left: auto; }
    100% { width: 0%; right: 0; left: auto; }
}

@keyframes pulse {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
}
'''
    with open('frontend/css/style.css', 'w') as f:
        f.write(css)

