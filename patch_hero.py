import re

# Patch index.html
with open('frontend/index.html', 'r') as f:
    html = f.read()

hero_html = """<body>

<div id="scroll-wrapper">
    <section id="hero-sticky">
        <img id="hero-bw" src="https://picsum.photos/id/1015/1920/1080?grayscale" alt="B&W Image">
        <img id="hero-color" src="https://picsum.photos/id/1015/1920/1080" alt="olor Image">
        <div class="hero-text text-white text-center position-absolute top-50 start-50 translate-middle w-100" style="z-index: 10; pointer-events: none; text-shadow: 0 4px 15px rgba(0,0,0,0.8);">
            <h1 class="display-3 fw-bold">ColorizeAI</h1>
            <p class="fs-4">Scroll to discover the magic</p>
        </div>
    </section>
</div>

<section id="app-workspace">
"""

# replace <body class="bg-light"> with the hero + section open
html = re.sub(r'<body class="bg-light">\s*', hero_html, html)

# find the end to close the app-workspace section
# right before the scripts at the bottom
close_section = """</simport re

# Patch index.htm->
# Patchl =with open('frontoot    html = f.read()

hero_html = """<body>op
hero_html = """<b.ht
<div id="scroll-wraf.w    <section id="hero-ste.        <img id="hero-bw" srcty        <img id="hero-color" src="https://picsum.photos/id/1015/1920/1080" alt="Color Image">
    3        <div class="hero-text text-white text-center position-absolute top-50 start-50 transh;            <h1 class="display-3 fw-bold">ColorizeAI</h1>
            <p class="fs-4">Scroll to discover the magic</p>
        </div>
    </section>
</div>

<section id="app-workspace">
"""

# rft            <p class="fs-4">Scro: 100%;
    object-fit: co        </div>
    </section>
</div>

<section id="app-workt-    </sectionin</div>

<sectto
<secom,"""

# replace <body class=al
#roghtml = re.sub(r'<body class="bg-light">\s*', hero_html, html), 
# find the end to close the app-workspace section
# right b, b# right before the scripts at the bottom
close_spaclose_section = """</simport re

# Patc25
# Patch index.htm->
# Patchl -wo# Patchl =with opeti
hero_html = """<body>op
hero_html = """<b.ht
<ghthero_html = """<b.ht
<he<div id="scroll-wrain    3        <div class="hero-text text-white text-center position-absolute top-50 start-50 transh;            <h1 class="display-3 fw-bold">ColorizeAI</h1>
            <pon            <p class="fs-4">Scroll to discover the magic</p>
        </div>
    </section>
</div>

<section id="app-workspace">
"""

# rft            <p cl-w        </div>
    </section>
</div>

<secetElementById('hero    </section  </div>

<sectpp
<sec he"""

# rft            <p cldd
#ent    object-fit: co        </div>
    </sectt     </section>
</div>

<sectiongC</div>

<sect  
<sec   
<sectto
<secom,"""

# replace <body class=  c<secomcr
# replacce #roghtml = re.sub(r'<bo  # find the end to close the app-workspace section
# right b, b# risc# right b, b# right before the scripts at the bopeclose_spaclose_section = """</simport re

# Patc25
nt
# Patc25
# Patch index.htm->
# Patchl    # Patchor.style.setProperty('hero_html = """<body>op
hero_htgehero_html = """<b.ht
<}
<ghthero_html = """at<he<div id="scroll-wraith            <pon            <p class="fs-4">Scroll to discover the magic</p>
        </div>
    </section>
cat << 'EOF' > patch_html.py
with open('frontend/index.html', 'r') as f:
    html = f.read()

html = html.replace('<body class="bg-light">', '''<body class="bg-light">

<div id="scroll-wrapper">
    <section id="hero-sticky">
        <img id="hero-bw" src="https://picsum.photos/id/1015/1920/1080?grayscale" alt="B&W Image">
        <img id="hero-color" src="https://picsum.photos/id/1015/1920/1080" alt="Color Image">
        <div class="hero-text text-white text-center position-absolute top-50 start-50 translate-middle w-100" style="z-index: 10; pointer-events: none; text-shadow: 0 4px 15px rgba(0,0,0,0.8);">
            <h1 class="display-3 fw-bold">ColorizeAI</h1>
            <p class="fs-4">Scroll to discover the magic</p>
        </div>
    </section>
</div>

<section id="app-workspace">
''')

html = html.replace('<!-- Bootstrap JS -->', '''</section>

<!-- Bootstrap JS -->''')

with open('frontend/index.html', 'w') as f:
    f.write(html)
