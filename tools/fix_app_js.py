import re

def fix_app_js():
    with open('frontend/js/app.js', 'r') as f:
        content = f.read()

    # Find the start of the pixel canvas code
    start_str = '// --- Pixel Canvas Background Animation ---'
    idx = content.find(start_str)
    
    if idx != -1:
        # We rewrite the whole block from start_str to the end of file
        content = content[:idx] + start_str + '''
document.addEventListener("DOMContentLoaded", function() {
    const canvas = document.getElementById("pixel-canvas");
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    let particles = [];
    
    function resizeCanvas() {
        const parent = document.getElementById("app-background-wrapper");
        if (parent) {
            canvas.width = parent.offsetWidth;
            canvas.height = parent.offsetHeight;
        } else {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
    }
    
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();

    class Particle {
        constructor() {
            this.reset();
            // Start spread across the screen
            this.x = Math.random() * canvas.width;
        }

        reset() {
            this.x = Math.random() * 100; // spawn on left side randomly
            this.y = Math.random() * canvas.height;
            this.size = Math.random() * 8 + 4; // bigger pixels 4px to 12px
            this.speedX = Math.random() * 2 + 0.5; // moving right
            this.hue = Math.random() * 360;
        }

        update() {
            this.x += this.speedX;
            if (this.x > canvas.width) {
                this.reset();
            }
        }

        draw() {
            const progress = this.x / canvas.width;
            ctx.fillStyle = `hsl(${this.hue}, ${progress * 100}%, 50%)`;
            ctx.fillRect(this.x, this.y, this.size, this.size);
        }
    }

    for (let i = 0; i < 200; i++) {
        particles.push(new Particle());
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        particles.forEach(particle => {
            particle.update();
            particle.draw();
        });
        
        requestAnimationFrame(animate);
    }

    animate();
});
'''
        with open('frontend/js/app.js', 'w') as f:
            f.write(content)

fix_app_js()
