import re

def main():
    content = ""
    with open('frontend/js/app.js', 'r') as f:
        content = f.read()

    # Remove the garbled output if it got appended
    garbled_idx = content.find('heredoc>')
    if garbled_idx != -1:
        content = content[:garbled_idx]
        
    js_code = """
// --- Pixel Canvas Background Animation ---
document.addEventListener("DOMContentLoaded", function() {
    const canvas = document.getElementById("pixel-canvas");
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    let particles = [];
    
    function resizeCanvas() {
        const parent = document.getElementById("app-workspace");
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
            this.x = Math.random() * canvas.width;
        }

        reset() {
            this.x = 0;
            this.y = Math.random() * canvas.height;
            this.size = Math.random() * 2 + 1; // 1px to 3px
            this.speedX = Math.random() * 1.5 + 0.5; // moving right
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
"""
    # Append the js_code only if it's not already there
    if "Pixel Canvas Background Animation" not in content:
        content += js_code

    with open('frontend/js/app.js', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    main()
