const API_URL = window.location.origin + '/api';

// --- Scroll Reveal Hero Effect ---
document.addEventListener('DOMContentLoaded', () => {
    const scrollWrapper = document.getElementById('scroll-wrapper');
    const heroColor = document.getElementById('hero-color');

    if (scrollWrapper && heroColor) {
        window.addEventListener('scroll', () => {
            const rect = scrollWrapper.getBoundingClientRect();
            const vh = window.innerHeight;
            
            // Total scrollable area inside the wrapper (wrapper height - 100vh)
            const scrollDistance = rect.height - vh;
            
            // Distance scrolled past the top of the wrapper
            const scrolled = -rect.top;

            let percentage = (scrolled / scrollDistance) * 100;
            if (percentage < 0) percentage = 0;
            if (percentage > 100) percentage = 100;

            heroColor.style.setProperty('--reveal-progress', `${percentage}%`);
        });
    }
});

// Sliders UI sync
document.getElementById('singleStrength').addEventListener('input', e => {
    document.getElementById('singleStrengthVal').innerText = e.target.value;
});
document.getElementById('batchStrength').addEventListener('input', e => {
    document.getElementById('batchStrengthVal').innerText = e.target.value;
});
document.getElementById('metricsStrength').addEventListener('input', e => {
    document.getElementById('metricsStrengthVal').innerText = e.target.value;
});

// Helper for Base64 image
function getImgSrc(dataUrlOrB64) {
    if (!dataUrlOrB64) return "";
    if (dataUrlOrB64.startsWith('data:')) {
        return dataUrlOrB64;
    }
    return `data:image/jpeg;base64,${dataUrlOrB64}`;
}


// -------------------------------------------------------------
// TAB 1: SINGLE IMAGE COLORIZE
// -------------------------------------------------------------
document.getElementById('singleForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('singleInputImg').files[0];
    const refInput = document.getElementById('singleRefImg').files[0];
    const style = document.getElementById('singleStyle').value;
    const strength = document.getElementById('singleStrength').value;
    const useDDColor = document.getElementById('singleUseDDColor').checked;
    const useEnsemble = document.getElementById('singleUseEnsemble').checked;

    if (!fileInput) return;

    // UI State
    document.getElementById('singleStatus').style.display = 'none';
    document.getElementById('singleSpinner').style.display = 'flex';
    document.getElementById('singleResults').style.display = 'none';
    document.getElementById('singleSubmitBtn').disabled = true;

    try {
        const formData = new FormData();
        formData.append('input_img', fileInput);
        if (refInput) {
            formData.append('reference_img', refInput);
        }
        formData.append('strength', strength);
        formData.append('use_ddcolor', useDDColor);
        formData.append('use_ensemble', useEnsemble);
        formData.append('style_type', style);
        
        const res = await fetch(`${API_URL}/colorize/enhanced`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();

        if (data.error) throw new Error(data.error);

        // Display results
        const primarySrc = getImgSrc(data.primary_image);
        const eccvSrc = getImgSrc(data.eccv_image);

        document.getElementById('singlePrimaryImg').src = primarySrc;
        document.getElementById('singlePrimaryDownload').href = primarySrc;

        document.getElementById('singleEccvImg').src = eccvSrc;
        document.getElementById('singleEccvDownload').href = eccvSrc;

        document.getElementById('singleResults').style.display = 'flex';

    } catch (err) {
        alert("Error executing colorization: " + err.message);
        console.error(err);
        document.getElementById('singleStatus').style.display = 'block';
        document.getElementById('singleStatus').innerText = 'Colorization Failed. See console.';
    } finally {
        document.getElementById('singleSpinner').style.display = 'none';
        document.getElementById('singleSubmitBtn').disabled = false;
    }
});

// -------------------------------------------------------------
// TAB 2: BATCH PROCESSING
// -------------------------------------------------------------
document.getElementById('batchForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInputs = document.getElementById('batchInputImgs').files;
    const strength = document.getElementById('batchStrength').value;
    const useDDColor = true;

    if (fileInputs.length === 0) return;

    // UI state
    document.getElementById('batchStatus').style.display = 'none';
    document.getElementById('batchSpinner').style.display = 'flex';
    document.getElementById('batchResults').style.display = 'none';
    document.getElementById('batchSubmitBtn').disabled = true;

    try {
        const formData = new FormData();
        for (let i = 0; i < fileInputs.length; i++) {
            formData.append('files', fileInputs[i]);
        }
        formData.append('strength', strength);
        formData.append('use_ddcolor', useDDColor);

        const res = await fetch(`${API_URL}/colorize/batch`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const errBody = await res.text();
            throw new Error(`Server returned ${res.status}: ${errBody}`);
        }

        // Return type is application/zip (blob)
        const blob = await res.blob();
        if (blob.type === "application/json") {
            const txt = await blob.text();
            throw new Error(JSON.parse(txt).error || "Error from server");
        }

        const url = URL.createObjectURL(blob);
        const downloadBtn = document.getElementById('batchDownload');
        downloadBtn.href = url;
        downloadBtn.download = "batch_colorized.zip";
        
        document.getElementById('batchResults').style.display = 'block';

    } catch (err) {
        alert("Error executing batch colorization: " + err.message);
        console.error(err);
        document.getElementById('batchStatus').style.display = 'block';
        document.getElementById('batchStatus').innerText = 'Batch Processing Failed. See console.';
    } finally {
        document.getElementById('batchSpinner').style.display = 'none';
        document.getElementById('batchSubmitBtn').disabled = false;
    }
});

// -------------------------------------------------------------
// TAB 4: VIDEO PROCESSING
// -------------------------------------------------------------
document.getElementById('videoForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('videoInput').files[0];
    const useTemporal = document.getElementById('videoTemporal').checked;
    const skipFrames = document.getElementById('videoSkipFrames').value;
    const strength = 1.0; 
    const useDDColor = true;

    if (!fileInput) return;

    document.getElementById('videoStatus').style.display = 'none';
    document.getElementById('videoSpinner').style.display = 'flex';
    document.getElementById('videoResults').style.display = 'none';
    document.getElementById('videoSubmitBtn').disabled = true;

    try {
        const formData = new FormData();
        formData.append('video_file', fileInput);
        formData.append('use_temporal', useTemporal);
        formData.append('skip_frames', skipFrames);
        formData.append('strength', strength);
        formData.append('use_ddcolor', useDDColor);

        const res = await fetch(`${API_URL}/colorize/video`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const errBody = await res.text();
            throw new Error(`Server returned ${res.status}: ${errBody}`);
        }

        const blob = await res.blob();
        if (blob.type === "application/json") {
            const txt = await blob.text();
            try {
                 throw new Error(JSON.parse(txt).error || "Error from server");
            } catch (jsonErr) {
                 throw new Error(txt);
            }
        }

        const url = URL.createObjectURL(blob);
        
        const vidPlayer = document.getElementById('videoPlayer');
        vidPlayer.src = url;
        vidPlayer.load();

        const downloadBtn = document.getElementById('videoDownload');
        downloadBtn.href = url;
        downloadBtn.download = "colorized_video.mp4";
        
        document.getElementById('videoResults').style.display = 'block';

    } catch (err) {
        alert("Error executing video colorization: " + err.message);
        console.error(err);
        document.getElementById('videoStatus').style.display = 'block';
        document.getElementById('videoStatus').innerText = 'Video Processing Failed. See console.';
    } finally {
        document.getElementById('videoSpinner').style.display = 'none';
        document.getElementById('videoSubmitBtn').disabled = false;
    }
});

// -------------------------------------------------------------
// TAB 5: METRICS LAB
// -------------------------------------------------------------
document.getElementById('metricsForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const bwInput = document.getElementById('metricsInputImg').files[0];
    const gtInput = document.getElementById('metricsGtImg').files[0];
    const strength = document.getElementById('metricsStrength').value;

    if (!bwInput || !gtInput) return;

    document.getElementById('metricsStatus').style.display = 'none';
    document.getElementById('metricsSpinner').style.display = 'flex';
    document.getElementById('metricsResults').style.display = 'none';
    document.getElementById('metricsSubmitBtn').disabled = true;

    try {
        const formData = new FormData();
        formData.append('input_img', bwInput);
        formData.append('gt_img', gtInput);
        formData.append('strength', strength);
        
        // Append advanced metric flags
        formData.append('lpips_flag', document.getElementById('evalLPIPS').checked);
        formData.append('colorfulness_flag', document.getElementById('evalColorfulness').checked);
        
        const res = await fetch(`${API_URL}/evaluate`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();

        if (data.error) throw new Error(data.error);

        // Display results table
        document.getElementById('metricsTableContainer').innerHTML = data.table_html;

        // Display best Image
        const bestSrc = getImgSrc(data.primary_image);
        document.getElementById('metricsBestImg').src = bestSrc;

        document.getElementById('metricsResults').style.display = 'block';

    } catch (err) {
        alert("Error running evaluation: " + err.message);
        console.error(err);
        document.getElementById('metricsStatus').style.display = 'block';
        document.getElementById('metricsStatus').innerText = 'Evaluation Failed. See console.';
    } finally {
        document.getElementById('metricsSpinner').style.display = 'none';
        document.getElementById('metricsSubmitBtn').disabled = false;
    }
});

// --- Pixel Canvas Background Animation ---
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
            this.size = Math.random() * 4 + 2; // medium pixels 2px to 6px
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
