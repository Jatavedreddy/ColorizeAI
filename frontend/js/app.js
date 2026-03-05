const API_URL = window.location.origin + '/api';

// Sliders UI sync
document.getElementById('basicStrength').addEventListener('input', e => {
    document.getElementById('basicStrengthVal').innerText = e.target.value;
});
document.getElementById('enhancedStrength').addEventListener('input', e => {
    document.getElementById('enhancedStrengthVal').innerText = e.target.value;
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
// TAB 1: BASIC COLORIZE
// -------------------------------------------------------------
document.getElementById('basicForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('basicInputImg').files[0];
    const strength = document.getElementById('basicStrength').value;
    const useDDColor = document.getElementById('basicUseDDColor').checked;

    if (!fileInput) return;

    // UI State
    document.getElementById('basicStatus').style.display = 'none';
    document.getElementById('basicSpinner').style.display = 'inline-block';
    document.getElementById('basicResults').style.display = 'none';
    document.getElementById('basicSubmitBtn').disabled = true;

    try {
        const formData = new FormData();
        formData.append('input_img', fileInput);
        formData.append('strength', strength);
        formData.append('use_ddcolor', useDDColor);

        const res = await fetch(`${API_URL}/colorize/basic`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();

        if (data.error) throw new Error(data.error);

        // Display results
        const primarySrc = getImgSrc(data.primary_image);
        const eccvSrc = getImgSrc(data.eccv_image);

        document.getElementById('basicPrimaryImg').src = primarySrc;
        document.getElementById('basicPrimaryDownload').href = primarySrc;

        document.getElementById('basicEccvImg').src = eccvSrc;
        document.getElementById('basicEccvDownload').href = eccvSrc;

        document.getElementById('basicResults').style.display = 'flex';

    } catch (err) {
        alert("Error executing colorization: " + err.message);
        console.error(err);
        document.getElementById('basicStatus').style.display = 'block';
        document.getElementById('basicStatus').innerText = 'Colorization Failed. See console.';
    } finally {
        document.getElementById('basicSpinner').style.display = 'none';
        document.getElementById('basicSubmitBtn').disabled = false;
    }
});


// -------------------------------------------------------------
// TAB 2: ADVANCED/ENHANCED COLORIZE
// -------------------------------------------------------------
document.getElementById('enhancedForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('enhancedInputImg').files[0];
    const refInput = document.getElementById('enhancedRefImg').files[0];
    const style = document.getElementById('enhancedStyle').value;
    const strength = document.getElementById('enhancedStrength').value;
    const useDDColor = document.getElementById('enhancedUseDDColor').checked;
    const useEnsemble = document.getElementById('enhancedUseEnsemble').checked;

    if (!fileInput) return;

    // UI State
    document.getElementById('enhancedStatus').style.display = 'none';
    document.getElementById('enhancedSpinner').style.display = 'inline-block';
    document.getElementById('enhancedResults').style.display = 'none';
    document.getElementById('enhancedSubmitBtn').disabled = true;

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

        document.getElementById('enhancedPrimaryImg').src = primarySrc;
        document.getElementById('enhancedPrimaryDownload').href = primarySrc;

        document.getElementById('enhancedEccvImg').src = eccvSrc;
        document.getElementById('enhancedEccvDownload').href = eccvSrc;

        document.getElementById('enhancedResults').style.display = 'flex';

    } catch (err) {
        alert("Error executing enhanced colorization: " + err.message);
        console.error(err);
        document.getElementById('enhancedStatus').style.display = 'block';
        document.getElementById('enhancedStatus').innerText = 'Colorization Failed. See console.';
    } finally {
        document.getElementById('enhancedSpinner').style.display = 'none';
        document.getElementById('enhancedSubmitBtn').disabled = false;
    }
});


// -------------------------------------------------------------
// TAB 3: BATCH PROCESSING
// -------------------------------------------------------------
document.getElementById('batchForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInputs = document.getElementById('batchInputImgs').files;
    const strength = document.getElementById('batchStrength').value;
    const useDDColor = true;

    if (fileInputs.length === 0) return;

    // UI state
    document.getElementById('batchStatus').style.display = 'none';
    document.getElementById('batchSpinner').style.display = 'inline-block';
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
    document.getElementById('videoSpinner').style.display = 'inline-block';
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
    document.getElementById('metricsSpinner').style.display = 'inline-block';
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
