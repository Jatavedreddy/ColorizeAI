import os
import io
import json
import base64
import tempfile
import zipfile
import traceback
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import time

# Core Colorization
from colorizeai.core.colorization import colorize_highres, colorize_highres_enhanced

# Features
from colorizeai.features.temporal_consistency import TemporalConsistencyEngine

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
CORS(app)

def read_image_from_request(req, field_name):
    if field_name not in req.files:
        return None
    file = req.files[field_name]
    if file.filename == '':
        return None
    try:
        img = Image.open(file.stream).convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Error reading {field_name}: {e}")
        return None

def to_uint8(img_float):
    if img_float is None:
        return None
    if img_float.dtype == np.uint8:
        return img_float
    return np.clip(img_float * 255.0, 0, 255).astype(np.uint8)

def numpy_to_base64(img_np):
    if img_np is None:
        return None
    try:
        img_np = to_uint8(img_np)
        img = Image.fromarray(img_np)
        # Downscale for UI preview to avoid massive base64 payloads crashing network requests
        img.thumbnail((800, 800), Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=80)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

@app.route('/')
def index():
    return send_from_directory(app.template_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/colorize/basic', methods=['POST'])
def colorize_basic():
    try:
        strength = float(request.form.get('strength', 1.0))
        use_ddcolor = request.form.get('use_ddcolor', 'true') == 'true'
        
        input_img = read_image_from_request(request, 'input_img')
        if input_img is None:
            return jsonify({'error': 'No input image provided'}), 400
            
        eccv_rgb, primary_rgb = colorize_highres(input_img, strength=strength, use_ddcolor=use_ddcolor)
        
        return jsonify({
            'success': True,
            'eccv_image': numpy_to_base64(eccv_rgb),
            'primary_image': numpy_to_base64(primary_rgb)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/colorize/enhanced', methods=['POST'])
def colorize_enhanced():
    try:
        strength = float(request.form.get('strength', 1.0))
        use_ddcolor = request.form.get('use_ddcolor', 'true') == 'true'
        use_ensemble = request.form.get('use_ensemble', 'false') == 'true'
        style_type = request.form.get('style_type', 'none')
        if style_type == 'none': style_type = None

        input_img = read_image_from_request(request, 'input_img')
        if input_img is None:
            return jsonify({'error': 'No input image provided'}), 400
            
        reference_img = read_image_from_request(request, 'reference_img')

        color_hints_json = request.form.get('color_hints_json', '')
        pts = None
        if color_hints_json and color_hints_json.strip():
            try:
                pts = json.loads(color_hints_json)
            except Exception as e:
                print(f"Warning: Invalid JSON for color hints: {e}")

        # Note: colorize_highres_enhanced returns exactly three objects!
        eccv_rgb, primary_rgb, metadata = colorize_highres_enhanced(
            input_img,
            strength=strength,
            use_ddcolor=use_ddcolor,
            use_ensemble=use_ensemble,
            reference_image=reference_img,
            style_type=style_type,
            color_points=pts
        )
        
        return jsonify({
            'success': True,
            'eccv_image': numpy_to_base64(eccv_rgb),
            'primary_image': numpy_to_base64(primary_rgb),
            'metadata': metadata
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/colorize/batch', methods=['POST'])
def colorize_batch():
    try:
        strength = float(request.form.get('strength', 1.0))
        use_ddcolor = request.form.get('use_ddcolor', 'true') == 'true'
        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files provided'}), 400

        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in files:
                try:
                    img = Image.open(file.stream).convert('RGB')
                    img_np = np.array(img)
                    _, primary_rgb = colorize_highres(img_np, strength=strength, use_ddcolor=use_ddcolor)
                    
                    pil_img = Image.fromarray(to_uint8(primary_rgb))
                    img_byte_arr = io.BytesIO()
                    pil_img.save(img_byte_arr, format='JPEG')
                    
                    name = secure_filename(file.filename)
                    if not name: name = "image.jpg"
                    zf.writestr(f"colorized_{name}", img_byte_arr.getvalue())
                except Exception as e:
                    print(f"Skipping file {file.filename}: {e}")

        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='batch_colorized.zip'
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/colorize/video', methods=['POST'])
def colorize_video():
    try:
        if 'video_file' not in request.files:
            return jsonify({'error': 'No video provided'}), 400
            
        video_file = request.files['video_file']
        strength = float(request.form.get('strength', 1.0))
        use_ddcolor = request.form.get('use_ddcolor', 'true') == 'true'
        use_temporal = request.form.get('use_temporal', 'true') == 'true'
        skip_frames = int(request.form.get('skip_frames', 1))
        
        temp_dir = tempfile.mkdtemp()
        in_path = os.path.join(temp_dir, 'input.mp4')
        out_path = os.path.join(temp_dir, 'output.mp4')
        video_file.save(in_path)
        
        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0.0 or np.isnan(fps): fps = 30.0
        width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Aggressive downscaling for video to prevent localtunnel 504 timeouts
        max_video_dim = 480 
        if max(width_orig, height_orig) > max_video_dim:
            scale = max_video_dim / max(width_orig, height_orig)
            width = int(width_orig * scale)
            height = int(height_orig * scale)
        else:
            width, height = width_orig, height_orig
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, int(fps), (width, height))
        
        temporal_engine = TemporalConsistencyEngine() if use_temporal else None
        frame_idx = 0
        last_colored_lab = None
        
        from skimage.color import rgb2lab, lab2rgb

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Resize frame to target dimensions
            if width != width_orig or height != height_orig:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if frame_idx % skip_frames == 0 or last_colored_lab is None:
                # Fully colorize frame
                _, colored = colorize_highres(frame_rgb, strength, use_ddcolor=use_ddcolor)
                
                # Cache the ab channels via Lab space for skipped frames
                colored_uint8 = to_uint8(colored)
                last_colored_lab = cv2.cvtColor(colored_uint8, cv2.COLOR_RGB2LAB)
            else:
                # Fast path: Mix current grayscale with previous frame's color
                curr_lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
                # Take L from current frame, A and B from previous frame
                curr_lab[:,:,1:] = last_colored_lab[:,:,1:]
                colored_uint8 = cv2.cvtColor(curr_lab, cv2.COLOR_LAB2RGB)
                colored = colored_uint8.astype(np.float64) / 255.0

            if use_temporal and temporal_engine:
                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                colored = temporal_engine.apply_temporal_consistency(colored, gray)
                # Update cached lab if temporal consistency changes it significantly!
                if skip_frames > 1:
                     colored_uint8 = to_uint8(colored)
                     last_colored_lab = cv2.cvtColor(colored_uint8, cv2.COLOR_RGB2LAB)

            colored_uint8 = to_uint8(colored)
            colored_bgr = cv2.cvtColor(colored_uint8, cv2.COLOR_RGB2BGR)
            out.write(colored_bgr)
            
            frame_idx += 1
            if frame_idx > 300: break # soft limit for web
            
        cap.release()
        out.release()
        
        return send_file(out_path, mimetype='video/mp4', as_attachment=True, download_name='colorized_video.mp4')
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate', methods=['POST'])
def handle_evaluate():
    """Evaluation Lab for computing and ranking standard metrics."""
    try:
        from colorizeai.utils.metrics import (
            compute_metrics, compute_lpips, colorfulness_index
        )
        input_img_uint8 = read_image_from_request(request, 'input_img')
        gt_img_uint8 = read_image_from_request(request, 'gt_img')
        strength = float(request.form.get('strength', 1.0))
        
        # Flags
        flag_lpips = request.form.get('lpips_flag', 'true') == 'true'
        flag_colorfulness = request.form.get('colorfulness_flag', 'true') == 'true'

        if input_img_uint8 is None or gt_img_uint8 is None:
            return jsonify({'error': 'Both Input and Ground Truth images are required.'}), 400

        # Downscale for web evaluation to prevent proxy timeouts
        max_dim = 800
        h, w = input_img_uint8.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            import cv2
            input_img_uint8 = cv2.resize(input_img_uint8, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        import skimage.transform
        if gt_img_uint8.shape[:2] != input_img_uint8.shape[:2]:
            gt_img_uint8 = skimage.transform.resize(
                gt_img_uint8, 
                input_img_uint8.shape[:2], 
                preserve_range=True, 
                anti_aliasing=True
            ).astype(np.uint8)
            
        gt_float = gt_img_uint8.astype(np.float64) / 255.0
        
        results = []
        
        def collect_metrics(method_name, pred_img, t_start):
            psnr_val, ssim_val = compute_metrics(gt_float, pred_img)
            
            row = {
                "method": method_name,
                "psnr": psnr_val,
                "ssim": ssim_val,
            }
            if flag_lpips:
                row["lpips"] = compute_lpips(gt_float, pred_img)
            if flag_colorfulness:
                row["colorfulness"] = colorfulness_index(pred_img)
            
            row["time_s"] = time.time() - t_start
            
            results.append(row)

        images = {}

        # 1. ECCV16 & 2. SIGGRAPH17
        t0 = time.time()
        eccv_img, sig_img = colorize_highres(input_img_uint8, strength, use_ddcolor=False)
        collect_metrics("ECCV16", eccv_img, t0)
        images["ECCV16"] = eccv_img
        
        t0 = time.time()
        collect_metrics("SIGGRAPH17", sig_img, t0)
        images["SIGGRAPH17"] = sig_img
        
        # 3. DDColor
        t0 = time.time()
        _, ddcolor_img = colorize_highres(input_img_uint8, strength, use_ddcolor=True)
        collect_metrics("DDColor", ddcolor_img, t0)
        images["DDColor"] = ddcolor_img
        
        # 4. DDColor + Adaptive Fusion
        t0 = time.time()
        _, fusion_img, _ = colorize_highres_enhanced(
            input_img_uint8, strength, use_ensemble=True, 
            use_ddcolor=True
        )
        collect_metrics("DDColor + Fusion", fusion_img, t0)
        images["DDColor + Fusion"] = fusion_img
        
        # 5. DDColor + Modern
        t0 = time.time()
        _, modern_img, _ = colorize_highres_enhanced(
            input_img_uint8, strength, style_type='modern', 
            use_ddcolor=True
        )
        collect_metrics("DDColor + Modern", modern_img, t0)
        images["DDColor + Modern"] = modern_img
        
        # 6. DDColor + Vintage
        t0 = time.time()
        _, vintage_img, _ = colorize_highres_enhanced(
            input_img_uint8, strength, style_type='vintage', 
            use_ddcolor=True
        )
        collect_metrics("DDColor + Vintage", vintage_img, t0)
        images["DDColor + Vintage"] = vintage_img
        
        # Sort by PSNR
        results_sorted = sorted(results, key=lambda x: x["psnr"], reverse=True)
        
        # Build Table HTML
        table_html = "<table class='table table-bordered table-striped mt-3 align-middle text-sm'>"
        table_html += "<thead class='table-dark'><tr>"
        table_html += "<th>Rank</th><th>Method</th><th>PSNR &uarr;</th><th>SSIM &uarr;</th>"
        if flag_lpips: table_html += "<th>LPIPS &darr;</th>"
        if flag_colorfulness: table_html += "<th>Color &uarr;</th>"
        table_html += "<th>Time &darr;</th></tr></thead><tbody>"
        
        best_img = None
        for rank, row in enumerate(results_sorted, 1):
            table_html += f"<tr><td><strong>#{rank}</strong></td><td>{row['method']}</td>"
            table_html += f"<td>{row['psnr']:.2f}</td><td>{row['ssim']:.4f}</td>"
            if flag_lpips: 
                v = row.get('lpips')
                table_html += f"<td>{v:.4f}</td>" if v is not None else "<td>N/A</td>"
            if flag_colorfulness: 
                v = row.get('colorfulness')
                table_html += f"<td>{v:.2f}</td>" if v is not None else "<td>N/A</td>"
            table_html += f"<td>{row['time_s']:.2f}s</td></tr>"
            
            if rank == 1:
                best_img = images.get(row['method'])

        table_html += "</tbody></table>"
        
        table_html += (
            "<div class='alert alert-info mt-3 mb-0' style='font-size: 0.9em;'>"
            "<strong>Note:</strong> Quantitative metrics do not always definitively rank the 'best' model "
            "as human perception of color is subjective. For certain specific images, legacy models like SIGGRAPH17 "
            "may visually outperform DDColor. However, in general across large datasets, <strong>DDColor + Adaptive Fusion</strong> "
            "has proven to perform the best."
            "</div>"
        )
        
        return jsonify({
            'success': True,
            'table_html': table_html,
            'primary_image': numpy_to_base64(best_img)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
