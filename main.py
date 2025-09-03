"""
ColorizeAI - Main Application Entry Point
"""

import sys
import os
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import gradio as gr
import numpy as np
from typing import List
import tempfile
import zipfile
import io
import cv2
from PIL import Image
from skimage.transform import resize
import json
import torch
import contextlib

# Import our organized modules
from colorizeai.core.models import get_models
from colorizeai.core.colorization import colorize_highres, colorize_highres_enhanced
# Cache removed: no caching helpers imported
from colorizeai.utils.metrics import compute_metrics
from colorizeai.features.temporal_consistency import TemporalConsistencyEngine

# Initialize components
temporal_engine = TemporalConsistencyEngine()

# ---------------- Performance Utilities ----------------
def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _get_device()

@contextlib.contextmanager
def _autocast(device):
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            yield
    elif device.type == "mps":
        with torch.autocast("mps", dtype=torch.float16):
            yield
    else:
        yield


def handler_single(input_img: np.ndarray, strength: float, gt_img: np.ndarray | None):
    """Basic single image handler"""
    if input_img is None:
        return None, None, "<span style='color:red'>Please upload an image.</span>"

    # Direct processing (cache removed)
    eccv_img, sig_img = colorize_highres(input_img, strength)

    slider_eccv = (input_img, (eccv_img * 255).astype(np.uint8))
    slider_sig = (input_img, (sig_img * 255).astype(np.uint8))

    metrics_html = "<b>No ground-truth supplied &ndash; metrics not computed.</b>"
    if gt_img is not None:
        # Resize GT if necessary
        if gt_img.shape[:2] != input_img.shape[:2]:
            gt_resized = resize(gt_img, input_img.shape[:2], preserve_range=True, anti_aliasing=True).astype(np.uint8)
        else:
            gt_resized = gt_img
        gt_float = gt_resized.astype(np.float64) / 255.0
        eccv_float = eccv_img
        sig_float = sig_img
        psnr_eccv, ssim_eccv = compute_metrics(gt_float, eccv_float)
        psnr_sig, ssim_sig = compute_metrics(gt_float, sig_float)
        metrics_html = f"""<table>
        <tr><th></th><th>PSNR</th><th>SSIM</th></tr>
        <tr><td>ECCV16</td><td>{psnr_eccv:.2f}</td><td>{ssim_eccv:.3f}</td></tr>
        <tr><td>SIGGRAPH17</td><td>{psnr_sig:.2f}</td><td>{ssim_sig:.3f}</td></tr>
        </table>"""

    return slider_eccv, slider_sig, metrics_html

def handler_single_enhanced(
    input_img: np.ndarray, 
    strength: float, 
    gt_img: np.ndarray | None,
    use_ensemble: bool,
    reference_img: np.ndarray | None,
    style_type: str,
    color_hints_json: str
):
    """Enhanced single image handler with all new features"""
    if input_img is None:
        return None, None, "<span style='color:red'>Please upload an image.</span>", None

    # Parse color hints from JSON string (if provided)
    color_hints = []
    if color_hints_json and color_hints_json.strip():
        try:
            color_hints = json.loads(color_hints_json)
        except:
            pass

    eccv_img, sig_img, metadata = colorize_highres_enhanced(
        input_img, strength, use_ensemble, reference_img, color_hints, style_type
    )

    slider_eccv = (input_img, (eccv_img * 255).astype(np.uint8))
    slider_sig = (input_img, (sig_img * 255).astype(np.uint8))

    # Generate enhanced metrics HTML
    metrics_html = "<b>No ground-truth supplied &ndash; metrics not computed.</b>"
    if gt_img is not None:
        if gt_img.shape[:2] != input_img.shape[:2]:
            gt_resized = resize(gt_img, input_img.shape[:2], preserve_range=True, anti_aliasing=True).astype(np.uint8)
        else:
            gt_resized = gt_img
        gt_float = gt_resized.astype(np.float64) / 255.0
        eccv_float = eccv_img
        sig_float = sig_img
        psnr_eccv, ssim_eccv = compute_metrics(gt_float, eccv_float)
        psnr_sig, ssim_sig = compute_metrics(gt_float, sig_float)
        
        metrics_html = f"""<table style="width:100%">
        <tr><th>Model</th><th>PSNR</th><th>SSIM</th></tr>
        <tr><td>ECCV16</td><td>{psnr_eccv:.2f}</td><td>{ssim_eccv:.3f}</td></tr>
        <tr><td>Enhanced</td><td>{psnr_sig:.2f}</td><td>{ssim_sig:.3f}</td></tr>
        </table>"""

    # Generate metadata display
    metadata_html = "<h4>Processing Information:</h4><ul>"
    for key, value in metadata.items():
        if key == 'ensemble_weights':
            metadata_html += f"<li><b>Model Weights:</b> ECCV16: {value['eccv16']:.2f}, SIGGRAPH17: {value['siggraph17']:.2f}</li>"
        elif key == 'image_characteristics':
            metadata_html += f"<li><b>Image Analysis:</b> Texture: {value.get('texture_complexity', 0):.2f}, Contrast: {value.get('contrast', 0):.2f}</li>"
        else:
            metadata_html += f"<li><b>{key.replace('_', ' ').title()}:</b> {value}</li>"
    metadata_html += "</ul>"

    return slider_eccv, slider_sig, metrics_html, metadata_html

def handler_batch(files: List[str] | None, strength: float, progress=gr.Progress()):
    """Batch processing handler"""
    if not files:
        return [], [], None, "<span style='color:red'>❌ No files uploaded. Please select images to process.</span>"

    progress(0, desc="Starting batch processing...")
    
    eccv_gallery, sig_gallery = [], []
    processed_count = 0
    total_files = len(files)

    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    
    try:
        with zipfile.ZipFile(tmp_zip, "w") as zf:
            for idx, file_path in enumerate(files):
                file_path = Path(file_path)
                
                # Update progress
                progress_pct = idx / total_files
                progress(progress_pct, desc=f"Processing {file_path.name} ({idx + 1}/{total_files})")
                
                try:
                    img = Image.open(file_path).convert("RGB")
                    img_np = np.array(img)
                    
                    # Resize large images for faster processing
                    h, w = img_np.shape[:2]
                    if max(h, w) > 1024:
                        scale = 1024 / max(h, w)
                        new_h, new_w = int(h * scale), int(w * scale)
                        img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    eccv_img, sig_img = colorize_highres(img_np, strength)

                    eccv_uint8 = (eccv_img * 255).astype(np.uint8)
                    sig_uint8 = (sig_img * 255).astype(np.uint8)

                    eccv_gallery.append(eccv_uint8)
                    sig_gallery.append(sig_uint8)

                    base = file_path.stem
                    eccv_name = f"{base}_eccv16.png"
                    sig_name = f"{base}_siggraph17.png"

                    # Save into ZIP
                    with io.BytesIO() as buff:
                        Image.fromarray(eccv_uint8).save(buff, format="PNG")
                        zf.writestr(eccv_name, buff.getvalue())
                    with io.BytesIO() as buff:
                        Image.fromarray(sig_uint8).save(buff, format="PNG")
                        zf.writestr(sig_name, buff.getvalue())
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Warning: Failed to process {file_path.name}: {e}")
                    continue

        tmp_zip.flush()
        
        progress(1.0, desc=f"Batch processing completed! Processed {processed_count}/{total_files} images.")
        
        # Generate status message
        if processed_count == total_files:
            status_msg = f"✅ <b>Success!</b> Processed all {processed_count} images successfully."
        elif processed_count > 0:
            status_msg = f"⚠️ <b>Partial Success:</b> Processed {processed_count} out of {total_files} images. Some files may have been skipped due to format issues."
        else:
            status_msg = "❌ <b>Failed:</b> No images could be processed. Please check file formats."
            
        return eccv_gallery, sig_gallery, tmp_zip.name, status_msg
        
    except Exception as e:
        progress(1.0, desc="Batch processing failed!")
        return [], [], None, f"❌ <b>Error:</b> Batch processing failed: {str(e)}"

def handler_video(video_file: str | None, strength: float, frame_skip: int = 1, resolution: str = "Original", custom_width: int = None, custom_height: int = None, fast_mode: bool = True, use_temporal_consistency: bool = False, style_type: str = 'none', progress=gr.Progress()):
    """Video processing pipeline with keyframe interpolation, optional temporal consistency & styles, and codec fallback.

    Features:
    - Keyframe sampling + lightweight interpolation instead of naive duplication.
    - Multi-codec fallback for broader compatibility.
    - Scene-change detection resets temporal state to avoid ghosting.
    - Disk space safety check before writing output.
    - Fast Mode auto-adjusts frame skipping + internal resolution for speed.
    """
    if video_file is None:
        return None

    # Sanitize frame_skip - default to 3 for better speed
    frame_skip = max(1, int(frame_skip) if frame_skip else 3)
    # Fast mode tweaks
    if fast_mode:
        frame_skip = max(frame_skip, 2)
        # Disable temporal for speed if user accidentally enabled
        if use_temporal_consistency:
            use_temporal_consistency = False
            gr.Info("Temporal consistency disabled in Fast Mode for speed.")

    # Cache removed: always process

    # Initialize temporal consistency engine if enhanced features are enabled
    if use_temporal_consistency:
        temporal_engine.reset()

    progress(0, desc="🎬 Starting video processing...")

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        gr.Warning("Failed to open video file. Please check the file format.")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        gr.Warning("Video appears to be empty or corrupted.")
        cap.release()
        return None

    # Automatic resolution downscaling for speed
    if fast_mode:
        # Force smaller resolution for speed
        if resolution == "Original" and (width > 1280 or height > 720):
            resolution = "720p"
        elif resolution == "1080p":
            resolution = "720p"

    # Determine output resolution
    if resolution == "Original":
        out_w, out_h = width, height
    elif resolution == "720p":
        out_w, out_h = 1280, 720
    elif resolution == "1080p":
        out_w, out_h = 1920, 1080
    elif resolution == "480p":  
        out_w, out_h = 854, 480
    elif resolution == "Custom" and custom_width and custom_height:
        out_w, out_h = int(custom_width), int(custom_height)
    else:
        out_w, out_h = width, height

    # Scale down for processing speed while maintaining aspect ratio
    if fast_mode and (out_w > 854 or out_h > 480):
        aspect_ratio = out_w / out_h
        if aspect_ratio > 1:
            processing_w, processing_h = 854, int(854 / aspect_ratio)
        else:
            processing_w, processing_h = int(480 * aspect_ratio), 480
    else:
        processing_w, processing_h = out_w, out_h

    # Ensure dimensions are even (required for some codecs)
    out_w = out_w if out_w % 2 == 0 else out_w - 1
    out_h = out_h if out_h % 2 == 0 else out_h - 1
    processing_w = processing_w if processing_w % 2 == 0 else processing_w - 1
    processing_h = processing_h if processing_h % 2 == 0 else processing_h - 1

    # Create cached output path with unique name
    # Output directory fallback (use temp dir within project cache/videos even without registry)
    cache_dir = Path(__file__).parent / "outputs" / "videos"
    cache_dir.mkdir(exist_ok=True)
    out_path = str(cache_dir / f"video_output.mp4")

    # Disk space safety (require at least 50MB free or 2x estimated size)
    try:
        statv = os.statvfs(str(cache_dir))
        free_bytes = statv.f_bavail * statv.f_frsize
        est_size = width * height * 3 * total_frames // (frame_skip if frame_skip else 1) // 4  # rough compressed estimate
        if free_bytes < max(50 * 1024 * 1024, est_size * 2):
            gr.Warning("Low disk space in output directory. Aborting video processing.")
            cap.release()
            return None
    except Exception:
        pass

    # Codec fallback list (quality preference order when not fast_mode)
    codec_candidates = ["mp4v", "avc1", "h264", "XVID"] if not fast_mode else ["mp4v", "avc1", "XVID"]
    writer = None
    chosen_codec = None
    for c in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*c)
        w = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
        if w.isOpened():
            writer = w
            chosen_codec = c
            break
    if writer is None:
        gr.Warning("Failed to initialize any video writer (codecs tried: mp4v, avc1, h264, XVID).")
        cap.release()
        return None

    processing_mode = "Enhanced" if use_temporal_consistency or style_type != 'none' else ("Fast" if fast_mode else "Quality")
    progress(0.1, desc=f"🔄 Processing video ({processing_mode} mode)...")

    # Precompute whether we process at final size directly
    single_resize = (processing_w == out_w and processing_h == out_h)

    # Preallocate output BGR buffer
    out_bgr_buffer = np.empty((out_h, out_w, 3), dtype=np.uint8)

    # Warm model once (to move weights to device / trigger compilation paths)
    try:
        dummy = np.zeros((processing_h, processing_w, 3), dtype=np.uint8)
        with torch.no_grad():
            with _autocast(DEVICE):
                colorize_highres(dummy, 1.0)
    except Exception:
        pass

    frame_idx = 0
    last_keyframe_rgb = None
    last_keyframe_idx = -1
    processed_frames = 0
    prev_gray_frame = None
    
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            # Update progress less frequently for speed
            if frame_idx % 60 == 0:  # Update every 60 frames instead of 30
                progress_pct = 0.1 + (frame_idx / total_frames) * 0.8
                progress(progress_pct, desc=f"Processing frame {frame_idx + 1}/{total_frames} (Skip: {frame_skip})")
            
            do_keyframe = (frame_idx % frame_skip == 0)
            frame_gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if use_temporal_consistency else None
            # Detect scene change (simple threshold on gray diff) when temporal enabled
            scene_change = False
            if use_temporal_consistency and prev_gray_frame is not None and frame_gray_full is not None:
                gray_diff = np.mean(np.abs(frame_gray_full.astype(float) - prev_gray_frame.astype(float))) / 255.0
                if gray_diff > 0.25:
                    scene_change = True
                    temporal_engine.reset()
            if use_temporal_consistency:
                prev_gray_frame = frame_gray_full

            if do_keyframe or scene_change:
                try:
                    if fast_mode and (frame_bgr.shape[1] != processing_w or frame_bgr.shape[0] != processing_h):
                        frame_bgr_small = cv2.resize(frame_bgr, (processing_w, processing_h), interpolation=cv2.INTER_AREA)
                    else:
                        frame_bgr_small = frame_bgr
                    frame_rgb_small = cv2.cvtColor(frame_bgr_small, cv2.COLOR_BGR2RGB)
                    with torch.no_grad():
                        with _autocast(DEVICE):
                            if use_temporal_consistency or style_type != 'none':
                                _, enhanced_frame, _ = colorize_highres_enhanced(
                                    frame_rgb_small, strength,
                                    use_ensemble=True,
                                    style_type=style_type
                                )
                                if use_temporal_consistency:
                                    frame_gray_small = cv2.cvtColor(frame_rgb_small, cv2.COLOR_RGB2GRAY)
                                    enhanced_frame = temporal_engine.apply_temporal_consistency(enhanced_frame, frame_gray_small)
                                key_rgb = (enhanced_frame * 255).astype(np.uint8)
                            else:
                                eccv_img, _ = colorize_highres(frame_rgb_small, strength)
                                key_rgb = (eccv_img * 255).astype(np.uint8)
                    if key_rgb.shape[:2] != (out_h, out_w):
                        key_rgb = cv2.resize(key_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                    last_keyframe_rgb = key_rgb.copy()
                    last_keyframe_idx = frame_idx
                    out_frame_rgb = key_rgb
                    processed_frames += 1
                except Exception as e:
                    print(f"Warning: Colorization failed for keyframe {frame_idx}: {e}")
                    out_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    if out_frame_rgb.shape[:2] != (out_h, out_w):
                        out_frame_rgb = cv2.resize(out_frame_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            else:
                # Interpolate between last keyframe and the future (cannot see future yet) -> hold with mild fade
                if last_keyframe_rgb is not None:
                    # Simple temporal fade-in weight
                    delta = frame_idx - last_keyframe_idx
                    weight = min(1.0, delta / frame_skip)
                    out_frame_rgb = last_keyframe_rgb.copy()
                    if weight < 1.0 and not fast_mode:
                        # Blend a little original luminance for variation
                        orig_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        if orig_rgb.shape[:2] != (out_h, out_w):
                            orig_rgb = cv2.resize(orig_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                        out_frame_rgb = cv2.addWeighted(out_frame_rgb, 0.9, orig_rgb, 0.1, 0)
                else:
                    out_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    if out_frame_rgb.shape[:2] != (out_h, out_w):
                        out_frame_rgb = cv2.resize(out_frame_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            
            # Convert and write (optimized)
            # Convert to BGR in preallocated buffer
            cv2.cvtColor(out_frame_rgb, cv2.COLOR_RGB2BGR, dst=out_bgr_buffer)
            writer.write(out_bgr_buffer)
            frame_idx += 1

    except Exception as e:
        gr.Warning(f"Error during video processing: {str(e)}")
        cap.release()
        writer.release()
        if os.path.exists(out_path):
            os.unlink(out_path)
        return None
    
    finally:
        cap.release()
        writer.release()

    progress(0.95, desc="� Finalizing video file...")
    
    # Check if output file was created successfully
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        gr.Warning("Failed to create output video file.")
        return None
    
    # Direct return (no caching layer)
    
    progress(1.0, desc=f"✅ Done! Processed {processed_frames}/{total_frames} keyframes ({processing_mode} • {chosen_codec})")
    
    return out_path

def build_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="🎨 ColorizeAI - Complete Image & Video Colorization Suite") as demo:
    
        gr.Markdown(
            """# 🎨 ColorizeAI

    High‑quality AI colorization for images & video – fast defaults, deep controls when you need them.

    **✨ Core Innovations**
    - 🧠 Smart Model Fusion (ECCV16 + SIGGRAPH17, region‑aware)
    - 🎯 Reference‑Guided Colorization (palette + tone transfer)
    - 🖌️ Interactive Color Hints (sparse RGB points)
    - ⏳ Temporal Consistency (optional flicker reduction)
    - � Cinematic / Artistic Style Presets

    **🧩 Toolkit**
    - � PSNR / SSIM evaluation (with ground truth)
    - � Batch processing (ZIP export)
    - 🎬 Video frame skipping + resolution control
    - �️ High‑res support (adaptive downscale in Fast Mode)
    - 🔍 Before / After sliders
    - �️ Strength blending control

    **⚡ Performance Philosophy**
    Fast Mode = frame skip + internal downscale. Enhanced Mode enables fusion, styles, temporal smoothing. No caching layer → every run is fresh and reproducible.

    Models included: ECCV16 (vibrant) + SIGGRAPH17 (natural). Fusion picks the best of both automatically.
    """
    )

        with gr.Tabs():
            # ----- Basic Single Image -----
            with gr.TabItem("🖼️ Basic Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        inp_img = gr.Image(type="numpy", label="📤 Upload Image", height=300)
                        gt_img = gr.Image(type="numpy", label="🎯 Ground-truth Image (optional)", height=300)
                    with gr.Column(scale=1):
                        strength_slider = gr.Slider(0, 1, value=1.0, step=0.1, label="🎨 Colorization Strength", 
                                                   info="Lower values blend with original grayscale")
                        run_btn = gr.Button("🚀 Colorize Image", variant="primary", size="lg")
                        
                        # Add some example images info
                        gr.Markdown("💡 **Basic Mode:** Fast processing with standard ECCV16 and SIGGRAPH17 models.")

                with gr.Row():
                    slider_eccv = gr.ImageSlider(label="📊 Original ↔ ECCV16 (Vibrant Colors)", height=400)
                    slider_sig = gr.ImageSlider(label="📊 Original ↔ SIGGRAPH17 (Realistic Colors)", height=400)
                
                metrics_html = gr.HTML()

                run_btn.click(
                    handler_single,
                    inputs=[inp_img, strength_slider, gt_img],
                    outputs=[slider_eccv, slider_sig, metrics_html],
                )

            # ----- Enhanced Single Image -----
            with gr.TabItem("🚀 Enhanced Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        inp_img_enh = gr.Image(type="numpy", label="📤 Upload B&W Image", height=300)
                        gt_img_enh = gr.Image(type="numpy", label="🎯 Ground-truth (optional)", height=200)
                        reference_img = gr.Image(type="numpy", label="🌈 Reference Image (optional)", height=200)
                        
                    with gr.Column(scale=1):
                        strength_slider_enh = gr.Slider(0, 1, value=1.0, step=0.1, 
                                                   label="🎨 Colorization Strength")
                        
                        use_ensemble = gr.Checkbox(value=True, label="🧠 Smart Model Fusion")
                        
                        style_type = gr.Dropdown([
                            'none', 'modern', 'vintage', 'cinematic', 'pastel', 'vibrant', 'cold',
                            'film_kodak', 'film_fuji', 'film_agfa',
                            'artistic_oil_painting', 'artistic_watercolor'
                        ], value='modern', label="🎭 Style Preset")
                        
                        color_hints_json = gr.Textbox(
                            label="🖌️ Color Hints (JSON)", 
                            placeholder='[{"x":100,"y":50,"r":255,"g":0,"b":0}]',
                            info="Add color hints as JSON: x,y coordinates with r,g,b values"
                        )
                        
                        run_enhanced_btn = gr.Button("🚀 Enhanced Colorization", variant="primary", size="lg")

                with gr.Row():
                    slider_eccv_enh = gr.ImageSlider(label="📊 Original ↔ ECCV16", height=400)
                    slider_sig_enh = gr.ImageSlider(label="📊 Original ↔ Enhanced Result", height=400)
                
                with gr.Row():
                    metrics_html_enh = gr.HTML()
                    metadata_html = gr.HTML()

                run_enhanced_btn.click(
                    handler_single_enhanced,
                    inputs=[inp_img_enh, strength_slider_enh, gt_img_enh, use_ensemble, reference_img, 
                           style_type, color_hints_json],
                    outputs=[slider_eccv_enh, slider_sig_enh, metrics_html_enh, metadata_html],
                )

            # ----- Batch Processing -----
            with gr.TabItem("📦 Batch Processing"):
                gr.Markdown("### Upload multiple images for batch colorization")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        files_input = gr.Files(label="📁 Upload Multiple Images", file_count="multiple", 
                                             file_types=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"])
                        
                        # Status display
                        batch_status = gr.HTML(value="<p>📋 <b>Ready:</b> Upload images to start batch processing.</p>")
                        
                    with gr.Column(scale=1):
                        strength_slider2 = gr.Slider(0, 1, value=1.0, step=0.1, label="🎨 Colorization Strength")
                        run_batch = gr.Button("⚡ Process Batch", variant="primary", size="lg")
                        
                        gr.Markdown("""
                        **📋 Quick Guide:**
                        1. **Upload**: Select multiple images (JPG, PNG, etc.)
                        2. **Adjust**: Set colorization strength
                        3. **Process**: Click the button and wait
                        4. **Download**: Get ZIP with both ECCV16 & SIGGRAPH17 results
                        
                        **💡 Tips:**
                        - Images larger than 1024px are auto-resized for speed
                        - Supports common formats: JPG, PNG, BMP, TIFF
                        - Processing time: ~2-5 seconds per image
                        """)

                with gr.Row():
                    gallery_eccv = gr.Gallery(label="🎨 ECCV16 Results (Vibrant Colors)", columns=4, height=400, show_label=True)
                    gallery_sig = gr.Gallery(label="🎨 SIGGRAPH17 Results (Realistic Colors)", columns=4, height=400, show_label=True)
                
                zip_out = gr.File(label="📥 Download ZIP Archive (Contains both model results)")

                run_batch.click(
                    handler_batch,
                    inputs=[files_input, strength_slider2],
                    outputs=[gallery_eccv, gallery_sig, zip_out, batch_status],
                )

            # ----- Video Colorization -----
            with gr.TabItem("🎬 Video Colorization"):
                gr.Markdown("### Transform black & white videos with basic and enhanced features")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        vid_input = gr.File(label="🎥 Upload Video (MP4)", file_types=[".mp4", ".avi", ".mov"])
                        strength_slider3 = gr.Slider(0, 1, value=1.0, step=0.1, label="🎨 Colorization Strength")
                        
                        # Fast mode toggle prominently displayed
                        fast_mode = gr.Checkbox(value=True, label="⚡ Fast Mode (Recommended)", 
                                              info="Enables speed optimizations: higher frame skip, lower resolution processing")
                        
                        # Enhanced features toggle
                        use_temporal = gr.Checkbox(value=False, label="⏳ Temporal Consistency (Enhanced, reduces flicker)")
                        
                        style_type_video = gr.Dropdown([
                            'none', 'modern', 'vintage', 'cinematic', 'pastel', 'vibrant',
                            'film_kodak', 'film_fuji'
                        ], value='none', label="🎭 Video Style (Enhanced feature)")
                        
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            frame_skip = gr.Slider(minimum=1, maximum=30, value=3, step=1, 
                                                 label="🔄 Frame Skip Interval", 
                                                 info="Process every Nth frame (higher = faster but less smooth)")
                            resolution = gr.Dropdown(["Original", "720p", "1080p", "480p", "Custom"], 
                                                    value="720p", label="📺 Output Resolution")
                            with gr.Row():
                                custom_width = gr.Number(value=None, minimum=1, label="📐 Custom Width (px)", visible=False)
                                custom_height = gr.Number(value=None, minimum=1, label="📐 Custom Height (px)", visible=False)
                        
                        run_vid = gr.Button("🎬 Process Video", variant="primary", size="lg")
                        
                        gr.Markdown("""
                        **🚀 Speed Guide**
                        - Fast Mode: Frame skip + internal downscale for big speedups
                        - Default Skip = 3; raise to 5–8 for long clips
                        - 720p is the best balance; 480p for drafts; Original only for short finals
                        - Temporal Consistency & Styles add overhead → enable for final render
                        - Output written directly (no caching layer)

                        **🧪 Tips**
                        - Prototype fast, then refine with lower skip
                        - Temporal helps most on stable scenes
                        - Subtle styles preserve detail; extreme styles trade texture for mood
                        - Lower strength can give a more photographic result
                        """)
                    
                    with gr.Column(scale=1):
                        vid_out = gr.Video(label="🎥 Colorized Video Result", height=400)

                def show_custom_fields(res):
                    return {
                        custom_width: gr.update(visible=(res=="Custom")), 
                        custom_height: gr.update(visible=(res=="Custom"))
                    }
                resolution.change(show_custom_fields, inputs=resolution, outputs=[custom_width, custom_height])

                run_vid.click(
                    handler_video,
                    inputs=[vid_input, strength_slider3, frame_skip, resolution, custom_width, custom_height, fast_mode, use_temporal, style_type_video],
                    outputs=vid_out,
                )
        
    # Enhanced Footer
    gr.Markdown("""
    ---
    ### 🧠 Feature Deep Dive & Workflow

    **Fusion** – Region‑aware blend of ECCV16 (vibrant) + SIGGRAPH17 (natural).
    **Reference** – Extracts palette & tone; best with similar lighting.
    **Hints** – Sparse, high‑confidence RGB points propagate better than many.
    **Temporal** – Optical flow chroma stabilization (use for finals to cut flicker).
    **Styles** – Film & creative looks; apply subtly to retain texture.

    **Suggested Flow**
    1. Draft: Fast Mode (skip=3, 720p)
    2. Refine: Adjust strength, add hints or reference
    3. Stylize: Add style preset (optional)
    4. Final: Lower skip, enable temporal if needed

    Transparent processing (no caching layer) → reproducible results every run.
    """)

    return demo

if __name__ == "__main__":
    # Initialize models on startup
    get_models()
    
    # Launch the interface
    iface = build_interface()
    iface.launch()
 