"""
ColorizeAI - Simplified Main Application (Fixed for Gradio 5.x)
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import gradio as gr
import numpy as np
from PIL import Image
import cv2

# Import our modules
from colorizeai.core.colorization import colorize_highres, colorize_highres_enhanced
from colorizeai.core.ddcolor_model import is_ddcolor_available
from colorizeai.utils.metrics import compute_metrics
from skimage.transform import resize
import json

print("Loading ColorizeAI...")
print(f"DDColor available: {is_ddcolor_available()}")

def safe_convert_to_numpy(img):
    """Safely convert any image format to numpy array"""
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, Image.Image):
        return np.array(img)
    if isinstance(img, str):  # File path
        return np.array(Image.open(img).convert("RGB"))
    return None

def handler_basic(input_img, strength, gt_img=None):
    """Basic colorization handler"""
    try:
        # Convert inputs
        input_img = safe_convert_to_numpy(input_img)
        if input_img is None:
            return None, None, "❌ Please upload an image."
        
        gt_img = safe_convert_to_numpy(gt_img)
        
        print(f"Processing image: {input_img.shape}")
        
        # Colorize
        eccv_img, primary_img = colorize_highres(input_img, strength, use_ddcolor=True)
        
        # Convert to uint8 for display
        eccv_uint8 = (eccv_img * 255).astype(np.uint8)
        primary_uint8 = (primary_img * 255).astype(np.uint8)
        
        # Create sliders (input, output) tuples
        slider_eccv = (input_img, eccv_uint8)
        slider_primary = (input_img, primary_uint8)
        
        # Compute metrics if ground truth provided
        if gt_img is not None:
            if gt_img.shape[:2] != input_img.shape[:2]:
                gt_img = resize(gt_img, input_img.shape[:2], preserve_range=True, anti_aliasing=True).astype(np.uint8)
            gt_float = gt_img.astype(np.float64) / 255.0
            psnr_eccv, ssim_eccv = compute_metrics(gt_float, eccv_img)
            psnr_primary, ssim_primary = compute_metrics(gt_float, primary_img)
            
            model_name = "DDColor" if is_ddcolor_available() else "SIGGRAPH17"
            metrics = f"""
            <table style='width:100%; border-collapse: collapse;'>
            <tr><th style='border:1px solid #ddd; padding:8px;'>Model</th>
                <th style='border:1px solid #ddd; padding:8px;'>PSNR</th>
                <th style='border:1px solid #ddd; padding:8px;'>SSIM</th></tr>
            <tr><td style='border:1px solid #ddd; padding:8px;'>ECCV16</td>
                <td style='border:1px solid #ddd; padding:8px;'>{psnr_eccv:.2f}</td>
                <td style='border:1px solid #ddd; padding:8px;'>{ssim_eccv:.3f}</td></tr>
            <tr><td style='border:1px solid #ddd; padding:8px;'>{model_name}</td>
                <td style='border:1px solid #ddd; padding:8px;'>{psnr_primary:.2f}</td>
                <td style='border:1px solid #ddd; padding:8px;'>{ssim_primary:.3f}</td></tr>
            </table>
            """
        else:
            metrics = "<p>ℹ️ <b>No ground truth provided</b> - metrics not computed.</p>"
        
        return slider_eccv, slider_primary, metrics
        
    except Exception as e:
        error_msg = f"❌ <b>Error:</b> {str(e)}"
        print(f"Error in handler_basic: {e}")
        import traceback
        traceback.print_exc()
        return None, None, error_msg

def handler_enhanced(input_img, strength, gt_img, use_ensemble, reference_img, style_type, color_hints_json):
    """Enhanced colorization handler"""
    try:
        # Convert inputs
        input_img = safe_convert_to_numpy(input_img)
        if input_img is None:
            return None, None, "❌ Please upload an image.", ""
        
        gt_img = safe_convert_to_numpy(gt_img)
        reference_img = safe_convert_to_numpy(reference_img)
        
        # Parse color hints
        color_hints = []
        if color_hints_json and color_hints_json.strip():
            try:
                color_hints = json.loads(color_hints_json)
            except:
                pass
        
        print(f"Enhanced processing: ensemble={use_ensemble}, style={style_type}, hints={len(color_hints)}")
        
        # Colorize with features
        eccv_img, enhanced_img, metadata = colorize_highres_enhanced(
            input_img, strength, use_ensemble, reference_img, color_hints, style_type, use_ddcolor=True
        )
        
        # Convert to uint8
        eccv_uint8 = (eccv_img * 255).astype(np.uint8)
        enhanced_uint8 = (enhanced_img * 255).astype(np.uint8)
        
        # Create sliders
        slider_eccv = (input_img, eccv_uint8)
        slider_enhanced = (input_img, enhanced_uint8)
        
        # Metrics
        if gt_img is not None:
            if gt_img.shape[:2] != input_img.shape[:2]:
                gt_img = resize(gt_img, input_img.shape[:2], preserve_range=True, anti_aliasing=True).astype(np.uint8)
            gt_float = gt_img.astype(np.float64) / 255.0
            psnr_eccv, ssim_eccv = compute_metrics(gt_float, eccv_img)
            psnr_enhanced, ssim_enhanced = compute_metrics(gt_float, enhanced_img)
            
            model_name = "DDColor Enhanced" if metadata.get('ddcolor_used', False) else "SIGGRAPH17 Enhanced"
            metrics = f"""
            <table style='width:100%; border-collapse: collapse;'>
            <tr><th style='border:1px solid #ddd; padding:8px;'>Model</th>
                <th style='border:1px solid #ddd; padding:8px;'>PSNR</th>
                <th style='border:1px solid #ddd; padding:8px;'>SSIM</th></tr>
            <tr><td style='border:1px solid #ddd; padding:8px;'>ECCV16</td>
                <td style='border:1px solid #ddd; padding:8px;'>{psnr_eccv:.2f}</td>
                <td style='border:1px solid #ddd; padding:8px;'>{ssim_eccv:.3f}</td></tr>
            <tr><td style='border:1px solid #ddd; padding:8px;'>{model_name}</td>
                <td style='border:1px solid #ddd; padding:8px;'>{psnr_enhanced:.2f}</td>
                <td style='border:1px solid #ddd; padding:8px;'>{ssim_enhanced:.3f}</td></tr>
            </table>
            """
        else:
            metrics = "<p>ℹ️ <b>No ground truth provided</b> - metrics not computed.</p>"
        
        # Metadata display
        metadata_info = f"""
        <div style='padding:10px; border:1px solid #ddd; border-radius:5px; background:#f9f9f9;'>
        <h4>Processing Information:</h4>
        <ul>
        <li><b>Base Model:</b> {'DDColor' if metadata.get('ddcolor_used', False) else 'SIGGRAPH17'}</li>
        <li><b>Features Applied:</b> {', '.join(metadata.get('features_applied', ['none']))}</li>
        {f"<li><b>Style:</b> {metadata.get('style_applied', 'none')}</li>" if metadata.get('style_applied') else ""}
        {f"<li><b>Color Hints:</b> {metadata.get('color_hints_applied', 0)} applied</li>" if metadata.get('color_hints_applied') else ""}
        {f"<li><b>Reference Guidance:</b> Applied</li>" if metadata.get('reference_guided') else ""}
        </ul>
        </div>
        """
        
        return slider_eccv, slider_enhanced, metrics, metadata_info
        
    except Exception as e:
        error_msg = f"❌ <b>Error:</b> {str(e)}"
        print(f"Error in handler_enhanced: {e}")
        import traceback
        traceback.print_exc()
        return None, None, error_msg, ""

# Build interface
with gr.Blocks(title="🎨 ColorizeAI", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🎨 ColorizeAI - Image Colorization Suite")
    
    # Status
    dd_status = "🟢 DDColor Active" if is_ddcolor_available() else "🔴 DDColor Inactive"
    gr.HTML(f"""
    <div style='padding:10px; background:#f0f0f0; border-radius:5px; margin-bottom:20px;'>
    <b>Status:</b> {dd_status} | <b>Device:</b> CPU
    </div>
    """)
    
    with gr.Tabs():
        # Basic Tab
        with gr.Tab("🖼️ Basic Colorization"):
            gr.Markdown("### Upload a grayscale image and colorize it")
            
            with gr.Row():
                with gr.Column():
                    inp_img = gr.Image(label="📤 Upload Grayscale Image", type="numpy")
                    gt_img = gr.Image(label="🎯 Ground Truth (optional)", type="numpy")
                    strength = gr.Slider(0, 1, value=1.0, step=0.1, label="🎨 Strength")
                    run_basic = gr.Button("🚀 Colorize", variant="primary", size="lg")
            
            with gr.Row():
                slider_eccv = gr.ImageSlider(label="Original ↔ ECCV16", type="numpy")
                slider_sig = gr.ImageSlider(label="Original ↔ DDColor/SIGGRAPH17", type="numpy")
            
            metrics = gr.HTML()
            
            run_basic.click(
                fn=handler_basic,
                inputs=[inp_img, strength, gt_img],
                outputs=[slider_eccv, slider_sig, metrics]
            )
        
        # Enhanced Tab
        with gr.Tab("🚀 Enhanced Colorization"):
            gr.Markdown("### Advanced features: fusion, reference, hints, styles")
            
            with gr.Row():
                with gr.Column():
                    inp_img_enh = gr.Image(label="📤 Upload Grayscale Image", type="numpy")
                    gt_img_enh = gr.Image(label="🎯 Ground Truth (optional)", type="numpy")
                    ref_img = gr.Image(label="🌈 Reference Image (optional)", type="numpy")
                    
                    strength_enh = gr.Slider(0, 1, value=1.0, step=0.1, label="🎨 Strength")
                    use_ensemble = gr.Checkbox(value=True, label="🧠 Smart Model Fusion")
                    
                    style_type = gr.Dropdown(
                        ['none', 'modern', 'vintage', 'cinematic', 'pastel', 'vibrant', 
                         'film_kodak', 'film_fuji', 'artistic_oil_painting'],
                        value='modern',
                        label="🎭 Style Preset"
                    )
                    
                    color_hints_json = gr.Textbox(
                        label="🖌️ Color Hints (JSON)",
                        placeholder='[{"x":100,"y":50,"r":255,"g":0,"b":0}]',
                        lines=2
                    )
                    
                    run_enhanced = gr.Button("🚀 Enhanced Colorization", variant="primary", size="lg")
            
            with gr.Row():
                slider_eccv_enh = gr.ImageSlider(label="Original ↔ ECCV16", type="numpy")
                slider_enh = gr.ImageSlider(label="Original ↔ Enhanced", type="numpy")
            
            with gr.Row():
                metrics_enh = gr.HTML()
                metadata_html = gr.HTML()
            
            run_enhanced.click(
                fn=handler_enhanced,
                inputs=[inp_img_enh, strength_enh, gt_img_enh, use_ensemble, ref_img, style_type, color_hints_json],
                outputs=[slider_eccv_enh, slider_enh, metrics_enh, metadata_html]
            )
    
        # Batch Processing Tab
        with gr.Tab("📦 Batch Processing"):
            gr.Markdown("### Upload multiple images for batch colorization")
            
            with gr.Row():
                with gr.Column():
                    files_input = gr.Files(label="📁 Upload Multiple Images", file_count="multiple")
                    strength_batch = gr.Slider(0, 1, value=1.0, step=0.1, label="🎨 Strength")
                    run_batch = gr.Button("⚡ Process Batch", variant="primary", size="lg")
                    
                    gr.Markdown("""
                    **📋 Guide:**
                    1. Upload multiple grayscale images (JPG, PNG, etc.)
                    2. Adjust colorization strength
                    3. Click Process Batch and wait
                    4. Download ZIP with results
                    
                    *Images > 1024px are auto-resized for speed*
                    """)
            
            batch_status = gr.HTML("<p>📋 Ready to process images...</p>")
            
            with gr.Row():
                gallery_eccv = gr.Gallery(label="🎨 ECCV16 Results", columns=4)
                gallery_sig = gr.Gallery(label="🎨 DDColor/SIGGRAPH17 Results", columns=4)
            
            zip_out = gr.File(label="📥 Download Results (ZIP)")
            
            def handler_batch(files, strength, progress=gr.Progress()):
                """Batch processing handler"""
                if not files:
                    return [], [], None, "❌ No files uploaded"
                
                import tempfile
                import zipfile
                import io
                from pathlib import Path
                
                progress(0, desc="Starting batch processing...")
                
                eccv_gallery, sig_gallery = [], []
                processed = 0
                total = len(files)
                
                tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
                
                try:
                    with zipfile.ZipFile(tmp_zip, "w") as zf:
                        for idx, file_path in enumerate(files):
                            progress(idx / total, desc=f"Processing {idx + 1}/{total}")
                            
                            try:
                                # Load and process
                                img = Image.open(file_path).convert("RGB")
                                img_np = np.array(img)
                                
                                # Resize if large
                                h, w = img_np.shape[:2]
                                if max(h, w) > 1024:
                                    scale = 1024 / max(h, w)
                                    new_h, new_w = int(h * scale), int(w * scale)
                                    img_np = cv2.resize(img_np, (new_w, new_h))
                                
                                # Colorize
                                eccv_img, sig_img = colorize_highres(img_np, strength, use_ddcolor=True)
                                
                                eccv_uint8 = (eccv_img * 255).astype(np.uint8)
                                sig_uint8 = (sig_img * 255).astype(np.uint8)
                                
                                eccv_gallery.append(eccv_uint8)
                                sig_gallery.append(sig_uint8)
                                
                                # Save to ZIP
                                base = Path(file_path).stem
                                with io.BytesIO() as buff:
                                    Image.fromarray(eccv_uint8).save(buff, format="PNG")
                                    zf.writestr(f"{base}_eccv16.png", buff.getvalue())
                                with io.BytesIO() as buff:
                                    Image.fromarray(sig_uint8).save(buff, format="PNG")
                                    zf.writestr(f"{base}_ddcolor.png", buff.getvalue())
                                
                                processed += 1
                            except Exception as e:
                                print(f"Failed to process {file_path}: {e}")
                                continue
                    
                    progress(1.0, desc=f"Complete! Processed {processed}/{total}")
                    
                    if processed == total:
                        status = f"✅ Success! Processed all {processed} images."
                    elif processed > 0:
                        status = f"⚠️ Partial success: {processed}/{total} images processed."
                    else:
                        status = "❌ Failed to process any images."
                    
                    return eccv_gallery, sig_gallery, tmp_zip.name, status
                    
                except Exception as e:
                    return [], [], None, f"❌ Error: {str(e)}"
            
            run_batch.click(
                fn=handler_batch,
                inputs=[files_input, strength_batch],
                outputs=[gallery_eccv, gallery_sig, zip_out, batch_status]
            )
        
        # Video Processing Tab
        with gr.Tab("🎬 Video Colorization"):
            gr.Markdown("### Transform black & white videos")
            
            with gr.Row():
                with gr.Column():
                    vid_input = gr.File(label="🎥 Upload Video", file_types=[".mp4", ".avi", ".mov"])
                    strength_vid = gr.Slider(0, 1, value=1.0, step=0.1, label="🎨 Strength")
                    
                    fast_mode = gr.Checkbox(value=True, label="⚡ Fast Mode (Recommended)")
                    
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        frame_skip = gr.Slider(1, 30, value=3, step=1, label="Frame Skip")
                        resolution = gr.Dropdown(
                            ["Original", "720p", "1080p", "480p"],
                            value="720p",
                            label="Resolution"
                        )
                        use_temporal = gr.Checkbox(value=False, label="Temporal Consistency")
                        style_vid = gr.Dropdown(
                            ['none', 'modern', 'vintage', 'cinematic', 'film_kodak'],
                            value='none',
                            label="Style"
                        )
                    
                    run_vid = gr.Button("🎬 Process Video", variant="primary", size="lg")
                    
                    gr.Markdown("""
                    **⚡ Tips:**
                    - Fast Mode: Faster processing with higher frame skip
                    - Frame Skip 3-5: Good balance of speed/quality
                    - 720p: Recommended for most videos
                    - Temporal Consistency: Reduces flicker (slower)
                    """)
                
                with gr.Column():
                    vid_out = gr.Video(label="🎥 Colorized Video")
            
            def handler_video(video_file, strength, frame_skip, resolution, fast_mode, use_temporal, style_vid, progress=gr.Progress()):
                """Video processing handler"""
                if video_file is None:
                    return None
                
                import tempfile
                from colorizeai.features.temporal_consistency import TemporalConsistencyEngine
                
                progress(0, desc="🎬 Starting video processing...")
                
                # Initialize temporal engine if needed
                temporal_engine = None
                if use_temporal:
                    temporal_engine = TemporalConsistencyEngine()
                
                try:
                    cap = cv2.VideoCapture(video_file)
                    if not cap.isOpened():
                        return None
                    
                    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Determine output resolution
                    if resolution == "720p":
                        out_w, out_h = 1280, 720
                    elif resolution == "1080p":
                        out_w, out_h = 1920, 1080
                    elif resolution == "480p":
                        out_w, out_h = 854, 480
                    else:
                        out_w, out_h = width, height
                    
                    # Ensure even dimensions
                    out_w = out_w if out_w % 2 == 0 else out_w - 1
                    out_h = out_h if out_h % 2 == 0 else out_h - 1
                    
                    # Create output file
                    output_path = tempfile.mktemp(suffix=".mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
                    
                    if not writer.isOpened():
                        cap.release()
                        return None
                    
                    frame_idx = 0
                    last_colored_frame = None
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Progress update
                        if frame_idx % 30 == 0:
                            progress(frame_idx / total_frames, desc=f"Frame {frame_idx}/{total_frames}")
                        
                        # Process keyframes
                        if frame_idx % frame_skip == 0:
                            try:
                                # Resize if needed
                                if frame.shape[:2] != (out_h, out_w):
                                    frame_resized = cv2.resize(frame, (out_w, out_h))
                                else:
                                    frame_resized = frame
                                
                                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                                
                                # Colorize
                                if use_temporal or style_vid != 'none':
                                    _, colored, _ = colorize_highres_enhanced(
                                        frame_rgb, strength,
                                        use_ensemble=True,
                                        style_type=style_vid,
                                        use_ddcolor=True
                                    )
                                    if use_temporal and temporal_engine:
                                        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                                        colored = temporal_engine.apply_temporal_consistency(colored, gray)
                                else:
                                    _, colored = colorize_highres(frame_rgb, strength, use_ddcolor=True)
                                
                                colored_uint8 = (colored * 255).astype(np.uint8)
                                last_colored_frame = colored_uint8
                            except Exception as e:
                                print(f"Frame {frame_idx} failed: {e}")
                                colored_uint8 = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                                last_colored_frame = colored_uint8
                        else:
                            # Use last colored frame for skipped frames
                            if last_colored_frame is not None:
                                colored_uint8 = last_colored_frame
                            else:
                                frame_resized = cv2.resize(frame, (out_w, out_h))
                                colored_uint8 = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        
                        # Write frame
                        frame_bgr = cv2.cvtColor(colored_uint8, cv2.COLOR_RGB2BGR)
                        writer.write(frame_bgr)
                        frame_idx += 1
                    
                    cap.release()
                    writer.release()
                    
                    progress(1.0, desc="✅ Video processing complete!")
                    return output_path
                    
                except Exception as e:
                    print(f"Video processing error: {e}")
                    return None
            
            run_vid.click(
                fn=handler_video,
                inputs=[vid_input, strength_vid, frame_skip, resolution, fast_mode, use_temporal, style_vid],
                outputs=vid_out
            )
    
    gr.Markdown("""
    ---
    ### 💡 Tips
    - **Basic Mode**: Fast colorization with ECCV16 and DDColor/SIGGRAPH17
    - **Enhanced Mode**: Add reference images, color hints, and style presets
    - **Batch Processing**: Upload multiple images, get ZIP with results
    - **Video Colorization**: Process videos with temporal consistency
    - **Strength**: Lower values preserve more grayscale information
    
    **Sample Data**: Check `assets/sample_images/` and `assets/sample_videos/`
    """)

if __name__ == "__main__":
    print("Starting ColorizeAI...")
    demo.launch()
