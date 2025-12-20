"""
Minimal test to debug Gradio UI issues
"""
import gradio as gr
import numpy as np
from PIL import Image

print("Testing Gradio components...")

def test_image_upload(img):
    if img is None:
        return None, "No image uploaded"
    
    print(f"Received image type: {type(img)}")
    print(f"Image shape: {img.shape if hasattr(img, 'shape') else 'N/A'}")
    
    # Convert to numpy if needed
    if isinstance(img, Image.Image):
        img = np.array(img)
        print("Converted PIL to numpy")
    
    # Return a simple processed version
    return img, f"✅ Image received! Type: {type(img)}, Shape: {img.shape}"

# Build simple interface
with gr.Blocks() as demo:
    gr.Markdown("# 🧪 Gradio Image Upload Test")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload Test Image")
            test_btn = gr.Button("Test Upload")
        
        with gr.Column():
            output_img = gr.Image(label="Output")
            status = gr.Textbox(label="Status")
    
    test_btn.click(
        fn=test_image_upload,
        inputs=input_img,
        outputs=[output_img, status]
    )

if __name__ == "__main__":
    print("Starting test server...")
    demo.launch(server_port=7861)
