import modal
import json
import os
import subprocess
import requests
import time
from pathlib import Path

# Modal app definition
app = modal.App("comfyui-wan22-workflow")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "wget", 
        "curl",
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
    )
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0", 
        "torchaudio==2.1.0",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "accelerate",
        "diffusers",
        "transformers",
        "xformers",
        "opencv-python",
        "Pillow",
        "numpy",
        "safetensors",
        "aiohttp",
        "server",
        "GitPython",
        "torchsde",
        "einops",
        "psutil",
        "kornia",
        "spandrel",
        "soundfile",
        "folder-paths",
    )
    .run_commands(
        # Clone ComfyUI
        "cd /root && git clone https://github.com/comfyanonymous/ComfyUI.git",
        "cd /root/ComfyUI && git checkout v0.3.59",
        
        # Install ComfyUI requirements
        "cd /root/ComfyUI && pip install -r requirements.txt",
        
        # Create model directories
        "mkdir -p /root/ComfyUI/models/diffusion_models",
        "mkdir -p /root/ComfyUI/models/vae", 
        "mkdir -p /root/ComfyUI/models/text_encoders",
        "mkdir -p /root/ComfyUI/models/loras",
        "mkdir -p /root/ComfyUI/models/rife",
        
        # Install custom nodes
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/kijai/ComfyUI-KJNodes.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/city96/ComfyUI-GGUF.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/yolain/ComfyUI-Easy-Use.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/cubiq/ComfyUI_essentials.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/rgthree/rgthree-comfy.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/ComfyNodePRs/PR-ComfyNodePRs-2407.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/WASasquatch/was-node-suite-comfyui.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/chflame163/ComfyUI_LayerStyle.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-Gemini.git",
        
        # Install custom node dependencies
        "cd /root/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && pip install -r requirements.txt",
        "cd /root/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation && pip install -r requirements.txt", 
        "cd /root/ComfyUI/custom_nodes/ComfyUI-KJNodes && pip install -r requirements.txt",
        "cd /root/ComfyUI/custom_nodes/ComfyUI-GGUF && pip install -r requirements.txt",
        "cd /root/ComfyUI/custom_nodes/ComfyUI_essentials && pip install -r requirements.txt",
    )
)

def download_file(url: str, filepath: str):
    """Download a file with progress tracking"""
    print(f"Downloading {url} to {filepath}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"Progress: {progress:.1f}%")
    
    print(f"Download completed: {filepath}")

@app.function(
    image=image,
    gpu=modal.gpu.L40S(),  # L40S GPU as requested
    timeout=3600,  # 1 hour timeout
    memory=32768,  # 32GB RAM
    volumes={"/models": modal.Volume.from_name("wan22-models", create_if_missing=True)}
)
def setup_models():
    """Download and setup all required models"""
    
    models_dir = "/models"
    comfy_models = "/root/ComfyUI/models"
    
    # Model download URLs
    model_urls = {
        # GGUF Models (choose these for L40S - lower VRAM usage)
        "diffusion_models/wan2.2_high_noise_Q8_0.gguf": 
            "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/Wan2.2-VACE-Fun-A14B-high-noise-Q8_0.gguf",
        "diffusion_models/wan2.2_low_noise_Q8_0.gguf": 
            "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/Wan2.2-VACE-Fun-A14B-low-noise-Q8_0.gguf",
        
        # VAE
        "vae/wan_2.1_vae.safetensors":
            "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
        
        # Text Encoder
        "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors":
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        
        # RIFE Model
        "rife/rife47.pth":
            "https://github.com/hzwer/Practical-RIFE/releases/download/v4.7/rife47.pth",
    }
    
    # Download models to persistent volume
    for relative_path, url in model_urls.items():
        filepath = os.path.join(models_dir, relative_path)
        if not os.path.exists(filepath):
            try:
                download_file(url, filepath)
            except Exception as e:
                print(f"Error downloading {url}: {e}")
                continue
    
    # Create symlinks to ComfyUI models directory
    for relative_path in model_urls.keys():
        source = os.path.join(models_dir, relative_path)
        target = os.path.join(comfy_models, relative_path)
        
        if os.path.exists(source):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if not os.path.exists(target):
                os.symlink(source, target)
                print(f"Linked {source} -> {target}")
    
    print("Model setup completed!")
    return "Models setup completed"

@app.function(
    image=image,
    gpu=modal.gpu.L40S(),
    timeout=1800,  # 30 minutes for workflow execution
    memory=32768,
    volumes={"/models": modal.Volume.from_name("wan22-models", create_if_missing=True)}
)
def run_workflow(
    workflow_json: str,
    start_image_url: str = None,
    prompt_text: str = "Cinematic scenes, high quality photography, photorealistic",
    video_length_seconds: int = 30,
    fps: float = 16,
    width: int = 832,
    height: int = 480,
):
    """Run the Wan 2.2 workflow with given parameters"""
    
    # Setup model links
    models_dir = "/models"
    comfy_models = "/root/ComfyUI/models"
    
    # Create symlinks
    model_types = ["diffusion_models", "vae", "text_encoders", "rife", "loras"]
    for model_type in model_types:
        source_dir = os.path.join(models_dir, model_type)
        target_dir = os.path.join(comfy_models, model_type)
        
        if os.path.exists(source_dir):
            os.makedirs(target_dir, exist_ok=True)
            for model_file in os.listdir(source_dir):
                source = os.path.join(source_dir, model_file)
                target = os.path.join(target_dir, model_file)
                if not os.path.exists(target) and os.path.isfile(source):
                    os.symlink(source, target)
    
    # Download start image if provided
    input_dir = "/root/ComfyUI/input"
    os.makedirs(input_dir, exist_ok=True)
    
    if start_image_url:
        image_path = os.path.join(input_dir, "start_image.jpg")
        download_file(start_image_url, image_path)
    
    # Parse and modify workflow
    workflow = json.loads(workflow_json)
    
    # Update workflow parameters
    for node_id, node in workflow["nodes"].items():
        # Update prompt text
        if node.get("type") == "PrimitiveStringMultiline":
            node["widgets_values"][0] = prompt_text
        
        # Update video settings
        elif node.get("type") == "PrimitiveInt":
            if "Video Total Seconds" in node.get("title", ""):
                node["widgets_values"][0] = video_length_seconds
            elif "Width" in node.get("title", ""):
                node["widgets_values"][0] = width
            elif "Height" in node.get("title", ""):
                node["widgets_values"][0] = height
        
        elif node.get("type") == "PrimitiveFloat":
            if "FPS" in node.get("title", ""):
                node["widgets_values"][0] = fps
        
        # Update image input
        elif node.get("type") == "LoadImage" and start_image_url:
            node["widgets_values"][0] = "start_image.jpg"
    
    # Save modified workflow
    workflow_path = "/tmp/workflow.json"
    with open(workflow_path, 'w') as f:
        json.dump(workflow, f)
    
    # Start ComfyUI server
    print("Starting ComfyUI server...")
    server_process = subprocess.Popen([
        "python", "/root/ComfyUI/main.py", 
        "--listen", "0.0.0.0", 
        "--port", "8188",
        "--dont-upcast-attention"
    ])
    
    # Wait for server to start
    time.sleep(30)
    
    try:
        # Submit workflow
        print("Submitting workflow...")
        with open(workflow_path, 'r') as f:
            workflow_data = json.load(f)
        
        response = requests.post(
            "http://localhost:8188/prompt",
            json={"prompt": workflow_data}
        )
        
        if response.status_code == 200:
            prompt_id = response.json()["prompt_id"]
            print(f"Workflow submitted with prompt_id: {prompt_id}")
            
            # Poll for completion
            while True:
                status_response = requests.get("http://localhost:8188/history")
                if status_response.status_code == 200:
                    history = status_response.json()
                    if prompt_id in history:
                        print("Workflow completed!")
                        break
                
                time.sleep(10)
                print("Waiting for workflow completion...")
            
            # Get output files
            output_dir = "/root/ComfyUI/output"
            output_files = []
            
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(('.mp4', '.avi', '.mov')):
                        full_path = os.path.join(root, file)
                        with open(full_path, 'rb') as f:
                            output_files.append({
                                'filename': file,
                                'data': f.read()
                            })
            
            return {
                'success': True,
                'prompt_id': prompt_id,
                'output_files': output_files
            }
        
        else:
            return {
                'success': False,
                'error': f"Failed to submit workflow: {response.status_code} - {response.text}"
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"Error running workflow: {str(e)}"
        }
    
    finally:
        server_process.terminate()

@app.local_entrypoint()
def main(
    workflow_file: str = "workflow.json",
    start_image_url: str = None,
    prompt: str = "Cinematic scenes, high quality photography, photorealistic",
    video_length: int = 30,
    fps: float = 16,
    setup_only: bool = False
):
    """Main entry point for running the Wan 2.2 workflow"""
    
    print("Setting up models...")
    setup_result = setup_models.remote()
    print(f"Setup result: {setup_result}")
    
    if setup_only:
        print("Setup completed. Exiting as requested.")
        return
    
    if not os.path.exists(workflow_file):
        print(f"Error: Workflow file {workflow_file} not found")
        return
    
    print("Loading workflow...")
    with open(workflow_file, 'r') as f:
        workflow_json = f.read()
    
    print("Running workflow...")
    result = run_workflow.remote(
        workflow_json=workflow_json,
        start_image_url=start_image_url,
        prompt_text=prompt,
        video_length_seconds=video_length,
        fps=fps
    )
    
    if result['success']:
        print("Workflow completed successfully!")
        
        # Save output files
        for output in result['output_files']:
            output_path = f"output_{output['filename']}"
            with open(output_path, 'wb') as f:
                f.write(output['data'])
            print(f"Saved output: {output_path}")
    
    else:
        print(f"Workflow failed: {result['error']}")

if __name__ == "__main__":
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        workflow_file = sys.argv[1]
    else:
        workflow_file = "workflow.json"  # Your uploaded workflow file
    
    main(
        workflow_file=workflow_file,
        start_image_url="https://example.com/your-start-image.jpg",  # Optional
        prompt="Cinematic scenes, high quality photography, photorealistic, smooth camera movement",
        video_length=30,  # seconds
        fps=16,
        setup_only=False  # Set to True for first run to just setup models
    )
