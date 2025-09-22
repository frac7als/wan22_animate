import subprocess
import os
import json
import requests
import time
import modal
from pathlib import Path

# Modal image with ComfyUI and required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "build-essential",
        "cmake",
        "wget",
    )
    .pip_install(
        "opencv-python-headless",
        "imageio[ffmpeg]",
        "moviepy",
        "fastapi[standard]==0.115.4",
        "comfy-cli==1.5.1",
        "requests",
    )
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.59"
    )
)

# Install required custom nodes for Wan 2.2 workflow
image = image.run_commands(
    "comfy node install --fast-deps was-node-suite-comfyui@1.0.2",
    "git clone https://github.com/kijai/ComfyUI-KJNodes.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    "git clone https://github.com/cubiq/ComfyUI_essentials.git /root/comfy/ComfyUI/custom_nodes/ComfyUI_essentials",
    "git clone https://github.com/city96/ComfyUI-GGUF.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-GGUF",
    "git clone https://github.com/rgthree/rgthree-comfy.git /root/comfy/ComfyUI/custom_nodes/rgthree-comfy",
    "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    "git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation",
    "git clone https://github.com/yolain/ComfyUI-Easy-Use.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-Easy-Use",
    "git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts",
    "git clone https://github.com/chflame163/ComfyUI_LayerStyle.git /root/comfy/ComfyUI/custom_nodes/ComfyUI_LayerStyle",
    "git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-Gemini.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-Gemini",
    "git clone https://github.com/ltdrdata/ComfyUI-Manager.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-Manager",
)

def download_wan22_models():
    """Download all required Wan 2.2 models"""
    from huggingface_hub import hf_hub_download

    # Create model directories
    os.makedirs("/root/comfy/ComfyUI/models/diffusion_models", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/vae", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/text_encoders", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/loras", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/rife", exist_ok=True)

    # Download Wan 2.2 High Noise Model
    wan_high_noise = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {wan_high_noise} /root/comfy/ComfyUI/models/diffusion_models/wan2.2_fun_vace_high_noise_14B_bf16.safetensors",
        shell=True,
        check=True,
    )

    # Download Wan 2.2 Low Noise Model
    wan_low_noise = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {wan_low_noise} /root/comfy/ComfyUI/models/diffusion_models/wan2.2_fun_vace_low_noise_14B_bf16.safetensors",
        shell=True,
        check=True,
    )

    # Download VAE
    vae_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/vae/wan_2.1_vae.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {vae_model} /root/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors",
        shell=True,
        check=True,
    )

    # Download Text Encoder
    text_encoder = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {text_encoder} /root/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp16.safetensors",
        shell=True,
        check=True,
    )

    # Download essential LoRAs for Wan 2.2
    try:
        # Try to download Seko LoRAs
        high_noise_lora = hf_hub_download(
            repo_id="Kijai/WanVideo_comfy",
            filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
            cache_dir="/cache",
        )
        subprocess.run(
            f"ln -sf {high_noise_lora} /root/comfy/ComfyUI/models/loras/high_noise_model.safetensors",
            shell=True,
            check=True,
        )

        low_noise_lora = hf_hub_download(
            repo_id="Kijai/WanVideo_comfy",
            filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
            cache_dir="/cache",
        )
        subprocess.run(
            f"ln -sf {low_noise_lora} /root/comfy/ComfyUI/models/loras/low_noise_model.safetensors",
            shell=True,
            check=True,
        )
    except Exception as e:
        print(f"Warning: Could not download Seko LoRAs: {e}")

    # Download RIFE model for frame interpolation
    try:
        import urllib.request
        rife_url = "https://github.com/hzwer/Practical-RIFE/releases/download/v4.7/rife47.pth"
        urllib.request.urlretrieve(rife_url, "/root/comfy/ComfyUI/models/rife/rife47.pth")
    except Exception as e:
        print(f"Warning: Could not download RIFE model: {e}")

vol = modal.Volume.from_name("wan22-cache", create_if_missing=True)

# Build image with models
image = (
    image.pip_install("huggingface_hub[hf_transfer]>=0.34.0,<1.0")
    .run_function(
        download_wan22_models,
        volumes={"/cache": vol},
    )
)

app = modal.App(name="comfyui-wan22-workflow", image=image)

@app.function(
    gpu="L40S",
    memory=32768,  # 32GB RAM
    timeout=3600,  # 1 hour timeout
    volumes={"/cache": vol},
    allow_concurrent_inputs=1,
)
def run_wan22_workflow(
    workflow_json: str,
    start_image_data: bytes = None,
    prompt_text: str = "Cinematic scenes, high quality photography, photorealistic",
    video_length_seconds: int = 30,
    fps: float = 16.0,
    width: int = 832,
    height: int = 480,
):
    """Run the Wan 2.2 VACE workflow"""
    
    # Save start image if provided
    if start_image_data:
        input_dir = "/root/comfy/ComfyUI/input"
        os.makedirs(input_dir, exist_ok=True)
        with open(f"{input_dir}/start_image.jpg", "wb") as f:
            f.write(start_image_data)
    
    # Parse and update workflow
    workflow = json.loads(workflow_json)
    
    # Update workflow parameters
    for node_id, node in workflow.get("nodes", []).items() if isinstance(workflow.get("nodes"), dict) else enumerate(workflow.get("nodes", [])):
        if isinstance(node, dict):
            node_type = node.get("type", "")
            title = node.get("title", "")
            
            # Update prompts
            if node_type == "PrimitiveStringMultiline":
                if node.get("widgets_values"):
                    node["widgets_values"][0] = prompt_text
            
            # Update video settings
            elif node_type == "PrimitiveInt":
                if "Video Total Seconds" in title:
                    node["widgets_values"][0] = video_length_seconds
                elif "Width" in title:
                    node["widgets_values"][0] = width
                elif "Height" in title:
                    node["widgets_values"][0] = height
            
            elif node_type == "PrimitiveFloat":
                if "FPS" in title:
                    node["widgets_values"][0] = fps
            
            # Update image input
            elif node_type == "LoadImage" and start_image_data:
                node["widgets_values"][0] = "start_image.jpg"
    
    # Save workflow to temp file
    workflow_path = "/tmp/workflow.json"
    with open(workflow_path, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    # Start ComfyUI server in background
    print("Starting ComfyUI server...")
    server_process = subprocess.Popen([
        "comfy", "launch", "--",
        "--listen", "0.0.0.0",
        "--port", "8188",
        "--dont-upcast-attention"
    ])
    
    # Wait for server to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8188/", timeout=5)
            if response.status_code == 200:
                print("ComfyUI server is ready!")
                break
        except:
            pass
        time.sleep(2)
        if i == max_retries - 1:
            server_process.terminate()
            raise Exception("ComfyUI server failed to start")
    
    try:
        # Submit workflow
        print("Submitting workflow...")
        with open(workflow_path, 'r') as f:
            workflow_data = json.load(f)
        
        response = requests.post(
            "http://localhost:8188/prompt",
            json={"prompt": workflow_data},
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to submit workflow: {response.status_code} - {response.text}")
        
        prompt_id = response.json()["prompt_id"]
        print(f"Workflow submitted with prompt_id: {prompt_id}")
        
        # Monitor progress
        start_time = time.time()
        while time.time() - start_time < 3000:  # 50 minute timeout
            try:
                # Check if workflow completed
                history_response = requests.get("http://localhost:8188/history", timeout=10)
                if history_response.status_code == 200:
                    history = history_response.json()
                    if prompt_id in history:
                        if history[prompt_id].get("status", {}).get("completed", False):
                            print("Workflow completed successfully!")
                            break
                        elif "error" in history[prompt_id].get("status", {}):
                            error_msg = history[prompt_id]["status"]["error"]
                            raise Exception(f"Workflow failed: {error_msg}")
                
                # Check queue status
                queue_response = requests.get("http://localhost:8188/queue", timeout=10)
                if queue_response.status_code == 200:
                    queue_data = queue_response.json()
                    if not any(item[1] == prompt_id for item in queue_data.get("queue_running", [])):
                        # Not in running queue, check if completed
                        time.sleep(5)
                        continue
                
                print("Workflow still processing...")
                time.sleep(10)
                
            except requests.RequestException as e:
                print(f"Error checking status: {e}")
                time.sleep(5)
        else:
            raise Exception("Workflow timed out")
        
        # Collect output files
        output_dir = "/root/comfy/ComfyUI/output"
        output_files = []
        
        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(('.mp4', '.avi', '.mov', '.gif')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'rb') as f:
                                output_files.append({
                                    'filename': file,
                                    'data': f.read(),
                                    'size': os.path.getsize(file_path)
                                })
                        except Exception as e:
                            print(f"Error reading output file {file_path}: {e}")
        
        return {
            'success': True,
            'prompt_id': prompt_id,
            'output_files': output_files,
            'message': f"Generated {len(output_files)} output files"
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
    
    finally:
        # Clean up
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()

@app.function(
    gpu="L40S",
    volumes={"/cache": vol},
)
@modal.web_server(8000, startup_timeout=120)
def serve_ui():
    """Serve ComfyUI web interface"""
    subprocess.Popen([
        "comfy", "launch", "--",
        "--listen", "0.0.0.0", 
        "--port", "8000"
    ])

@app.local_entrypoint()
def main(
    workflow_path: str = "workflow.json",
    start_image: str = None,
    prompt: str = "Cinematic scenes, high quality photography, photorealistic",
    video_length: int = 30,
    fps: float = 16.0,
    serve_ui: bool = False
):
    """Main entry point"""
    
    if serve_ui:
        print("Starting ComfyUI web interface...")
        print("Access at: https://your-modal-app-url--serve-ui-8000.modal.run")
        return
    
    # Load workflow
    if not os.path.exists(workflow_path):
        print(f"Error: Workflow file {workflow_path} not found")
        return
    
    with open(workflow_path, 'r') as f:
        workflow_json = f.read()
    
    # Load start image if provided
    start_image_data = None
    if start_image and os.path.exists(start_image):
        with open(start_image, 'rb') as f:
            start_image_data = f.read()
    
    # Run workflow
    print("Running Wan 2.2 workflow...")
    result = run_wan22_workflow.remote(
        workflow_json=workflow_json,
        start_image_data=start_image_data,
        prompt_text=prompt,
        video_length_seconds=video_length,
        fps=fps
    )
    
    if result['success']:
        print(f"Success! {result['message']}")
        
        # Save output files
        for i, output in enumerate(result['output_files']):
            filename = f"output_{i}_{output['filename']}"
            with open(filename, 'wb') as f:
                f.write(output['data'])
            print(f"Saved: {filename} ({output['size']} bytes)")
    else:
        print(f"Failed: {result['error']}")
