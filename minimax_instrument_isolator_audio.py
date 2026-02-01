
# Cell 1: Clean Install & Protobuf Fix
import os

# 1. Force a "safe" protobuf version that Meta's research code expects
!pip install -U "protobuf==3.20.3"

# 2. Set the "Secret Sauce" environment variable (fixes the GetPrototype error)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 3. Install SAM-Audio and core dependencies
!pip install -q git+https://github.com/facebookresearch/sam-audio.git
!pip install -q gradio==4.* hf_transfer

# 4. Handle the specific Colab torchcodec crash
!pip uninstall -y torchcodec
!pip install -q --no-cache-dir "torchcodec==0.7.0" -f https://download.pytorch.org/whl/torchcodec/

print("✓ Done. Now go to Runtime -> Restart session")

# 1. Purge the crashing native library
!pip uninstall -y torchcodec

# 2. Mock the library so imports don't break
import sys
from unittest.mock import MagicMock
sys.modules["torchcodec"] = MagicMock()

# 3. Apply the previous protobuf fix (mandatory for Colab)
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
!pip install -U "protobuf==3.20.3"

# 4. Final SAM-Audio check
print("✓ Environment patched. Now try the import test below.")

import sys
import types
import importlib.machinery
from unittest.mock import MagicMock

def create_full_mock(name):
    # If it's already in sys.modules, use it; otherwise create it
    if name in sys.modules:
        return sys.modules[name]
    mock_mod = types.ModuleType(name)
    mock_mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mock_mod.__path__ = []
    sys.modules[name] = mock_mod
    return mock_mod

# 1. Create the hierarchy
tc = create_full_mock("torchcodec")
tc_dec = create_full_mock("torchcodec.decoders")
tc.decoders = tc_dec

# 2. Attach the specific missing classes that sam_audio/processor.py expects
# We make them MagicMocks so that if code tries to instantiate them, it won't crash
tc_dec.AudioDecoder = MagicMock
tc_dec.VideoDecoder = MagicMock

# 3. Apply the mandatory protobuf fix for Colab
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 4. Attempt the import
try:
    import torch
    import torchaudio
    from sam_audio import SAMAudio, SAMAudioProcessor
    print("✓ SAM-Audio imported successfully!")
    print(f"✓ AudioDecoder Mock: {tc_dec.AudioDecoder}")
except Exception as e:
    # Use traceback to see exactly where it fails if it's still stuck
    import traceback
    traceback.print_exc()
    print(f"✗ Import failed: {e}")

# 1. Install missing model requirements
!pip install -q sentencepiece

# 2. Advanced Mock: Inject metadata for torchcodec so importlib doesn't crash
import sys
from unittest.mock import MagicMock

# This satisfies importlib.metadata.version('torchcodec')
import importlib.metadata
original_version = importlib.metadata.version

def mocked_version(package_name):
    if package_name == "torchcodec":
        return "0.7.0"
    return original_version(package_name)

importlib.metadata.version = mocked_version

# 3. Repeat the structural mock from before
import types
import importlib.machinery

def create_full_mock(name):
    if name in sys.modules: return sys.modules[name]
    mock_mod = types.ModuleType(name)
    mock_mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mock_mod.__path__ = []
    sys.modules[name] = mock_mod
    return mock_mod

tc = create_full_mock("torchcodec")
tc_dec = create_full_mock("torchcodec.decoders")
tc.decoders = tc_dec
tc_dec.AudioDecoder = MagicMock
tc_dec.VideoDecoder = MagicMock

# 4. Mandatory Protobuf fix
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

print("✓ Metadata patched and sentencepiece installed.")

from huggingface_hub import login
login()

import torch
from sam_audio import SAMAudio, SAMAudioProcessor

# Use base for stability
MODEL_ID = "facebook/sam-audio-base"

print(f"Loading {MODEL_ID}...")
processor = SAMAudioProcessor.from_pretrained(MODEL_ID)
model = SAMAudio.from_pretrained(
    MODEL_ID,
    device_map="cuda",
    torch_dtype=torch.float16
).eval()

print("✓ SAM-Audio is loaded and ready!")

"""# Test Model"""

import torch
import gc
from sam_audio import SAMAudio, SAMAudioProcessor

MODEL_ID = "facebook/sam-audio-base"

# 1. AGGRESSIVE CLEANUP (Free up GPU space)
try:
    del model
    del processor
    del inputs
    del result
except: pass
gc.collect()
torch.cuda.empty_cache()
print(f"GPU Free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

# 2. Load Processor
processor = SAMAudioProcessor.from_pretrained(MODEL_ID)

# 3. Load Model to CPU (System RAM) first
print("Loading model to System RAM...")
model = SAMAudio.from_pretrained(MODEL_ID) # No device_map, no .to() yet

# 4. Move to GPU explicitly (The critical fix)
print("Moving model to GPU...")
model = model.to("cuda")

# 5. TEST IMMEDIATELY
print("Testing...")
dummy = torch.randn(1, 44100*2).to("cuda")
inputs = processor(audios=[dummy], descriptions=["test"]).to("cuda")

with torch.inference_mode():
    model.separate(inputs)

print("✓ SUCCESS! Model is fully on GPU and working.")

# 1. Run inference one more time to capture the result variable
with torch.inference_mode():
    result = model.separate(inputs)

# 2. Inspect the output
stem = result.target[0] # Get the first (and only) item in the batch

print(f"Output Type: {type(stem)}")
print(f"Output Shape: {stem.shape}")
# Expected: torch.Size([88200]) (approx, for ~2s audio)

# 3. Verify it's not silent (sanity check)
print(f"Max Amplitude: {stem.abs().max().item():.4f}")
# If this is > 0.0, the model actually produced sound!

import torch
import gc
import time
from sam_audio import SAMAudio, SAMAudioProcessor

# 1. Flush previous model
try:
    del model
    del processor
except: pass
gc.collect()
torch.cuda.empty_cache()

# 2. Load SMALL
MODEL_ID = "facebook/sam-audio-small"  # <--- The Switch
print(f"Loading {MODEL_ID}...")

processor = SAMAudioProcessor.from_pretrained(MODEL_ID)
model = SAMAudio.from_pretrained(MODEL_ID).to("cuda").eval()

print(f"✓ Model Loaded! VRAM Usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# 3. Time Inference
print("Timing Small Inference...")
dummy = torch.randn(1, 44100*2).to("cuda") # 2s audio
inputs = processor(audios=[dummy], descriptions=["test"]).to("cuda")

start = time.time()
with torch.inference_mode():
    model.separate(inputs)
end = time.time()

print(f"Small Inference Time (10s): {end - start:.2f} s")

# 1. Run inference one more time to capture the 'result' variable
with torch.inference_mode():
    result = model.separate(inputs)

# 2. Inspect the first target stem
stem = result.target[0] # List[Tensor] -> Tensor

print(f"Output Shape: {stem.shape}")

"""# Trying out demucs"""

# Cell 1: Install dependencies
!pip -q install demucs yt-dlp soundfile librosa

# Commented out IPython magic to ensure Python compatibility.
import sys

# Remove the torchcodec/torchaudio combo that is currently breaking demucs saves
# %pip -q uninstall -y torch torchaudio torchvision torchcodec

# Install a matched trio (CUDA 12.6 like your current torch==2.9.0+cu126)
# %pip -q install --no-cache-dir \
  torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126

# Reinstall demucs + tools (avoid pulling in unwanted upgrades)
# %pip -q install -U demucs yt-dlp soundfile --upgrade-strategy only-if-needed

# Sanity checks
!{sys.executable} -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)"

# Cell 2: Import libraries
import torch
import torchaudio
import soundfile as sf
import os
from pathlib import Path
import IPython.display as ipd

# Check GPU
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Commented out IPython magic to ensure Python compatibility.
import sys

# %pip -q install -U yt-dlp

!{sys.executable} -m yt_dlp -x --audio-format wav --audio-quality 0 \
  --postprocessor-args "-ss 00:00:12 -t 00:00:10" \
  -o "tchaikovsky_sample.%(ext)s" \
  "https://www.youtube.com/watch?v=BWerj8FcprM"

!{sys.executable} -m yt_dlp --no-playlist -x --audio-format wav --audio-quality 0 \
  --postprocessor-args "-ss 00:12:24 -t 00:00:10" \
  -o "rach3_sample.%(ext)s" \
  "https://www.youtube.com/watch?v=OnSXxMEIDp0"

!demucs -n htdemucs_6s --two-stems=piano -d cuda tchaikovsky_sample.wav

!demucs -n htdemucs_6s --two-stems=piano -d cuda rach3_sample.wav

!demucs -n htdemucs_6s --two-stems=piano -d cuda minimax_soaring.mp3

import IPython.display as ipd

ipd.display(ipd.Audio("separated/htdemucs_6s/minimax_soaring/piano.wav"))

ipd.display(ipd.Audio("separated/htdemucs_6s/minimax_soaring/no_piano.wav"))

ipd.display(ipd.Audio("tchaikovsky_sample.wav"))
ipd.display(ipd.Audio("separated/htdemucs_6s/tchaikovsky_sample/piano.wav"))
ipd.display(ipd.Audio("separated/htdemucs_6s/tchaikovsky_sample/no_piano.wav"))

"""Trying Rach 3 with Flute and Piano :D"""

ipd.display(ipd.Audio("rach3_sample.wav"))

ipd.display(ipd.Audio("separated/htdemucs_6s/rach3_sample/piano.wav"))
ipd.display(ipd.Audio("separated/htdemucs_6s/rach3_sample/no_piano.wav"))

import subprocess
import IPython.display as ipd

def isolate_piano(input_filename, output_dir="separated/htdemucs_6s"):
    """
    Isolates the piano stem from an audio file using Demucs and displays both stems.

    Args:
        input_filename (str): The name of your input audio file (e.g., 'minimax_soaring.mp3')
        output_dir (str): The base directory where Demucs saves output (default matches your command)

    Returns:
        tuple: Paths to the generated piano and no_piano audio files.
    """
    # Build the separation command
    command = [
        'demucs', '-n', 'htdemucs_6s',
        '--two-stems=piano',
        '-d', 'cuda',
        input_filename
    ]

    print(f"🎹 Separating piano from '{input_filename}'...")
    # Run the Demucs command
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Demucs encountered an error:\n{result.stderr}")
        return None, None

    print("✅ Separation complete!")

    # Construct the expected output paths
    # Demucs creates a folder named after the input file without extension
    track_name = input_filename.rsplit('.', 1)[0]  # Remove file extension
    piano_path = f"{output_dir}/{track_name}/piano.wav"
    no_piano_path = f"{output_dir}/{track_name}/no_piano.wav"

    # Display the audio files
    print("\n🔊 Playing isolated piano stem:")
    display(ipd.Audio(piano_path))

    print("\n🎵 Playing accompaniment (no piano) stem:")
    display(ipd.Audio(no_piano_path))

    return piano_path, no_piano_path

piano_track, other_track = isolate_piano("minimax_soaring.mp3")

piano_track, other_track = isolate_piano("Prism.mp3")

"""# Generate Hailuo Video"""

# Method 1: Using a Colab 'Secret' (Most Secure)
from google.colab import userdata
api_key = userdata.get('MINIMAX_API_KEY')  # Set this in Colab's 'Secrets' pane

import os
import time
import requests
from google.colab import userdata  # For API key retrieval in Colab

# --- CONFIGURATION ---
# Set your API key (choose one method below)
# METHOD 1: Using Colab Secrets (Recommended)
api_key = userdata.get('MINIMAX_API_KEY')  # Set this in Colab: Key="MINIMAX_API_KEY", Value="your-actual-key"

# METHOD 2: Direct input (alternative)
# api_key = input("Paste your Minimax API Key: ").strip()

headers = {"Authorization": f"Bearer {api_key}"}

# Define your 5 scene prompts
SCENE_PROMPTS = [
    "A Chinese woman in a vibrant red and gold embroidered Chinese dress walks slowly through an ancient palace courtyard at sunset, cinematic lighting, period drama, flowing silk, emotional atmosphere.",
    "Close-up of a Chinese woman's face looking pensively out from a traditional wooden pavilion, intricate hair ornaments, soft morning light filtering through lattice windows, tear on cheek.",
    "A Chinese man in a sleek black robe, pensively sitting on a stone pavilion, contemplating, cinematic lighting, period drama, emotional atmosphere, dusk",
    "A panorama of a beautiful peaceful monastery on a cliff, like Paro Taktsang, with a waterfall nearby",
    "A Chinese woman in a veil and beautiful red and gold embroidered Chinese dress, handling a sharp samurai sword, dashing through a Chinese palace, candles on the walls being extinguished as she runs past"
]

# Video settings
DURATION = 6  # Use 6 or 10 seconds per scene (check model limits)
MODEL = "MiniMax-Hailuo-2.3"
RESOLUTION = "1080P"

# --- Step 1: Create a video generation task ---
def invoke_text_to_video(prompt: str, scene_num: int) -> str:
    """Create a video generation task from text description."""
    url = "https://api.minimax.io/v1/video_generation"
    payload = {
        "prompt": prompt,
        "model": MODEL,
        "duration": DURATION,
        "resolution": RESOLUTION,
    }

    print(f"\n📽️ Generating Scene {scene_num}: '{prompt[:50]}...'")
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    task_id = response.json()["task_id"]
    print(f"   Task ID: {task_id}")
    return task_id

# --- Step 2: Poll task status ---
def query_task_status(task_id: str) -> str:
    """Poll task status by task_id until it succeeds or fails."""
    url = "https://api.minimax.io/v1/query/video_generation"
    params = {"task_id": task_id}

    print("   ⏳ Waiting for generation...", end="")
    while True:
        time.sleep(10)  # Poll every 10 seconds
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        response_json = response.json()
        status = response_json["status"]

        if status == "Success":
            print(" ✅ Done!")
            return response_json["file_id"]
        elif status == "Fail":
            error_msg = response_json.get('error_message', 'Unknown error')
            print(f" ❌ Failed: {error_msg}")
            raise Exception(f"Video generation failed: {error_msg}")
        else:
            print(".", end="")  # Show progress dots

# --- Step 3: Retrieve and save the video file ---
def fetch_video(file_id: str, filename: str):
    """Retrieve the download URL from file_id and save the video locally."""
    url = "https://api.minimax.io/v1/files/retrieve"
    params = {"file_id": file_id}

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    download_url = response.json()["file"]["download_url"]

    print(f"   💾 Downloading as {filename}...", end="")
    with open(filename, "wb") as f:
        video_response = requests.get(download_url)
        video_response.raise_for_status()
        f.write(video_response.content)
    print(" ✅ Saved!")

# --- Step 4: Combine videos (optional) ---
def combine_videos(scene_files, output_file="final_30s_video.mp4"):
    """Combine all scene videos into one continuous video."""
    try:
        # Install moviepy if not available
        import subprocess
        import sys

        # Try to import, install if needed
        try:
            from moviepy.editor import VideoFileClip, concatenate_videoclips
        except ImportError:
            print("\n🎬 Installing moviepy for video concatenation...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy"])
            from moviepy.editor import VideoFileClip, concatenate_videoclips

        print(f"\n🔗 Combining {len(scene_files)} scenes into {output_file}...")

        # Load all clips
        clips = [VideoFileClip(scene) for scene in scene_files]

        # Concatenate them
        final_clip = concatenate_videoclips(clips, method="compose")

        # Write the result
        final_clip.write_videofile(output_file, codec="libx264", audio=False)

        print(f"✅ Successfully created {output_file} ({len(clips)} scenes, {final_clip.duration:.1f}s)")
        return output_file

    except Exception as e:
        print(f"⚠️ Could not combine videos: {e}")
        print("   You can manually combine the scene files using video editing software.")
        return None

# --- Main execution ---
def main():
    print("=" * 60)
    print("🎬 MINIMAX VIDEO GENERATOR - 5 SCENES")
    print("=" * 60)

    generated_files = []

    # Generate each scene
    for i, prompt in enumerate(SCENE_PROMPTS, 1):
        scene_filename = f"scene_{i}_{DURATION}s.mp4"

        try:
            # Step 1: Create task
            task_id = invoke_text_to_video(prompt, i)

            # Step 2: Wait for completion
            file_id = query_task_status(task_id)

            # Step 3: Download video
            fetch_video(file_id, scene_filename)

            generated_files.append(scene_filename)

        except Exception as e:
            print(f"\n❌ Error generating Scene {i}: {e}")
            print("   Continuing with remaining scenes...")

    print("\n" + "=" * 60)
    print("📊 GENERATION SUMMARY")
    print("=" * 60)
    print(f"Successfully generated: {len(generated_files)}/{len(SCENE_PROMPTS)} scenes")

    for i, filename in enumerate(generated_files, 1):
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"  Scene {i}: {filename} ({size_mb:.1f} MB)")

    # Optional: Combine scenes
    if len(generated_files) > 1:
        combine = input("\n🔗 Combine all scenes into one video? (y/n): ").lower()
        if combine == 'y':
            combine_videos(generated_files)

    print("\n✨ All done! Your scenes are ready for your hackathon demo.")
    print("   For your 60-second social clip:")
    print("   1. Use the original audio with each scene")
    print("   2. Repeat with piano-only audio")
    print("   3. Add titles showing 'Full Audio' vs 'Isolated Piano'")

# Run the main function
if __name__ == "__main__":
    main()



"""# Stitching Together Video + Song"""

# 1. Video with Original Audio
!ffmpeg -y -i final_30s_video.mp4 -i minimax_soaring.mp3 \
    -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -t 29 \
    video_original.mp4

# 2. Video with Piano Filtered Out (no_piano.wav)
!ffmpeg -y -i final_30s_video.mp4 -i separated/htdemucs_6s/minimax_soaring/no_piano.wav \
    -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -t 29 \
    video_no_piano.mp4

from IPython.display import HTML
from base64 import b64encode

def show_video(file_path):
    mp4 = open(file_path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(f'<video width=600 controls><source src="{data_url}" type="video/mp4"></video>')

# To view :
show_video('video_no_piano.mp4')

show_video('video_original.mp4')

"""# Gradio UI"""

import gradio as gr
import requests
import json
import os
import time
import subprocess
from google.colab import userdata

# Load API Key from Colab Secrets
MINIMAX_API_KEY = userdata.get('MINIMAX_API_KEY')

def get_headers():
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINIMAX_API_KEY}"
    }

# --- MINIMAX FUNCTIONS (A1 & A2) ---
def generate_music(prompt, lyrics):
    url = "https://api.minimax.io/v1/music_generation"
    payload = {
        "model": "music-2.5",
        "prompt": prompt,
        "lyrics": lyrics,
        "audio_setting": {"sample_rate": 44100, "bitrate": 256000, "format": "mp3"},
        "output_format": "url"
    }
    response = requests.post(url, headers=get_headers(), json=payload)
    result = response.json()
    return result.get("data", {}).get("audio_url", "Error generating audio")

def generate_video_task(prompt):
    url = "https://api.minimax.io/v1/video_generation"
    payload = {"prompt": prompt, "model": "MiniMax-Hailuo-2.3", "duration": 6, "resolution": "1080P"}
    response = requests.post(url, headers=get_headers(), json=payload)
    return response.json().get("task_id")

# --- UI COMPONENTS ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="rose", secondary_hue="slate")) as demo:
    gr.Markdown("# 🎨 Minimax Fusion & 🎹 Stem Separation")
    gr.Markdown("🎹 AI-powered suite for generating and isolating piano.")

    with gr.Tabs():
        # TAB 1: GENERATION
        with gr.TabItem("🚀 Generation Studio"):
            with gr.Row():
                with gr.Column():
                    music_prompt = gr.Textbox(label="Music Style Prompt", value="Classical Romantic era / Chinese fusion composition, solo piano and concert flute.")
                    music_lyrics = gr.Textbox(label="Musical Cues", value="[Instrumental] [Piano and Flute Intro]")
                    gen_music_btn = gr.Button("Generate Minimax 2.5 Audio", variant="primary")
                    music_output = gr.Audio(label="Generated Audio Result")

                with gr.Column():
                    video_prompt = gr.Textbox(label="Video Scene Prompt", value="Chinese woman in vibrant red dress walking through ancient palace at sunset.")
                    gen_video_btn = gr.Button("Generate Cinematic Video", variant="primary")
                    video_output = gr.Video(label="Generated Video Result")

        # TAB 2: SEPARATION & COMPARISON
        with gr.TabItem("🎹 Stem Isolation & Comparison"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_audio = gr.Audio(label="Input Audio for Separation", type="filepath", value="minimax_soaring.mp3")
                    isolate_btn = gr.Button("Isolate Piano Stems", variant="secondary")

                with gr.Column(scale=2):
                    gr.Markdown("### 🔊 Audio Stems")
                    with gr.Row():
                        original_play = gr.Audio(label="Original (minimax_soaring.mp3)", value="minimax_soaring.mp3")
                        piano_play = gr.Audio(label="Isolated Piano", value="separated/htdemucs_6s/minimax_soaring/piano.wav")
                        bg_play = gr.Audio(label="Background (No Piano)", value="separated/htdemucs_6s/minimax_soaring/no_piano.wav")

            gr.Markdown("---")
            gr.Markdown("### 📺 Video Comparison")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Original Video + Full Music**")
                    video_orig = gr.Video(value="video_original.mp4", label="Original Mix")
                with gr.Column():
                    gr.Markdown("**Video + Isolated Background**")
                    video_no_p = gr.Video(value="video_no_piano.mp4", label="No Piano Mix")

    # Hooking up functions (Placeholders for demo)
    gen_music_btn.click(fn=generate_music, inputs=[music_prompt, music_lyrics], outputs=music_output)

demo.launch(share=True, debug=True)

