import torch, soundfile as sf, numpy as np, os
from pydub import AudioSegment
from qwen_tts import Qwen3TTSModel

# --- Configuration ---
# Paths are relative to where you run the script
INPUT_FILE = "inputs/sample_english.wav"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ai_cloned_english.mp3")
REF_TEXT = "Hello, this is a sample input. I am recording this clear English sample to ensure the highest possible audio quality for the Blackwell system."

# --- Setup Directories ---
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found at: {INPUT_FILE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Initialize Blackwell Engine ---
print(f"Loading Qwen3-TTS on NVIDIA GB10 (Blackwell)...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base", 
    device_map="cuda", 
    dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2"
)

# --- Process Reference Audio ---
data, sr_ref = sf.read(INPUT_FILE)
# Convert to mono if stereo
ref_audio = np.mean(data, axis=1) if len(data.shape) > 1 else data
# Save temp mono file
sf.write("temp_en.wav", ref_audio, sr_ref)

# --- Generate Clone ---
print("Generating English clone...")
with torch.inference_mode():
    wavs, sr = model.generate_voice_clone(
        text="This is a demonstration of high-performance inference on the NVIDIA DGX Spark. The voice quality is crisp, natural, and highly responsive.",
        ref_audio="temp_en.wav",
        ref_text=REF_TEXT,
        language="English",
        instruct="Clear, professional American English speaker with high fidelity.",
        temperature=0.7,
        top_p=0.9
    )

# --- Export ---
audio_data = wavs[0]
if hasattr(audio_data, 'cpu'):
    audio_data = audio_data.cpu().numpy()
audio_data = audio_data.flatten()

AudioSegment(
    (audio_data * 32767).astype(np.int16).tobytes(), 
    frame_rate=sr, 
    sample_width=2, 
    channels=1
).export(OUTPUT_FILE, format="mp3", bitrate="192k")

os.remove("temp_en.wav")
print(f"✔ Success! Saved to {OUTPUT_FILE}")