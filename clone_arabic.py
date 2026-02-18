import torch, soundfile as sf, numpy as np, os
from pydub import AudioSegment, effects
from qwen_tts import Qwen3TTSModel

# --- Configuration ---
INPUT_FILE = "inputs/sample_arabic.wav"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ai_cloned_arabic.mp3")
REF_TEXT = "مرحبا، هيدا تسجيل تجريبي. عم بسجل هيدا المقطع كرمال السيستم يلقط طريقتي بالحكي مية بالمية."

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
ref_audio = np.mean(data, axis=1) if len(data.shape) > 1 else data
sf.write("temp_ar.wav", ref_audio, sr_ref)

# --- Generate Clone (Loop-Killer Mode) ---
print("Generating Arabic (Lebanese) clone...")
with torch.inference_mode():
    wavs, sr = model.generate_voice_clone(
        text="مرحبا، معكم Assix. هيدا صوتي عم يحكي لبناني من قلب الـ دي جي إكس. السرعة خيالية!",
        ref_audio="temp_ar.wav",
        ref_text=REF_TEXT,
        language="auto",
        instruct="A warm, natural male voice with a clear Lebanese accent.",
        temperature=0.5,
        repetition_penalty=1.6,
        top_p=0.9
    )

# --- Export with Normalization ---
audio_data = wavs[0]
if hasattr(audio_data, 'cpu'):
    audio_data = audio_data.cpu().numpy()
audio_data = audio_data.flatten()

seg = AudioSegment((audio_data * 32767).astype(np.int16).tobytes(), frame_rate=sr, sample_width=2, channels=1)
effects.normalize(seg).export(OUTPUT_FILE, format="mp3", bitrate="192k")

os.remove("temp_ar.wav")
print(f"✔ Success! Saved to {OUTPUT_FILE}")