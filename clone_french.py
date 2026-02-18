import torch, soundfile as sf, numpy as np, os
from pydub import AudioSegment
from qwen_tts import Qwen3TTSModel

# --- Configuration ---
INPUT_FILE = "inputs/sample_french.wav"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ai_cloned_french.mp3")
REF_TEXT = "Bonjour, c'est un exemple. J'enregistre cet échantillon en français métropolitain pour garantir une prononciation claire et précise."

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
sf.write("temp_fr.wav", ref_audio, sr_ref)

# --- Generate Clone ---
print("Generating French clone...")
with torch.inference_mode():
    wavs, sr = model.generate_voice_clone(
        text="Voici une démonstration de la puissance de l'architecture Blackwell. L'accent est maintenant parfaitement stable et professionnel.",
        ref_audio="temp_fr.wav",
        ref_text=REF_TEXT,
        language="French",
        instruct="Standard Metropolitan French, clear and professional.",
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

os.remove("temp_fr.wav")
print(f"✔ Success! Saved to {OUTPUT_FILE}")