# ai-voice-clone-blackwell-aarch64

**A high-performance, multilingual voice cloning engine for English, French and Arabic (EN, FR, AR) optimized for NVIDIA Blackwell GPUs and ARM64 architecture.**



## 🌟 Overview

This repository provides a production-ready toolkit for zero-shot voice cloning using **Qwen3-TTS**. It is specifically engineered to leverage the **sm_121** compute capability and the **aarch64** instruction set found in the **NVIDIA DGX Spark**.

Unlike standard implementations, this engine features a custom **"Loop-Killer"** logic to stabilize Levantine Arabic dialects and prevents regional drift in French generation.

### Key Features
* **Blackwell Native:** Optimized for `bfloat16` and **FlashAttention-2** (via custom aarch64 builds).
* **Identity-First Cloning:** Prioritizes vocal timbre and resonance over generic regional accents.
* **Lebanese Dialect Stability:** Custom logic eliminates phonetic stuttering (e.g., "ya-ya-ya" loops) common in zero-shot Arabic cloning.
* **Cross-Lingual Prosody:** Prevents accent drift (e.g., Canadian French) by utilizing Metropolitan French anchors.

---

## 🚀 Performance on DGX Spark

| Metric | Value |
| :--- | :--- |
| **GPU Architecture** | NVIDIA Grace Blackwell (GB10/GB100) |
| **Precision** | Native `bfloat16` |
| **Inference Latency** | ~140ms (5s audio generation) |
| **Optimization** | FlashAttention-2 (sm_121) |

---

## 🛠️ Installation

### 1. System Dependencies
Ensure `ffmpeg` and `sox` are installed for audio processing:
```bash
sudo apt update && sudo apt install -y ffmpeg sox
```

### 2. Install Blackwell-Optimized Binaries (CRITICAL)
Standard `pip` installs for FlashAttention will fail on aarch64. You **must** install the pre-compiled wheels from my specialized binary repository:

**📦 Get Wheels Here:** [assix/flash-attention-blackwell-aarch64-wheels](https://github.com/assix/flash-attention-blackwell-aarch64-wheels)

```bash
# Example: Install the custom Blackwell FlashAttention build
pip install [https://github.com/assix/flash-attention-blackwell-aarch64-wheels/releases/download/v2.8.3/flash_attn-2.8.3+cu130sm121-cp310-cp310-linux_aarch64.whl](https://github.com/assix/flash-attention-blackwell-aarch64-wheels/releases/download/v2.8.3/flash_attn-2.8.3+cu130sm121-cp310-cp310-linux_aarch64.whl)
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎙️ Input Data

To clone a voice, you need a reference audio file.
1.  **Format:** `.wav` (16-bit PCM recommended).
2.  **Length:** 15-30 seconds.
3.  **Location:** Place your file in the `inputs/` folder.

*Included Samples:*
* `inputs/sample_english.wav`
* `inputs/sample_french.wav`
* `inputs/sample_arabic.wav`

---

## 💻 Usage

Run the language-specific pipelines from the root directory. The scripts automatically handle audio preprocessing and post-processing (normalization).

### 🇺🇸 English Clone
*Optimized for high-fidelity timbre and neutral accent.*
```bash
python src/clone_english.py
```
**Output:** `outputs/ai_cloned_english.mp3`

### 🇫🇷 French Clone
*Optimized for Metropolitan prosody.*
```bash
python src/clone_french.py
```
**Output:** `outputs/ai_cloned_french.mp3`

### 🇱🇧 Arabic (Lebanese) Clone
*Optimized with Loop-Killer logic for dialect stability.*
```bash
python src/clone_arabic.py
```
**Output:** `outputs/ai_cloned_arabic.mp3`

---

## 📂 Project Structure

```text
.
├── src/
│   ├── clone_english.py     # Main engine for English
│   ├── clone_french.py      # Main engine for French
│   └── clone_arabic.py      # Main engine for Lebanese Arabic
├── inputs/                  # Place reference audio here
├── outputs/                 # Generated audio appears here
├── requirements.txt         # Python dependencies
└── README.md                # Documentation
```

---

## ⚠️ Troubleshooting

**Issue:** The Lebanese model stutters (repeats the first syllable, e.g., "Ya... Ya...").
**Solution:**
1.  **Trim Silence:** Ensure your reference audio has no silence at the very beginning.
2.  **Change Start Token:** Do not start your target text with the same phoneme as your reference audio (e.g., if reference starts with "Ya", start target with "Marhaba").
3.  **Repetition Penalty:** The script `src/clone_arabic.py` is pre-configured with a penalty of `1.6`. Increase to `2.0` if looping persists.

---

## 🔒 License & Privacy

**License:** MIT
