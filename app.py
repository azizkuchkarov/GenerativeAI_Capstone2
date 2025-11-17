import os
import json
from datetime import datetime

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
import requests


# ============================================================
#  Project: Voice-to-Image Generator (Capstone Project 2)
#  Author: Aziz Kuchkarov
#  Description:
#       This application converts a user's voice input into an
#       AI-generated image by using a Hugging Face based pipeline:
#           1. Speech-to-Text (Whisper)
#           2. Prompt builder (Python logic)
#           3. Prompt-to-Image (Stable Diffusion 3)
#
#       This version does NOT use OpenAI and works entirely on
#       Hugging Face Inference API for ASR and image generation.
#       The prompt generation step is implemented locally in Python.
# ============================================================


# -----------------------------
# Load environment variables
# -----------------------------

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

# Default models (can be overridden by .env variables)
HF_ASR_MODEL_ID = os.getenv("HF_ASR_MODEL_ID", "openai/whisper-large-v3")
HF_IMAGE_MODEL_ID = os.getenv(
    "HF_IMAGE_MODEL_ID",
    "stabilityai/stable-diffusion-3-medium-diffusers"
)

if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY is missing in .env file. Add your Hugging Face token.")


# Common API headers for different content types
HF_HEADERS_JSON = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Accept": "application/json",
}
HF_HEADERS_AUDIO = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Accept": "application/json",
    "Content-Type": "audio/wav",
}
HF_HEADERS_IMAGE = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Accept": "image/png",
}


# -----------------------------
# Streamlit App UI
# -----------------------------

st.set_page_config(
    page_title="Voice → Image (Hugging Face)",
    page_icon="🎨",
    layout="centered",
)

st.title("🎙️ Voice → 🖼️ Image Generator (Capstone Project 2)")
st.write(
    """
    This application was developed as part of **Capstone Project 2** by **Aziz Kuchkarov**.

    It performs the following pipeline:

    1. **Speech-to-Text** using Whisper on Hugging Face  
    2. **Prompt building** in Python based on the transcript and a style hint  
    3. **Prompt-to-Image** using Stable Diffusion 3 on Hugging Face  
    """
)

# Local runtime log storage
if "logs" not in st.session_state:
    st.session_state.logs = []


def log(message: str):
    """Utility function to log events both to console and the UI."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    st.session_state.logs.append(entry)


# -----------------------------
# Sidebar configuration
# -----------------------------

st.sidebar.header("Model & Style Configuration")

st.sidebar.write("**ASR Model (Speech → Text)**")
st.sidebar.code(HF_ASR_MODEL_ID)

st.sidebar.write("**Image Model (Prompt → Image)**")
st.sidebar.code(HF_IMAGE_MODEL_ID)

default_style = "high quality, detailed, 4k, digital art"
style_hint = st.sidebar.text_input(
    "Style hint for image generation",
    value=default_style,
    help="This text will be appended to the image prompt.",
)

st.sidebar.markdown("---")
st.sidebar.write("All remote calls are made via Hugging Face Inference API.")


# -----------------------------
# Audio Recording Section
# -----------------------------

st.subheader("1️⃣ Record Your Voice")
st.write("Click the button below, speak your request, and click again to finish recording.")

audio_bytes = audio_recorder(
    pause_threshold=2.0,
    sample_rate=41_000,
    text="Click to Record / Stop",
)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    st.success("Voice recorded successfully. Proceed to the next step.")
else:
    st.info("Waiting for audio input...")


# ============================================================
#  Hugging Face Pipeline Functions
# ============================================================

def hf_asr_transcribe(audio_data: bytes) -> str:
    """
    Step 1: Convert recorded audio to text using HF Whisper model.
    """
    log(f"Sending audio to HF ASR model: {HF_ASR_MODEL_ID}")

    url = f"https://router.huggingface.co/hf-inference/models/{HF_ASR_MODEL_ID}"
    response = requests.post(url, headers=HF_HEADERS_AUDIO, data=audio_data, timeout=300)

    if response.status_code != 200:
        log(f"ASR error {response.status_code}: {response.text[:300]}")
        raise RuntimeError(f"HF ASR error {response.status_code}")

    output = response.json()

    # Whisper returns either {"text": "..."} or a direct string
    if isinstance(output, dict) and "text" in output:
        text = output["text"]
    elif isinstance(output, str):
        text = output
    else:
        text = json.dumps(output)

    text = (text or "").strip()
    log(f"Transcription result: {text}")
    return text


def build_prompt_from_transcript(transcript: str, style: str) -> str:
    """
    Step 2: Build a clean, single-sentence image prompt from the transcript.
    This function is intentionally simple and deterministic.

    The goal is to:
      - Keep the original user intent
      - Add a bit of structure and style
      - Produce a description that works well with Stable Diffusion
    """
    log("Building image prompt locally from transcript and style hint.")

    # Basic normalization
    base_text = transcript.strip()
    if not base_text.endswith("."):
        base_text += "."

    prompt = (
        f"Create a detailed, visually appealing illustration of the following scene: "
        f"{base_text} "
        f"The image should follow this style: {style}. "
        f"Focus on clarity, composition, lighting, and overall aesthetics."
    )

    log(f"Final image prompt: {prompt}")
    return prompt


def hf_generate_image(prompt: str) -> bytes:
    """
    Step 3: Generate an image from the final prompt using Stable Diffusion 3.
    """
    log(f"Generating image using HF model: {HF_IMAGE_MODEL_ID}")

    url = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL_ID}"
    payload = {"inputs": prompt}

    response = requests.post(url, headers=HF_HEADERS_IMAGE, json=payload, timeout=600)

    if response.status_code != 200:
        log(f"Image error {response.status_code}: {response.text[:300]}")
        raise RuntimeError(f"HF Image generation error {response.status_code}")

    return response.content


# ============================================================
#  Main Voice → Image Pipeline
# ============================================================

st.subheader("2️⃣ Convert & Generate")

if st.button("🚀 Generate Image from Voice", type="primary"):
    if not audio_bytes:
        st.error("Please record your voice first.")
    else:
        try:
            # Step 1: Audio → Text
            with st.spinner("Transcribing voice with Whisper (Hugging Face)..."):
                transcript = hf_asr_transcribe(audio_bytes)
            st.markdown("**Transcript:**")
            st.code(transcript)

            # Step 2: Transcript → Image Prompt (local logic)
            with st.spinner("Building image prompt locally..."):
                image_prompt = build_prompt_from_transcript(transcript, style_hint)
            st.markdown("**Generated Image Prompt:**")
            st.code(image_prompt)

            # Step 3: Prompt → Image
            with st.spinner("Generating image with Stable Diffusion 3 (Hugging Face)..."):
                img_bytes = hf_generate_image(image_prompt)

            # Step 4: Display Result
            st.subheader("3️⃣ Resulting Image")
            st.image(
                img_bytes,
                caption="AI-generated image based on your voice",
                use_column_width=True
            )

            # Download button
            filename = f"voice2image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            st.download_button(
                "💾 Download Image",
                data=img_bytes,
                file_name=filename,
                mime="image/png"
            )

            st.success("Image generation completed successfully!")

        except Exception as e:
            log(str(e))
            st.error(f"An error occurred: {e}")


# -----------------------------
# Logs Section
# -----------------------------

st.markdown("---")
st.subheader("📜 Logs (Last 50 Events)")
for line in reversed(st.session_state.logs[-50:]):
    st.text(line)
