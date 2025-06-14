import os
import io
import gradio as gr
from dotenv import load_dotenv
import requests
from pydub import AudioSegment

os.makedirs('cache',exist_ok=True)
# Load environment variables
load_dotenv()
SHARVAM_API_KEY = os.getenv("SHARVAM_API_KEY")

# Sarvam API URLs
TRANS_API_URL = "https://api.sarvam.ai/speech-to-text-translate"
CHAT_API_URL = "https://api.sarvam.ai/v1/chat/completions"

# Headers
trans_headers = {
    "api-subscription-key": SHARVAM_API_KEY
}

chat_headers = {
    "api-subscription-key": SHARVAM_API_KEY,
    "Content-Type": "application/json",
}

data = {
    "model": "saaras:v2",
    "with_diarization": False
}

def split_audio(audio_path, chunk_duration_ms):
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    if len(audio) > chunk_duration_ms:
        for i in range(0, len(audio), chunk_duration_ms):
            chunks.append(audio[i:i + chunk_duration_ms])
    else:
        chunks.append(audio)
    return chunks

def translate_audio(audio_file_path, api_url, headers, data, progress, chunk_duration_ms=30 * 1000):
    chunks = split_audio(audio_file_path, chunk_duration_ms)
    responses = []

    for idx, chunk in  progress.tqdm(enumerate(chunks),total=len(chunks), desc="Translating Audio Chunks"):
        chunk_buffer = io.BytesIO()
        chunk.export(chunk_buffer, format="wav")
        chunk_buffer.seek(0)
        files = {'file': ('audiofile.wav', chunk_buffer, 'audio/wav')}

        try:
            response = requests.post(api_url, headers=headers, files=files, data=data)
            if response.status_code in (200, 201):
                response_data = response.json()
                transcript = response_data.get("transcript", "")
                responses.append({"transcript": transcript})
            else:
                print(f"Chunk {idx} failed with status code {response.status_code}")
        except Exception as e:
            print(f"Chunk {idx} error: {e}")
        finally:
            chunk_buffer.close()

    collated_transcript = " ".join([resp["transcript"] for resp in responses])
    return collated_transcript

def summarize_text(transcript):
    payload = {
        "model": "sarvam-m",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant who is also an expert meeting summarizer."},
            {"role": "user", "content": f"Summarize this meeting in bullet points, Do NOT USE MARKDOWN format: {transcript}"},
        ],
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 1000,
        "n": 1,
    }
    response = requests.post(CHAT_API_URL, headers=chat_headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print("Summary request failed:", response.status_code, response.text)
        return "Summary generation failed."

def process_audio(audio_file):
    audio_path = "cache/uploaded_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_file)

    transcript = translate_audio(audio_path, TRANS_API_URL, trans_headers, data, progress=gr.Progress())
    summary = summarize_text(transcript)

    # Save transcript and summary
    transcript_path = "cache/translated_output.txt"
    summary_path = "cache/summary_output.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    return transcript, summary, transcript_path, summary_path

# Gradio Interface
app = gr.Interface(
    fn=process_audio,
    inputs=gr.File(type="binary", label="Upload Audio File"),
    outputs=[
        gr.Textbox(label="Full Transcription"),
        gr.Textbox(label="Meeting Summary"),
        gr.File(label="Download Transcription"),
        gr.File(label="Download Summary")
    ],
    allow_flagging="never", # allow_flagging="never" hides the "Report a Bug" button
    title="Meeting Transcriber & Summarizer",
    description="Upload a long audio file. The app will transcribe and summarize it using Sarvam AI APIs."
)

app.launch()
