# TEMPORARY workaround: set before heavy libs if you still see OpenMP conflicts.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import shutil
import requests
import pyttsx3
import re
import sounddevice as sd
import numpy as np
import tempfile
import wave
import glob
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document

# Optional image captioning imports (wrapped)
try:
    from transformers import pipeline
    HAS_CAPTION = True
except Exception:
    HAS_CAPTION = False

# ====== Load environment ======
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)
embeddings = OpenAIEmbeddings(api_key=API_KEY)

# ====== Config ======
JSON_PATH = "logs/thread_log_queue_ETqEJ_json_report.json"
IMAGES_DIR = "executionScreens"         
OUTPUT_IMAGES_DIR = "queried_images"     # where we save images when a query references them
DEFAULT_RECORD_DURATION = 6              # seconds (initial voice and subsequent voice queries)
HISTORY_MAX_MESSAGES = 6                 # keep last N messages in conversation history
RETRIEVER_K = 3                          # number of results to retrieve per retriever

# ====== Utilities: file-safe names and save images ======
def safe_filename(name):
    name = os.path.basename(name)
    name = name.replace(":", "_").replace(" ", "_")
    return name

def save_images_to_folder(image_paths, dest_dir=OUTPUT_IMAGES_DIR, images_dir=IMAGES_DIR):
    os.makedirs(dest_dir, exist_ok=True)
    saved = []
    missing = []
    for ip in image_paths:
        ip_str = str(ip)
        try:
            if ip_str.lower().startswith("http://") or ip_str.lower().startswith("https://"):
                # download
                resp = requests.get(ip_str, stream=True, timeout=15)
                if resp.status_code == 200:
                    fname = safe_filename(ip_str)
                    dest_path = os.path.join(dest_dir, fname)
                    base, ext = os.path.splitext(dest_path)
                    counter = 1
                    while os.path.exists(dest_path):
                        dest_path = f"{base}_{counter}{ext}"
                        counter += 1
                    with open(dest_path, "wb") as f:
                        for chunk in resp.iter_content(1024 * 8):
                            f.write(chunk)
                    saved.append(dest_path)
                else:
                    missing.append(ip_str)
            else:
                # local path attempt
                if os.path.exists(ip_str):
                    src = ip_str
                else:
                    candidate = os.path.join(images_dir, os.path.basename(ip_str))
                    if os.path.exists(candidate):
                        src = candidate
                    else:
                        src = None

                if src:
                    fname = safe_filename(src)
                    dest_path = os.path.join(dest_dir, fname)
                    base, ext = os.path.splitext(dest_path)
                    counter = 1
                    while os.path.exists(dest_path):
                        dest_path = f"{base}_{counter}{ext}"
                        counter += 1
                    shutil.copy2(src, dest_path)
                    saved.append(dest_path)
                else:
                    missing.append(ip_str)
        except Exception:
            missing.append(ip_str)
    return saved, missing

# ====== Audio recording & Whisper transcription (STT) ======
def record_audio(filename, duration=DEFAULT_RECORD_DURATION, samplerate=16000):
    print(f"🎤 Recording for {duration} sec... Speak now!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    print(f"✅ Recording saved to {filename}.")

def transcribe_with_whisper(filename):
    try:
        with open(filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file
            )
        return transcript.text.strip()
    except Exception as e:
        print("❌ Transcription failed:", e)
        return ""

# ====== Image helpers & captioning ======
def find_images_by_step_number(step_number, images_dir=IMAGES_DIR):
    patterns = [
        os.path.join(images_dir, f"*{step_number}*.*"),
        os.path.join(images_dir, f"step*{step_number}*.*"),
        os.path.join(images_dir, f"*_{step_number}*.*"),
    ]
    found = []
    for p in patterns:
        found.extend(glob.glob(p))
    found = sorted(list(dict.fromkeys(found)))
    return found

def init_image_captioner():
    if not HAS_CAPTION:
        return None
    try:
        captioner = pipeline("image-captioning", model="Salesforce/blip-image-captioning-base")
        return captioner
    except Exception:
        return None

def caption_image(image_path, captioner=None):
    """
    Always defined. If captioner is provided, try to generate a caption.
    If captioning fails or captioner is None, return the basename of the image.
    """
    if captioner:
        try:
            out = captioner(image_path, max_length=64, truncation=True)
            if isinstance(out, list) and len(out) > 0:
                # different transformers versions return different keys
                if "caption" in out[0]:
                    return out[0]["caption"]
                if "generated_text" in out[0]:
                    return out[0]["generated_text"]
            # fallback to str representation
        except Exception:
            pass
    return os.path.basename(image_path)

# ====== Parse JSON: produce both ALL steps and FAILED steps ======
def parse_log_file(json_path, images_dir=IMAGES_DIR):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "test_logs" not in data or not isinstance(data["test_logs"], list):
        raise ValueError("Error: 'test_logs' key not found or not a list.")

    steps = data["test_logs"]
    all_entries = []
    failed_entries = []

    def extract_image_paths(value):
        collected = []
        if not value:
            return collected
        if isinstance(value, str):
            collected.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    collected.append(item)
        elif isinstance(value, dict):
            for v in value.values():
                if isinstance(v, str):
                    collected.append(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str):
                            collected.append(item)
        return collected

    for step in steps:
        step_number = step.get("step_number", "N/A")
        step_command = step.get("step_command", "") or ""
        images = []
        for key in ("images", "image", "step_image", "step_images", "attachments"):
            if key in step:
                 imgs = extract_image_paths(step.get(key))
                 images.extend(imgs[:4])  # limit to first 3 images here
        for sub_step in step.get("step_logs", []):
            for key in ("images", "image", "step_image", "attachments"):
                if key in sub_step:
                    imgs = extract_image_paths(sub_step.get(key))
                    images.extend(imgs[:2])  # limit to first 2 images per sub-step

        normalized = []
        for p in images:
            if not p:
                continue
            p = str(p).strip()
            if os.path.isabs(p) or os.path.exists(p):
                normalized.append(p)
            else:
                trial = os.path.join(images_dir, p)
                if os.path.exists(trial):
                    normalized.append(trial)
                else:
                    normalized.append(p)
        images = list(dict.fromkeys(normalized))
        # if not images:
            # images = find_images_by_step_number(step_number)

        # build short textual context
        summary_texts = []
        for sub_step in step.get("step_logs", []):
            msg = sub_step.get("message", {})
            if isinstance(msg, dict):
                txt = msg.get("llm_response") or msg.get("response")
                if txt:
                    summary_texts.append(str(txt))
            elif isinstance(msg, str):
                summary_texts.append(msg)
        combined_text = f"Step {step_number}: {step_command}"
        if summary_texts:
            combined_text += " | " + " ".join(summary_texts)

        result_overall = "SUCCESS"
        for sub_step in step.get("step_logs", []):
            if sub_step.get("result") == "FAILURE":
                result_overall = "FAILURE"
                break

        entry = {
            "step_number": step_number,
            "step_command": step_command,
            "status": result_overall,
            "text": combined_text,
            "images": images
        }
        all_entries.append(entry)
        if result_overall == "FAILURE":
            failed_entries.append(entry)

    return all_entries, failed_entries

    all_entries, failed_entries = parse_log_file(JSON_PATH, images_dir=IMAGES_DIR)
    print("=== Checking images in all entries ===")
    for e in all_entries:
        print(f"Step {e['step_number']} images:", e['images'])


# ====== Build retriever over ALL steps (so queries can be about anything) ======
def build_retrievers_from_all(entries, embeddings_obj):
    docs = [Document(page_content=e["text"], metadata={"id": idx, "step_number": e["step_number"], "status": e["status"], "images": e["images"]})
            for idx, e in enumerate(entries)]
    vectorstore = FAISS.from_documents(docs, embeddings_obj)
    dense_retriever = vectorstore.as_retriever(search_kwargs       ={"k": RETRIEVER_K})
    sparse_retriever = BM25Retriever.from_documents(docs)
    sparse_retriever.k = RETRIEVER_K
    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble

# ====== TTS (pyttsx3) ======
def speak_text(text):
    if not text or not text.strip():
        return
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    engine.setProperty("volume", 1.0)
    voices = engine.getProperty("voices")
    if voices:
        engine.setProperty("voice", voices[0].id)
    max_chunk = 800
    start = 0
    while start < len(text):
        chunk = text[start:start+max_chunk]
        engine.say(chunk)
        engine.runAndWait()
        start += max_chunk

# ====== Build textual chunks (with optional captions) for display/indexing ======
def build_entry_texts(entries, captioner=None):
    out = []
    for e in entries:
        images = e.get("images", []) or []
        captions = []
        for img in images:
            if os.path.exists(img) and captioner:
                captions.append(caption_image(img, captioner))
            else:
                captions.append(os.path.basename(img))
        image_section = ""
        if images:
            pairs = [f"{os.path.basename(p)} (caption: {c})" for p, c in zip(images, captions)]
            image_section = "Images: " + "; ".join(pairs)
        text = e.get("text", "")
        if image_section:
            text = f"{text}\n{image_section}"
        out.append({"text": text, "images": images, "step_number": e.get("step_number"), "status": e.get("status")})
    return out

# ====== Retriever safe call (invoke or fallback) ======
def retrieve_docs(retriever, query):
    try:
        maybe = retriever.invoke(query)
        if isinstance(maybe, list):
            return maybe
    except Exception:
        pass
    try:
        return retriever.get_relevant_documents(query)
    except Exception as e:
        print("❌ Retriever call failed:", e)
        return []

# ====== LLM helpers with history trimming ======
def trim_history(history, max_messages=HISTORY_MAX_MESSAGES):
    if not history:
        return history
    return history[-max_messages:]

def ask_for_summary_of_failed(failed_entries, conversation_history=None):
    if not failed_entries:
        return "Good news — there are no failed steps in the logs."
    context = "\n\n".join([f"Step {e['step_number']}: {e['step_command']} — Reason: {e.get('text','')}" for e in failed_entries])
    system_prompt = (
        "You are a helpful QA assistant. Write a short (3-6 sentences), human-friendly, non-technical summary(avoid OCR based any information, avoid technical jargon unless necessary) "
        "of the following failed test steps. Avoid unnecessary symbols and technical jargon. If images are present, "
        "mention them briefly and what they likely show. Keep language simple for a non-technical reader."
    )
    messages = [{"role":"system", "content": system_prompt}]
    if conversation_history:
        messages.extend(trim_history(conversation_history))
    messages.append({"role":"user", "content": f"Failed steps context:\n{context}"})
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Summary generation failed:", e)
        return "Sorry, I couldn't generate the summary right now."

def answer_query_with_context(query, contexts, conversation_history=None):
    system_prompt = (
        "You are a helpful QA assistant. Answer the user's question clearly and in natural, human-friendly language. "
        "Use the provided contexts (which may include image filenames and captions) to support your answer. "
        "Keep it short and precise; do not include image file paths in the main answer — the program will save images separately."
        "Avoid OCR based any information, avoid technical jargon unless necessary."
    )
    messages = [{"role":"system", "content": system_prompt}]
    if conversation_history:
        messages.extend(trim_history(conversation_history))
    messages.append({"role":"user", "content": f"Context:\n{contexts}\n\nUser question:\n{query}"})
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Answer generation failed:", e)
        return "Sorry, I couldn't answer that right now."

# ====== Main flow ======
if __name__ == "__main__":
    # Parse log file -> all entries and failed_entries
    all_entries, failed_entries = parse_log_file(JSON_PATH, images_dir=IMAGES_DIR)

    # optional captioner (if user installed transformers)
    captioner = init_image_captioner() if HAS_CAPTION else None

    # Build presentation texts (used for indexing & display)
    entry_dicts = build_entry_texts(all_entries, captioner=captioner)   # used for indexing ALL steps
    failed_entry_dicts = build_entry_texts(failed_entries, captioner=captioner)  # used for summary only

    # Build retriever over ALL steps (so queries can be about anything)
    hybrid_retriever = build_retrievers_from_all(all_entries, embeddings)

    conversation_history = []

    # INITIAL VOICE PROMPT: ask user (via mic) whether they want the summary.
    print("Startup: Say 'summary' (or 'give summary') to hear a summary of failed steps, or ask a question. Waiting for ~6s...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        start_wav = tmp.name
    try:
        record_audio(start_wav, duration=DEFAULT_RECORD_DURATION)
        initial_spoken = transcribe_with_whisper(start_wav)
    finally:
        try:
            os.remove(start_wav)
        except Exception:
            pass

    if initial_spoken:
        print("🗣 You said:", initial_spoken)
    else:
        print("No initial speech detected.")

    # Detect whether initial command asks for a summary
    if initial_spoken and re.search(r"\bsummary\b|\bsummarize\b|\bgive (me )?the summary\b", initial_spoken, re.IGNORECASE):
        summary = ask_for_summary_of_failed(failed_entries, conversation_history=conversation_history)
        print("\n--- Summary (failed steps only) ---\n")
        print(summary)
        speak_text(summary)
        conversation_history.append({"role":"assistant", "content": summary})
    elif initial_spoken and initial_spoken.strip():
        # treat initial transcript as first query
        query = initial_spoken
        results = retrieve_docs(hybrid_retriever, query)
        contexts = "\n\n".join([r.page_content for r in results]) if results else "No relevant context found."
        answer = answer_query_with_context(query, contexts, conversation_history=conversation_history)
        print("\n--- Answer (initial) ---\n")
        print(answer)

        # Simple heuristic to detect no relevant context
        
        referenced_images = []
        for r in results[:3]:
            imgs = r.metadata.get("images") if isinstance(r.metadata, dict) else None
            if imgs:
                for ip in imgs:
                    if ip not in referenced_images:
                        referenced_images.append(ip)
        referenced_images = referenced_images[:5]  # limit max 5 images

        if referenced_images:
            saved, missing = save_images_to_folder(referenced_images, dest_dir=OUTPUT_IMAGES_DIR, images_dir=IMAGES_DIR)
            print(f"\nSaved {len(saved)} images to '{OUTPUT_IMAGES_DIR}/'.")
            if missing:
                print(f"{len(missing)} image(s) could not be found/downloaded:")
                for m in missing:
                    print(" -", m)
        else:
            print("\nNo relevant images found or saved for this query.")


        speak_text(answer)
        conversation_history.append({"role":"user", "content": query})
        conversation_history.append({"role":"assistant", "content": answer})
    else:
        # no initial speech; proceed silently
        pass

    # Interactive loop (typed or voice queries)
    print("\nReady. Type 'summary' to get spoken summary, say 'voice' to ask by speaking, or type a question. Type 'exit' to quit.")
    while True:
        cmd = input("\nEnter command ('summary' / 'voice' / any question / 'exit'): ").strip()
        if not cmd:
            continue
        if cmd.lower() == "exit":
            break

        if cmd.lower() == "summary":
            summary = ask_for_summary_of_failed(failed_entries, conversation_history=conversation_history)
            print("\n--- Summary (failed steps only) ---\n")
            print(summary)
            speak_text(summary)
            conversation_history.append({"role":"user", "content":"Give me the summary"})
            conversation_history.append({"role":"assistant", "content":summary})
            continue

        if cmd.lower() == "voice":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            try:
                record_audio(wav_path, duration=DEFAULT_RECORD_DURATION)
                spoken = transcribe_with_whisper(wav_path)
            finally:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

            print(f"\n🗣 You said: {spoken}")
            if not spoken or not spoken.strip():
                print("No speech recognized, try again.")
                continue

            if re.search(r"\bsummary\b|\bsummarize\b|\bgive (me )?the summary\b", spoken, re.IGNORECASE):
                summary = ask_for_summary_of_failed(failed_entries, conversation_history=conversation_history)
                print("\n--- Summary (failed steps only) ---\n")
                print(summary)
                speak_text(summary)
                conversation_history.append({"role":"user", "content": spoken})
                conversation_history.append({"role":"assistant", "content": summary})
                continue

            query = spoken
            results = retrieve_docs(hybrid_retriever, query)
            contexts = "\n\n".join([r.page_content for r in results]) if results else "No relevant context found."
            answer = answer_query_with_context(query, contexts, conversation_history=conversation_history)
            print("\n--- Answer ---\n")
            print(answer)

            
            referenced_images = []
            for r in results[:3]:
                imgs = r.metadata.get("images") if isinstance(r.metadata, dict) else None
                if imgs:
                    for ip in imgs:
                        if ip not in referenced_images:
                            referenced_images.append(ip)
            referenced_images = referenced_images[:5]  # limit max 5 images

            if referenced_images:
                saved, missing = save_images_to_folder(referenced_images, dest_dir=OUTPUT_IMAGES_DIR, images_dir=IMAGES_DIR)
                print(f"\nSaved {len(saved)} images to '{OUTPUT_IMAGES_DIR}/'.")
                if missing:
                    print(f"{len(missing)} image(s) could not be found/downloaded:")
                    for m in missing:
                        print(" -", m)
            else:
                print("\nNo relevant images found or saved for this query.")


            speak_text(answer)
            conversation_history.append({"role":"user", "content": query})
            conversation_history.append({"role":"assistant", "content": answer})
            continue

        # Default: typed query
        query = cmd
        results = retrieve_docs(hybrid_retriever, query)
        contexts = "\n\n".join([r.page_content for r in results]) if results else "No relevant context found."
        

        answer = answer_query_with_context(query, contexts, conversation_history=conversation_history)
        print("\n--- Answer ---\n")
        print(answer)

        
        
        referenced_images = []
        for r in results[:3]:
            imgs = r.metadata.get("images") if isinstance(r.metadata, dict) else None
            if imgs:
                for ip in imgs:
                        if ip not in referenced_images:
                            referenced_images.append(ip)
            referenced_images = referenced_images[:3] 


        speak_text(answer)
        conversation_history.append({"role":"user", "content": query})
        conversation_history.append({"role":"assistant", "content": answer})

