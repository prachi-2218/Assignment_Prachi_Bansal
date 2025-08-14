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
        step_id = step.get("step_id", step_number)  # use step_number if no explicit id
        step_command = step.get("step_command", "") or ""
        images = []

        # Collect images from step level
        for key in ("images", "image", "step_image", "step_images", "attachments"):
            if key in step:
                imgs = extract_image_paths(step.get(key))
                images.extend(imgs[:4])

        # Collect images from sub-step level
        for sub_step in step.get("step_logs", []):
            for key in ("images", "image", "step_image", "attachments"):
                if key in sub_step:
                    imgs = extract_image_paths(sub_step.get(key))
                    images.extend(imgs[:2])

        # Normalize image paths
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

        # Text summary from messages
        summary_texts = []
        retry_count = 0
        final_error = None
        result_overall = "SUCCESS"

        for sub_step in step.get("step_logs", []):
            msg = sub_step.get("message", {})
            if isinstance(msg, dict):
                txt = msg.get("llm_response") or msg.get("response")
                if txt:
                    summary_texts.append(str(txt))
            elif isinstance(msg, str):
                summary_texts.append(msg)

            if sub_step.get("result") == "FAILURE":
                retry_count += 1
                final_error = (msg.get("llm_response") if isinstance(msg, dict) else msg) or "Unknown error"
                result_overall = "FAILURE"

            elif sub_step.get("result") == "SUCCESS" and result_overall == "FAILURE":
                result_overall = "SUCCESS"

        combined_text = f"Step {step_number}: {step_command}"
        if summary_texts:
            combined_text += " | " + " ".join(summary_texts)

        entry = {
            "step_number": step_number,
            "step_id": step_id,
            "step_command": step_command,
            "status": result_overall,
            "retry_count": retry_count,
            "final_error": final_error,
            "text": combined_text,
            "images": images
        }
        all_entries.append(entry)
        if entry["status"] == "FAILURE":
            failed_entries.append(entry)

    return all_entries, failed_entries


import re

def find_step_by_query(query, all_entries):
    match = re.search(r"\bstep(?:\s+number)?\s*(\d+)\b", query, re.IGNORECASE)
    if match:
        step_id = match.group(1)
        for entry in all_entries:
            if str(entry["step_id"]) == step_id or str(entry["step_number"]) == step_id:
                return entry
    return None



# === Summary printing ===
all_entries, failed_entries = parse_log_file(JSON_PATH, images_dir=IMAGES_DIR)

# print("\n--- Step Execution Summary ---")
for entry in all_entries:
    step_info = f"Step {entry['step_number']} ({entry['step_command']})"
    status_info = f"Status: {entry['status']}"
    retry_info = f"Retries: {entry['retry_count']}"
    error_info = f"Final error: {entry['final_error']}" if entry['final_error'] else ""
    # print(f"{step_info} | {status_info} | {retry_info} {error_info}")

if failed_entries:
    # print("\n--- Failed Steps ---")
    for entry in failed_entries:
        # print(f"Step {entry['step_number']}: {entry['step_command']} "
            f"(Retries: {entry['retry_count']}) - Error: {entry['final_error']}"
else:
    print("\nGood news — no steps failed in the logs!")



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
def ask_model_about_image(client, image_path, prompt, model="gpt-4o"):
    """Send an image + prompt to OpenAI and return the model's text."""
    import base64, mimetypes, os

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    mime = mimetypes.guess_type(image_path)[0] or "image/png"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                ]
            }
        ],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def analyze_images_and_get_texts(image_paths, client, model="gpt-4o", prompt_extra="Please describe the image and list any readable text. Be concise."):
    analyses = []
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

    for p in image_paths:
        if not os.path.exists(p):
            print(f"[WARN] Skipping missing image: {p}")
            continue
        try:
            # Keep a local copy in OUTPUT_IMAGES_DIR (optional)
            dest = os.path.join(OUTPUT_IMAGES_DIR, os.path.basename(p))
            if os.path.abspath(p) != os.path.abspath(dest):
                shutil.copy2(p, dest)

            resp_text = ask_model_about_image(client, dest, prompt_extra, model=model)
            print(f"Analysis for {os.path.basename(dest)}:\n{resp_text}\n")
            analyses.append(resp_text)
        except Exception as e:
            print(f"Error analyzing {p}: {e}")
            analyses.append(f"[Error analyzing {p}] {e}")

    return analyses


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


def answer_query_with_context(query, contexts, conversation_history=None, referenced_images=None):
    """
    Uses text contexts (from retriever) and ONLY images referenced by the retriever
    result metadata. Downloads/copies to OUTPUT_IMAGES_DIR, then analyzes just those.
    """
    # 1) Prepare image context from referenced images
    image_context_block = ""
    if referenced_images:
        # Ensure local copies for URLs and relative paths
        saved, missing = save_images_to_folder(
            referenced_images,
            dest_dir=OUTPUT_IMAGES_DIR,
            images_dir=IMAGES_DIR
        )
        if missing:
            print(f"[INFO] {len(missing)} image(s) missing or not downloadable. Skipping: {missing[:3]}...")

        if saved:
            try:
                image_texts = analyze_images_and_get_texts(
                    saved, client, model="gpt-4o",
                    prompt_extra="Describe key UI elements and any visible status/alerts. Be concise."
                )
                image_context_block = "\n\n".join(
                    [f"Image {i+1} analysis:\n{txt}" for i, txt in enumerate(image_texts)]
                )
            except Exception as e:
                print(f"[WARN] Image analysis failed: {e}")

    # 2) Merge textual + image context
    combined_context = f"{contexts}\n\n{image_context_block}" if image_context_block else contexts

    # 3) Build system message
    system_prompt = (
        "You are a helpful QA assistant. Answer the user's question clearly and in natural, human-friendly language. "
        "Use the provided contexts (which may include image analyses) to support your answer. "
        "Keep it short and precise; do not include image file paths in the main answer. "
        "Avoid technical jargon unless necessary."
    )

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(trim_history(conversation_history))
    messages.append({"role": "user", "content": f"Context:\n{combined_context}\n\nUser question:\n{query}"})

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
    
def collect_referenced_images(results, limit=4):
    refs = []
    for r in results:
        imgs = r.metadata.get("images") if isinstance(r.metadata, dict) else None
        if imgs:
            for ip in imgs:
                if ip not in refs:
                    refs.append(ip)
        if len(refs) >= limit:
            break
    return refs[:limit]

# ====== Main flow ======
if __name__ == "__main__":
    # Parse log file -> all entries and failed_entries
    all_entries, failed_entries = parse_log_file(JSON_PATH, images_dir=IMAGES_DIR)
    
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
        # Step-specific check
        step_entry = find_step_by_query(query, all_entries)

        if step_entry:
            # Only use this step's text & images
            contexts = step_entry["text"]
            referenced_images = step_entry["images"]
        else:
            # Normal retrieval flow
            results = retrieve_docs(hybrid_retriever, query)
            contexts = "\n\n".join([r.page_content for r in results]) if results else "No relevant context found."
            referenced_images = collect_referenced_images(results, limit=4)

        answer = answer_query_with_context(
            query, contexts,
            conversation_history=conversation_history,
            referenced_images=referenced_images
        )

        print("\n--- Answer (initial) ---\n")
        print(answer)
        

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
            # Step-specific check
            step_entry = find_step_by_query(query, all_entries)

            if step_entry:
                # Only use this step's text & images
                contexts = step_entry["text"]
                referenced_images = step_entry["images"]
            else:
                # Normal retrieval flow
                results = retrieve_docs(hybrid_retriever, query)
                contexts = "\n\n".join([r.page_content for r in results]) if results else "No relevant context found."
                referenced_images = collect_referenced_images(results, limit=4)

            answer = answer_query_with_context(
                query, contexts,
                conversation_history=conversation_history,
                referenced_images=referenced_images
            )

            print("\n--- Answer ---\n")
            print(answer)
            speak_text(answer)
            conversation_history.append({"role":"user", "content": query})
            conversation_history.append({"role":"assistant", "content": answer})
            continue

        # Default: typed query
        query = cmd
        # Step-specific check
        step_entry = find_step_by_query(query, all_entries)

        if step_entry:
            # Only use this step's text & images
            contexts = step_entry["text"]
            referenced_images = step_entry["images"]
        else:
            # Normal retrieval flow
            results = retrieve_docs(hybrid_retriever, query)
            contexts = "\n\n".join([r.page_content for r in results]) if results else "No relevant context found."
            referenced_images = collect_referenced_images(results, limit=4)

        answer = answer_query_with_context(
            query, contexts,
            conversation_history=conversation_history,
            referenced_images=referenced_images
        )

        print("\n--- Answer ---\n")
        print(answer)
        speak_text(answer)
        conversation_history.append({"role":"user", "content": query})
        conversation_history.append({"role":"assistant", "content": answer})


        
