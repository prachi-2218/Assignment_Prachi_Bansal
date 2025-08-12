
This Python application allows you to interact with automated test logs using **voice** or **text queries**.  
It uses **speech-to-text**, **language models**, and optional **image captioning** to answer questions, summarize failed steps, and retrieve related images.

## Features

- **Voice and Text Interaction** — Ask questions or request summaries via microphone or terminal input.
- **Failed Step Summaries** — Generates easy-to-understand summaries of failed test steps.
- **Image Retrieval** — Finds and copies relevant screenshots or attachments for reference.
- **Optional Image Captioning** — Uses HuggingFace Transformers (`blip-image-captioning`) to describe images.
- **Hybrid Retrieval System** — Combines dense (FAISS) and sparse (BM25) search for relevant log entries.
- **Text-to-Speech Output** — Reads summaries and answers aloud.
- **Customizable Configuration** — Set durations, directories, and retrieval parameters in one place.

