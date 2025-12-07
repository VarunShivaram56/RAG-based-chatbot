
# **RAG-Based RAG-Powered Course Navigator & Topic Guide (Local-First, Privacy-Preserved)**

This project is a **local-first Retrieval-Augmented Generation (RAG) teaching assistant** built using **Ollama**, **Llama 3.2**, **bge-m3 embeddings**, **Whisper transcription**, and **Streamlit UI**.
It allows users to ask questions based on a collection of educational videos.
The system retrieves relevant video segments, constructs prompts, and generates human-friendly answers along with timestamps.

Everything works locally — **no cloud APIs, no external dependencies, fully private**.

---

## 🚀 **Core Features**

* Local LLM inference using **Llama 3.2 (Ollama)**
* Embedding generation using **bge-m3**
* Video → Audio → Whisper transcription → Chunking pipeline
* Intelligent RAG retrieval using cosine similarity
* Automatic prompt generation from relevant video chunks
* Streamlit chatbot interface (`app.py`)
* All embeddings stored locally using **joblib**
* Output prompts and responses logged in `prompt.txt` & `response.txt`

---

# 📂 **Project Structure**

```
rag-project/
│
├── audios/                # Audio extracted from input videos (.mp3)
├── videos/                # Your raw input video files
├── jsons/                 # Whisper transcripts + chunked JSON files
│
├── app.py                 # Streamlit UI chatbot
├── process_incoming.py    # RAG retrieval + prompt construction + LLM inference
├── preprocess_json.py     # Converts chunked JSON → embeddings dataframe → embeddings.joblib
├── video_to_mp3.py        # Converts all videos → mp3
├── mp3_to_json.py         # Transcribes mp3 using Whisper → JSON
├── embeddings.joblib      # Saved embeddings used for retrieval
│
├── prompt.txt             # (Runtime generated) prompt sent to the LLM
├── response.txt           # (Runtime generated) model output
│
└── README.md              # Project documentation
```

---

# 📘 **How the System Works (Full Pipeline)**

This is the complete end-to-end workflow that powers your AI teaching assistant.

---

## **Step 1 — Collect Videos**

Place all your course videos inside the `videos/` folder.

```
videos/
    video1.mp4
    video2.mp4
    ...
```

---

## **Step 2 — Convert Videos to MP3**

Run:

```
python video_to_mp3.py
```

This extracts the audio from each video and stores it inside:

```
audios/
    video1.mp3
    video2.mp3
```

---

## **Step 3 — Transcribe Audio to JSON**

Run:

```
python mp3_to_json.py
```

This uses **Whisper** (locally) to generate subtitle-style JSON transcripts for each audio file.

Output goes into:

```
jsons/
    video1.json
    video2.json
```

---

## **Step 4 — Chunk Transcripts + Create Embeddings**

Run:

```
python preprocess_json.py
```

This script:

1. Reads the Whisper JSON transcript
2. Breaks it into meaningful subtitle chunks
3. Sends chunks to **Ollama bge-m3** to generate embeddings
4. Stores all chunks + embeddings in a single Pandas DataFrame
5. Saves it as:

```
embeddings.joblib
```

This file is your **RAG knowledge base**.

---

## **Step 5 — Query Processing (RAG + LLM Response)**

Whenever a user asks a question, `process_incoming.py`:

1. Embeds the user query using **bge-m3**
2. Computes cosine similarity against all stored chunk embeddings
3. Selects the **top 5 most relevant chunks**
4. Builds a detailed RAG prompt with:

   * video title
   * video number
   * timestamp (start/end)
   * text content
5. Sends the prompt to **Llama 3.2 (Ollama)** for final answer generation
6. Writes:

   * the prompt → `prompt.txt`
   * the answer → `response.txt`

---

## **Step 6 — Streamlit Chatbot UI**

Run:

```
streamlit run app.py
```

This launches your chatbot:

* User types a question
* RAG system finds relevant video segments
* LLM generates a human-friendly explanation
* User gets a reference to the exact video and timestamp

### UI Title:

**“varun's rag chatbot”**

The chatbot behaves just like a personalized course assistant.

---

# 🧠 **Technologies Used**

| Category      | Technology                       |
| ------------- | -------------------------------- |
| LLM           | Llama 3.2 via Ollama             |
| Embeddings    | bge-m3 (Ollama)                  |
| Transcription | Whisper                          |
| Retrieval     | Cosine similarity (scikit-learn) |
| Storage       | Pandas + Joblib                  |
| Frontend      | Streamlit chat UI                |
| Language      | Python                           |

---

# 📦 **Installation & Setup**

### Install dependencies:

```
pip install -r requirements.txt
```

*(Create this file if needed — I can generate one for you.)*

### Install Ollama models:

```
ollama pull llama3.2
ollama pull bge-m3
ollama pull whisper
```

---

# ▶️ **Run the Chatbot**

```
streamlit run app.py
```

---

# 🔮 **Future Improvements**

* Add GPU acceleration for Whisper + Llama
* Add support for PDFs or text documents
* Implement vector database (ChromaDB / FAISS)
* Add multi-video summary generation

---

# ❤️ **Author**

**Varun Shivaram**
Data Science Engineering Student • Full-Stack Developer  

