import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests

st.set_page_config(page_title="varun's rag chatbot", layout="centered")

st.title("varun's rag chatbot")
st.caption("Ask questions related to your Sigma web development course 📚")

# ---- Load embeddings once ----
@st.cache_resource
def load_embeddings():
    df = joblib.load("embeddings.joblib")
    return df

df = load_embeddings()

# ---- Helper functions ----
def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    r.raise_for_status()
    embedding = r.json()["embeddings"]
    return embedding

def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            # "model": "deepseek-r1",
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )
    r.raise_for_status()
    response = r.json()
    return response

def build_prompt(incoming_query, df, top_results=5):
    # Create embedding for question
    question_embedding = create_embedding([incoming_query])[0]

    # Compute similarity with stored embeddings
    similarities = cosine_similarity(
        np.vstack(df["embedding"]),
        [question_embedding]
    ).flatten()

    max_indx = similarities.argsort()[::-1][0:top_results]
    new_df = df.loc[max_indx]

    # Build the RAG prompt
    context_json = new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")

    prompt = f"""I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{context_json}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
"""
    return prompt

# ---- Chat history ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Chat input ----
user_input = st.chat_input("Ask a question about the Sigma web dev course")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Build RAG prompt
                prompt = build_prompt(user_input, df)

                # Also write prompt to file (to match your original behavior)
                with open("prompt.txt", "w", encoding="utf-8") as f:
                    f.write(prompt)

                # Call local LLM via Ollama
                raw_response = inference(prompt)
                llm_response = raw_response.get("response", "")

                # Write response to file (to match your original behavior)
                with open("response.txt", "w", encoding="utf-8") as f:
                    f.write(llm_response)

                st.markdown(llm_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": llm_response}
                )

            except Exception as e:
                error_msg = f"⚠️ Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
