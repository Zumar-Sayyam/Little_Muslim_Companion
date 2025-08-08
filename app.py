import streamlit as st
import random, json, os, re
from gtts import gTTS
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# ---------- CONFIG ----------
openai.api_key = os.getenv("MY_API_KEY")  # Streamlit Secret
openai.api_base = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama3-8b-8192"
PROGRESS_FILE = "progress.json"

st.set_page_config(page_title="Little Muslim Companion", page_icon="🌙")

st.title("🌙 Little Muslim Companion")
st.write("_As-salamu Alaikum! Let's learn and have fun together!_")

# ---------- LOAD PDFs INTO VECTOR DB ----------
def load_pdfs(folder="assets"):
    texts = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, file))
            for page in reader.pages:
                texts.append(page.extract_text())
    return "\n".join(texts)

@st.cache_resource
def build_rag():
    pdf_text = load_pdfs("assets")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(pdf_text)
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key, openai_api_base=openai.api_base)
    db = FAISS.from_texts(chunks, embedding=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.3, openai_api_key=openai.api_key, openai_api_base=openai.api_base)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

qa_chain = build_rag()

# ---------- STORY TAB ----------
st.header("📖 Storytime")
topic = st.text_input("Enter topic for story (e.g., honesty, Prophet Yusuf)")
if st.button("Generate Story"):
    query = f"Tell an authentic Islamic story for children about {topic}. 200-400 words. Use only the PDF content."
    story = qa_chain.run(query)
    st.write(story)

    tts = gTTS(story, lang='en')
    audio_path = "story.mp3"
    tts.save(audio_path)
    st.audio(audio_path)
