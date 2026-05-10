"""
Smart Notes AI — unified app combining:
  • 🃏 Flashcard generation (Claude API)
  • 🗣️  RAG Q&A on your notes (Ollama + ChromaDB + cross-encoder)

Run:
    pip install streamlit pdfplumber pypdf chromadb ollama \
                langchain-community langchain-text-splitters \
                sentence-transformers pymupdf
    streamlit run studycard_app.py
"""

import io
import json
import os
import re
import tempfile

import chromadb
import ollama
import pdfplumber
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="StudyCard AI",
    page_icon="🃏",
    layout="centered",
)

# ── Shared system prompts ─────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0d0d12; color: #e8e6f0; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem; font-weight: 800;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin-bottom: 0.25rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif; font-size: 1.05rem;
    color: #7c7a8a; font-weight: 300; margin-bottom: 2rem; letter-spacing: 0.01em;
}

/* Upload box */
.stFileUploader > div {
    border: 1.5px dashed #3a3850 !important; border-radius: 16px !important;
    background: #13121f !important; transition: border-color 0.2s;
}
.stFileUploader > div:hover { border-color: #7c6ef5 !important; }

/* Tab bar */
.stTabs [data-baseweb="tab-list"] {
    background: #13121f; border-radius: 14px; padding: 4px;
    border: 1px solid #2d2b45; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    border-radius: 10px !important; color: #6a6888 !important;
    padding: 0.55rem 1.4rem !important; transition: all 0.2s !important;
    border: none !important; background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c6ef5, #5a9cf8) !important;
    color: white !important; box-shadow: 0 2px 12px rgba(124,110,245,0.4) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.6rem !important; }

/* Flashcard */
.flashcard-outer {
    perspective: 1200px; width: 100%; max-width: 640px;
    margin: 0 auto 1.5rem auto; height: 260px; cursor: pointer;
}
.flashcard-inner {
    position: relative; width: 100%; height: 100%;
    transform-style: preserve-3d; transition: transform 0.55s cubic-bezier(.4,0,.2,1);
}
.flashcard-inner.flipped { transform: rotateY(180deg); }
.flashcard-face {
    position: absolute; width: 100%; height: 100%; backface-visibility: hidden;
    border-radius: 20px; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 2rem 2.4rem; box-sizing: border-box; text-align: center;
}
.flashcard-front {
    background: linear-gradient(145deg, #1c1a2e 0%, #16152a 100%);
    border: 1.5px solid #2d2b45;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04);
}
.flashcard-back {
    background: linear-gradient(145deg, #141e2e 0%, #101a2a 100%);
    border: 1.5px solid #1f3050;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04);
    transform: rotateY(180deg);
}
.card-label {
    font-family: 'Syne', sans-serif; font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 1rem; opacity: 0.5;
}
.front-label { color: #a78bfa; }
.back-label  { color: #60a5fa; }
.card-text   { font-family: 'DM Sans', sans-serif; font-size: 1.15rem; font-weight: 400; line-height: 1.6; color: #ddd9f3; }
.card-hint   { position: absolute; bottom: 1.1rem; font-size: 0.72rem; color: #4a4860; font-style: italic; }

/* Progress */
.progress-track { width:100%; height:3px; background:#1e1d2e; border-radius:99px; margin-bottom:0.5rem; overflow:hidden; }
.progress-fill  { height:100%; border-radius:99px; background:linear-gradient(90deg,#7c6ef5,#60a5fa); transition:width 0.4s ease; }
.card-counter   { text-align:center; font-size:0.8rem; color:#4a4860; margin-bottom:1.5rem; font-family:'Syne',sans-serif; letter-spacing:0.05em; }

/* Buttons */
.stButton > button {
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    border-radius: 12px !important; transition: all 0.2s !important; letter-spacing: 0.03em !important;
}
.nav-btn > button {
    background: #1c1a2e !important; color: #9d9bb5 !important;
    border: 1.5px solid #2d2b45 !important; padding: 0.6rem 1.6rem !important;
}
.nav-btn > button:hover { background: #28253e !important; color: #e8e6f0 !important; border-color: #5a56a0 !important; }
.gen-btn > button {
    background: linear-gradient(135deg, #7c6ef5, #5a9cf8) !important; color: white !important;
    padding: 0.75rem 2.5rem !important; font-size: 1rem !important; width: 100% !important;
    box-shadow: 0 4px 20px rgba(124,110,245,0.35) !important; border: none !important;
}
.gen-btn > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 28px rgba(124,110,245,0.5) !important; }
.flip-btn > button {
    background: transparent !important; color: #6a68a0 !important;
    border: 1px solid #2d2b45 !important; font-size: 0.8rem !important; padding: 0.4rem 1.2rem !important;
}
.flip-btn > button:hover { background: #1c1a2e !important; color: #a78bfa !important; }
.restart-btn > button {
    background: #1c1a2e !important; color: #a78bfa !important;
    border: 1.5px solid #3a3358 !important; font-size: 0.85rem !important;
}
.restart-btn > button:hover { background: #28253e !important; }
.ask-btn > button {
    background: linear-gradient(135deg, #1f4068, #163759) !important; color: #60a5fa !important;
    border: 1.5px solid #1f3050 !important; width: 100% !important;
    padding: 0.75rem !important; font-size: 1rem !important;
    box-shadow: 0 4px 20px rgba(32,96,160,0.25) !important;
}
.ask-btn > button:hover { background: linear-gradient(135deg, #265080, #1c4570) !important; box-shadow: 0 6px 28px rgba(32,96,160,0.4) !important; }
.process-btn > button {
    background: linear-gradient(135deg, #1a3a28, #122b1e) !important; color: #34d399 !important;
    border: 1.5px solid #1f4a32 !important; width: 100% !important;
    box-shadow: 0 4px 20px rgba(52,211,153,0.2) !important;
}
.process-btn > button:hover { box-shadow: 0 6px 28px rgba(52,211,153,0.35) !important; }

/* Select boxes */
.stSelectbox > div > div {
    background: #13121f !important; border: 1.5px solid #2d2b45 !important;
    border-radius: 10px !important; color: #e8e6f0 !important;
}

/* Text area */
.stTextArea textarea {
    background: #13121f !important; border: 1.5px solid #2d2b45 !important;
    border-radius: 12px !important; color: #e8e6f0 !important; font-family: 'DM Sans', sans-serif !important;
}
.stTextArea textarea:focus { border-color: #60a5fa !important; box-shadow: 0 0 0 2px rgba(96,165,250,0.15) !important; }

/* Badge */
.badge {
    display: inline-block; padding: 0.25rem 0.75rem; border-radius: 99px;
    font-size: 0.75rem; font-family: 'Syne', sans-serif; font-weight: 700;
    letter-spacing: 0.05em; margin-right: 0.4rem;
}
.badge-violet { background: rgba(124,110,245,0.18); color: #a78bfa; }
.badge-blue   { background: rgba(96,165,250,0.18);  color: #7db8fb; }
.badge-green  { background: rgba(52,211,153,0.18);  color: #5ddbb5; }

/* RAG answer box */
.rag-answer {
    background: #13121f; border: 1.5px solid #1f3050; border-radius: 16px;
    padding: 1.4rem 1.6rem; margin-top: 1rem; line-height: 1.7; font-size: 0.97rem;
}

/* Completion */
.celebrate {
    text-align: center; padding: 2.5rem 1rem;
    background: linear-gradient(145deg, #1c1a2e, #13121f);
    border: 1.5px solid #2d2b45; border-radius: 20px;
}

/* Divider / misc */
hr { border-color: #1e1d2e !important; margin: 1.8rem 0 !important; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── Shared helpers ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── PDF text extraction (pdfplumber — for flashcard generation) ───────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n\n".join(text_parts)


# ── PDF chunking (PyMuPDF via LangChain — for RAG indexing) ──────────────────

def process_document_for_rag(file_bytes: bytes, suffix: str = ".pdf") -> list[Document]:
    """Write bytes to a temp file, load with PyMuPDFLoader, chunk."""
    tmp = tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False)
    tmp.write(file_bytes)
    tmp.flush()
    tmp.close()
    loader = PyMuPDFLoader(tmp.name)
    docs = loader.load()
    os.unlink(tmp.name)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return splitter.split_documents(docs)


# ── ChromaDB ──────────────────────────────────────────────────────────────────

@st.cache_resource
def get_chroma_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    client = chromadb.PersistentClient(path="./studycard-chroma")
    return client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(splits: list[Document], file_name: str) -> None:
    collection = get_chroma_collection()
    documents, metadatas, ids = [], [], []
    for idx, split in enumerate(splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)


def query_collection(prompt: str, n_results: int = 10) -> dict:
    collection = get_chroma_collection()
    return collection.query(query_texts=[prompt], n_results=n_results)


# ── Cross-encoder re-ranking ──────────────────────────────────────────────────

@st.cache_resource
def load_cross_encoder() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def re_rank_cross_encoders(prompt: str, documents: list[str]) -> tuple[str, list[int]]:
    encoder = load_cross_encoder()
    ranks = encoder.rank(prompt, documents, top_k=3)
    relevant_text = ""
    relevant_ids: list[int] = []
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_ids.append(rank["corpus_id"])
    return relevant_text, relevant_ids


# ── Ollama LLM (RAG answers — streamed) ──────────────────────────────────────

def call_ollama_stream(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context: {context}, Question: {prompt}"},
        ],
    )
    for chunk in response:
        if not chunk["done"]:
            yield chunk["message"]["content"]
        else:
            break


# ── Ollama LLM (flashcard generation — fully local) ───────────────────────────

def _extract_json_array(text: str) -> list[dict]:
    """
    Try several increasingly aggressive strategies to pull a JSON array
    out of whatever the model returned.
    """
    # 1. Strip markdown fences (```json ... ``` or ``` ... ```)
    cleaned = re.sub(r"```[a-z]*", "", text).replace("```", "").strip()

    # 2. Direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 3. Find the first '[' … last ']' and parse that slice
    start = cleaned.find("[")
    end   = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(cleaned[start : end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # 4. Extract all {...} objects individually and wrap in a list
    objects = re.findall(r'\{[^{}]+\}', cleaned, re.DOTALL)
    if objects:
        cards = []
        for obj in objects:
            try:
                parsed = json.loads(obj)
                if "question" in parsed and "answer" in parsed:
                    cards.append({"question": parsed["question"], "answer": parsed["answer"]})
            except json.JSONDecodeError:
                continue
        if cards:
            return cards

    raise ValueError(f"Could not extract a JSON array from model output:\n{text[:300]}")


def generate_flashcards(text: str, num_cards: int, difficulty: str) -> list[dict]:
    diff_guide = {
        "Easy":   "focus on key definitions and basic facts.",
        "Medium": "include application and comparison questions.",
        "Hard":   "include analysis, synthesis, and edge-case questions.",
    }
    # Show the model exactly the format we want
    example = '[{"question": "What is X?", "answer": "X is ..."}, {"question": "...", "answer": "..."}]'
    system = (
        "You are a flashcard creator. Your only job is to output a JSON array of flashcard objects. "
        "Each object has exactly two keys: \"question\" and \"answer\". "
        "Output ONLY the raw JSON array. No explanation, no markdown, no code fences, no extra text. "
        f"Example format: {example}"
    )
    prompt = (
        f"Create exactly {num_cards} flashcards at {difficulty} difficulty from these study notes.\n"
        f"({diff_guide[difficulty]})\n\n"
        "Respond with ONLY a JSON array. Start your response with [ and end with ].\n\n"
        f"Study notes:\n{text[:5000]}"
    )

    for attempt in range(3):
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": 0.2},  # lower temp = more predictable formatting
        )
        raw = response["message"]["content"].strip()
        try:
            cards = _extract_json_array(raw)
            # Validate every card has the right keys
            valid = [c for c in cards if isinstance(c, dict) and "question" in c and "answer" in c]
            if valid:
                return valid[:num_cards]
        except (ValueError, json.JSONDecodeError):
            if attempt == 2:
                raise
            continue  # retry

    raise ValueError("Model failed to return valid flashcards after 3 attempts.")


# ══════════════════════════════════════════════════════════════════════════════
# ── Session state ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    # shared
    "pdf_bytes": None,          # raw bytes of the uploaded PDF
    "pdf_name": "",             # original filename
    "rag_indexed": False,       # whether the current PDF is indexed in Chroma
    # flashcards
    "cards": [],
    "current": 0,
    "flipped": False,
    "done": False,
    # rag
    "rag_history": [],          # list of {"q": ..., "a": ...}
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# ── Sidebar — shared PDF upload ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.5rem">🃏 Smart Notes AI</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub" style="font-size:0.88rem">Flashcards + AI Q&A</p>', unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader(
        "📄 Upload your PDF notes",
        type=["pdf"],
        help="Used by both tabs — upload once, study everywhere.",
    )

    # When a new file is uploaded, store bytes and reset state
    if uploaded is not None:
        new_bytes = uploaded.read()
        if new_bytes != st.session_state.pdf_bytes:
            st.session_state.pdf_bytes   = new_bytes
            st.session_state.pdf_name    = uploaded.name
            st.session_state.rag_indexed = False
            st.session_state.cards       = []
            st.session_state.current     = 0
            st.session_state.flipped     = False
            st.session_state.done        = False
            st.session_state.rag_history = []

    st.divider()

    # ── Flashcard sidebar settings ────────────────────────────────────────────
    st.markdown('<p style="font-size:0.78rem;color:#7c7a8a;font-family:Syne,sans-serif;font-weight:700;letter-spacing:0.1em;text-transform:uppercase">Flashcard settings</p>', unsafe_allow_html=True)
    num_cards  = st.selectbox("Number of cards", [5, 10, 15, 20], index=1)
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)

    if st.session_state.cards:
        total = len(st.session_state.cards)
        seen  = min(st.session_state.current + 1, total)
        st.divider()
        st.markdown(f'<p style="font-size:0.78rem;color:#4a4860">📚 {total} cards in deck</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:0.78rem;color:#4a4860">✅ {seen} reviewed</p>', unsafe_allow_html=True)

    st.divider()

    # ── RAG index button ──────────────────────────────────────────────────────
    st.markdown('<p style="font-size:0.78rem;color:#7c7a8a;font-family:Syne,sans-serif;font-weight:700;letter-spacing:0.1em;text-transform:uppercase">RAG settings</p>', unsafe_allow_html=True)
    st.caption("Index your PDF into the vector store so you can ask it questions.")

    st.markdown('<div class="process-btn">', unsafe_allow_html=True)
    process_clicked = st.button(
        "⚡ Index PDF for Q&A",
        disabled=(st.session_state.pdf_bytes is None),
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if process_clicked and st.session_state.pdf_bytes:
        with st.spinner("Chunking & embedding your notes…"):
            try:
                safe_name = st.session_state.pdf_name.translate(
                    str.maketrans({"-": "_", ".": "_", " ": "_"})
                )
                splits = process_document_for_rag(st.session_state.pdf_bytes)
                add_to_vector_collection(splits, safe_name)
                st.session_state.rag_indexed = True
                st.success(f"✅ Indexed {len(splits)} chunks!")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    if st.session_state.rag_indexed:
        st.markdown('<p style="font-size:0.75rem;color:#34d399;margin-top:0.3rem">● Vector index ready</p>', unsafe_allow_html=True)
    elif st.session_state.pdf_bytes:
        st.markdown('<p style="font-size:0.75rem;color:#f59e0b;margin-top:0.3rem">○ Not yet indexed — click above</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── Main area ─────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="hero-title">Smart Notes AI</div>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload your notes once — generate flashcards & ask AI questions.</p>', unsafe_allow_html=True)

tab_flash, tab_rag = st.tabs(["🃏  Flashcards", "🗣️  Ask Your Notes"])


# ══════════════════════════════════════════════════════════════════════════════
# ── TAB 1 — Flashcards ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

with tab_flash:

    if st.session_state.pdf_bytes is None:
        st.info("👆  Upload a PDF in the sidebar to get started.")

    elif not st.session_state.cards:
        # ── Generate prompt ───────────────────────────────────────────────────
        st.markdown(
            f'<span class="badge badge-violet">PDF ready</span>'
            f'<span class="badge badge-blue">{num_cards} cards</span>'
            f'<span class="badge badge-green">{difficulty}</span>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown('<div class="gen-btn">', unsafe_allow_html=True)
        if st.button("✨  Generate Flashcards"):
            with st.spinner("Reading your notes and crafting cards…"):
                try:
                    text = extract_text_from_pdf(st.session_state.pdf_bytes)
                    if not text.strip():
                        st.error("Could not extract text. The PDF may be scanned or image-based.")
                    else:
                        cards = generate_flashcards(text, num_cards, difficulty)
                        st.session_state.cards   = cards
                        st.session_state.current = 0
                        st.session_state.flipped = False
                        st.session_state.done    = False
                        st.rerun()
                except (json.JSONDecodeError, ValueError) as e:
                    st.error(f"Could not parse flashcards after 3 attempts. Try a smaller number of cards or a simpler PDF. Detail: {e}")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # ── Deck view ─────────────────────────────────────────────────────────
        cards   = st.session_state.cards
        idx     = st.session_state.current
        total   = len(cards)

        if st.session_state.done:
            # Completion screen
            st.markdown(f"""
            <div class="celebrate">
                <div style="font-size:3rem;margin-bottom:0.5rem">🎉</div>
                <h2>Deck Complete!</h2>
                <p style="color:#7c7a8a;font-size:0.95rem">You reviewed all {total} flashcards.</p>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="restart-btn">', unsafe_allow_html=True)
                if st.button("🔄  Restart deck", use_container_width=True):
                    st.session_state.current = 0
                    st.session_state.flipped = False
                    st.session_state.done    = False
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with col_b:
                st.markdown('<div class="restart-btn">', unsafe_allow_html=True)
                if st.button("🆕  New deck", use_container_width=True):
                    st.session_state.cards   = []
                    st.session_state.current = 0
                    st.session_state.flipped = False
                    st.session_state.done    = False
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            card    = cards[idx]
            flipped = st.session_state.flipped

            # Progress bar
            progress_pct = int((idx / total) * 100)
            st.markdown(f'<div class="progress-track"><div class="progress-fill" style="width:{progress_pct}%"></div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card-counter">Card {idx + 1} of {total}</div>', unsafe_allow_html=True)

            # Card HTML
            flipped_class = "flipped" if flipped else ""
            q_esc = card["question"].replace('"', '&quot;').replace("'", "&#39;")
            a_esc = card["answer"].replace('"', '&quot;').replace("'", "&#39;")
            st.markdown(f"""
            <div class="flashcard-outer">
              <div class="flashcard-inner {flipped_class}">
                <div class="flashcard-face flashcard-front">
                  <span class="card-label front-label">Question</span>
                  <div class="card-text">{q_esc}</div>
                  <span class="card-hint">click Flip to reveal answer</span>
                </div>
                <div class="flashcard-face flashcard-back">
                  <span class="card-label back-label">Answer</span>
                  <div class="card-text">{a_esc}</div>
                  <span class="card-hint">click Flip to see question</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Flip + navigation
            col_flip, _, _ = st.columns([1, 1, 1])
            with col_flip:
                st.markdown('<div class="flip-btn">', unsafe_allow_html=True)
                if st.button("🔄  Flip card", use_container_width=True):
                    st.session_state.flipped = not st.session_state.flipped
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            st.write("")
            col_prev, col_sp, col_next = st.columns([1, 2, 1])
            with col_prev:
                st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
                if st.button("← Prev", disabled=(idx == 0), use_container_width=True):
                    st.session_state.current -= 1
                    st.session_state.flipped  = False
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with col_next:
                st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
                label = "Finish ✓" if idx == total - 1 else "Next →"
                if st.button(label, use_container_width=True):
                    if idx == total - 1:
                        st.session_state.done    = True
                        st.session_state.flipped = False
                    else:
                        st.session_state.current += 1
                        st.session_state.flipped  = False
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="restart-btn">', unsafe_allow_html=True)
            if st.button("📄  Regenerate deck with new settings"):
                st.session_state.cards   = []
                st.session_state.current = 0
                st.session_state.flipped = False
                st.session_state.done    = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── TAB 2 — Ask Your Notes (RAG) ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

with tab_rag:

    if st.session_state.pdf_bytes is None:
        st.info("👆  Upload a PDF in the sidebar, then index it to start asking questions.")

    elif not st.session_state.rag_indexed:
        st.warning('⚡  Your PDF is uploaded but not yet indexed. Click **"Index PDF for Q&A"** in the sidebar first.')

    else:
        st.markdown('<p style="color:#7c7a8a;font-size:0.9rem;margin-bottom:1rem">Ask anything about your notes — answers are grounded entirely in your PDF.</p>', unsafe_allow_html=True)

        # ── Chat history ──────────────────────────────────────────────────────
        for entry in st.session_state.rag_history:
            with st.chat_message("user"):
                st.write(entry["q"])
            with st.chat_message("assistant"):
                st.markdown(f'<div class="rag-answer">{entry["a"]}</div>', unsafe_allow_html=True)
                if entry.get("ids"):
                    with st.expander("📎 Most relevant chunk indices"):
                        st.write(entry["ids"])

        # ── Question input ────────────────────────────────────────────────────
        prompt = st.chat_input("Ask your notes a question…")

        if prompt:
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching your notes…"):
                    try:
                        results        = query_collection(prompt)
                        context_docs   = results.get("documents", [[]])[0]

                        if not context_docs:
                            st.warning("No relevant passages found in your notes for that question.")
                        else:
                            relevant_text, relevant_ids = re_rank_cross_encoders(prompt, context_docs)

                            answer_placeholder = st.empty()
                            full_answer = ""
                            for chunk in call_ollama_stream(context=relevant_text, prompt=prompt):
                                full_answer += chunk
                                answer_placeholder.markdown(
                                    f'<div class="rag-answer">{full_answer}▌</div>',
                                    unsafe_allow_html=True,
                                )
                            answer_placeholder.markdown(
                                f'<div class="rag-answer">{full_answer}</div>',
                                unsafe_allow_html=True,
                            )

                            # Save to history
                            st.session_state.rag_history.append({
                                "q":   prompt,
                                "a":   full_answer,
                                "ids": relevant_ids,
                            })

                            with st.expander("📎 Most relevant chunk indices"):
                                st.write(relevant_ids)

                    except Exception as e:
                        st.error(f"Something went wrong: {e}")

        # ── Clear history ─────────────────────────────────────────────────────
        if st.session_state.rag_history:
            st.write("")
            st.markdown('<div class="restart-btn">', unsafe_allow_html=True)
            if st.button("🗑️  Clear chat history"):
                st.session_state.rag_history = []
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)