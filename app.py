"""RAG Chatbot
=============
Chat with your own documents. Sign in with an OpenAI API key, attach PDF, Word,
Excel (XLS/XLSX) or CSV files with the + button, and ask questions. Answers come
only from your files and cite the file and page / sheet / rows they used.

Run:  streamlit run app.py
"""
import hashlib
import html
import io
import re

import numpy as np
import pandas as pd
import streamlit as st
from docx import Document
from openai import AuthenticationError, OpenAI
from pypdf import PdfReader

EMBED_MODEL = "text-embedding-3-small"
FILE_TYPES = ["pdf", "xls", "xlsx", "docx", "csv"]
ICONS = {"pdf": "📕", "docx": "📘", "xls": "📗", "xlsx": "📗", "csv": "🧾"}
MODES = ["Normal", "DeepThink"]
MODE_LABELS = {"Normal": "🍃 Normal", "DeepThink": "⚡ DeepThink"}
TOP_K = {"Normal": 4, "DeepThink": 8}
MAX_CHUNKS_PER_FILE = 1500  # keeps embedding cost sensible for huge files
NORMAL_PREFS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini", "gpt-4o"]
DEEP_PREFS = ["gpt-4.1", "gpt-4o", "gpt-5", "gpt-4-turbo"]

SYSTEM = """You are a precise assistant that answers using ONLY the context below, which comes from the user's uploaded documents (PDF, Word, Excel and CSV files).
Rules:
1. Answer only from the context.
2. If the answer is not in the context, say exactly: "I couldn't find that in the documents."
3. Cite the sources you used after the facts, as (file name, location), e.g. (report.pdf, p. 3) or (sales.xlsx, sheet Q1, rows 2-21).
4. For spreadsheet or CSV data, read the column names carefully. Do any calculations step by step and show the numbers you used.
5. Keep the answer clear and well organised.

Context:
{context}"""
DEEP_EXTRA = "\n\nThis is a DeepThink request: work through the question carefully, compare all the relevant passages, resolve conflicts between sources, check any arithmetic, and give a thorough but well-organised answer."


# ======================================================================
# Reading files  ->  list of (location, text)
# ======================================================================
def parse_pdf(data):
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("this PDF is password-protected")
    pages = [(f"p. {i}", (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages, 1)]
    return [x for x in pages if x[1]]


def parse_docx(data):
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return [("document", "\n".join(parts))] if parts else []


def df_blocks(df, label, rows_per=20):
    df = df.dropna(how="all").dropna(axis=1, how="all").fillna("")
    cols = [str(c) for c in df.columns]
    out = []
    for start in range(0, len(df), rows_per):
        block = df.iloc[start:start + rows_per]
        lines = ["; ".join(f"{c}: {v}" for c, v in zip(cols, row) if str(v).strip()) for row in block.itertuples(index=False)]
        text = "Columns: " + ", ".join(cols) + "\n" + "\n".join(lines)
        out.append((f"{label}, rows {start + 2}-{start + len(block) + 1}", text))
    return out


def parse_excel(data, name):
    engine = "xlrd" if name.lower().endswith(".xls") else "openpyxl"
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, engine=engine, dtype=str)
    out = []
    for sheet, df in sheets.items():
        out += df_blocks(df, f"sheet {sheet}")
    return out


def parse_csv(data):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), dtype=str, encoding=enc, sep=None, engine="python")
            return df_blocks(df, "csv")
        except UnicodeDecodeError:
            continue
    raise ValueError("could not read this CSV file")


def extract(name, data):
    ext = name.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return parse_pdf(data)
    if ext == "docx":
        return parse_docx(data)
    if ext in ("xls", "xlsx"):
        return parse_excel(data, name)
    if ext == "csv":
        return parse_csv(data)
    raise ValueError("unsupported file type")


def split_text(text, size=1000, overlap=150):
    text = re.sub(r"[ \t]+", " ", text).strip()
    out, i = [], 0
    while i < len(text):
        j = min(i + size, len(text))
        if j < len(text):
            k = max(text.rfind("\n", i + size // 2, j), text.rfind(". ", i + size // 2, j))
            if k > 0:
                j = k + 1
        out.append(text[i:j].strip())
        if j >= len(text):
            break
        i = max(j - overlap, i + 1)
    return [c for c in out if c]


def to_chunks(name, data):
    chunks, part = [], 0
    for loc, text in extract(name, data):
        for piece in split_text(text, size=1800 if loc != "document" and not loc.startswith("p.") else 1000):
            if loc == "document":
                part += 1
                shown = f"part {part}"
            else:
                shown = loc
            chunks.append({"source": name, "loc": shown, "text": piece})
    if not chunks:
        raise ValueError("no readable text found (is it a scanned file?)")
    return chunks


# ======================================================================
# Embeddings, search, answer
# ======================================================================
def make_client(key):
    return OpenAI(api_key=key)


def embed(client, texts):
    vecs = []
    for i in range(0, len(texts), 96):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + 96])
        vecs.extend(d.embedding for d in resp.data)
    arr = np.asarray(vecs, dtype=np.float32)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)


def new_kb():
    return {"chunks": [], "emb": None, "files": {}, "hashes": set()}


def ingest(client, uploads, status):
    kb = st.session_state.kb
    added, problems = [], []
    for f in uploads:
        data = f.getvalue()
        digest = hashlib.sha1(data).hexdigest()
        if digest in kb["hashes"]:
            problems.append(f"{f.name}: already added")
            continue
        try:
            status.update(label=f"Reading {f.name}…")
            chunks = to_chunks(f.name, data)
            capped = len(chunks) > MAX_CHUNKS_PER_FILE
            chunks = chunks[:MAX_CHUNKS_PER_FILE]
            status.update(label=f"Embedding {f.name} ({len(chunks)} parts)…")
            vec = embed(client, [c["text"] for c in chunks])
        except Exception as exc:
            problems.append(f"{f.name}: {exc}")
            continue
        kb["chunks"] += chunks
        kb["emb"] = vec if kb["emb"] is None else np.vstack([kb["emb"], vec])
        kb["hashes"].add(digest)
        kb["files"][f.name] = {"chunks": len(chunks), "capped": capped}
        added.append(f.name + (" (first part only)" if capped else ""))
    return added, problems


def retrieve(client, query, k):
    kb = st.session_state.kb
    q = embed(client, [query])[0]
    scores = kb["emb"] @ q
    return [kb["chunks"][i] for i in np.argsort(-scores)[:k]]


def stream_reply(client, model, messages):
    try:
        stream = client.chat.completions.create(model=model, messages=messages, temperature=0, stream=True)
    except Exception as exc:
        if "temperature" not in str(exc).lower():
            raise
        stream = client.chat.completions.create(model=model, messages=messages, stream=True)
    for part in stream:
        if part.choices and part.choices[0].delta.content:
            yield part.choices[0].delta.content


def pick_model(models, prefs):
    for p in prefs:
        if p in models:
            return p
    return models[0] if models else "gpt-4o-mini"


def chat_models(ids):
    bad = re.compile(r"(embed|audio|tts|whisper|transcribe|realtime|image|dall|moderation|instruct|search|davinci|babbage|codex|computer)")
    keep = [i for i in ids if re.match(r"^(gpt-|chatgpt|o\d)", i) and not bad.search(i) and not re.search(r"\d{4}-\d{2}-\d{2}$", i)]
    return sorted(keep) or sorted(i for i in ids if i.startswith("gpt-"))


# ======================================================================
# Look and feel (black stage, iridescent glass orbs, blue robot)
# ======================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
:root{color-scheme:dark;--ink:#f3f3f8;--mute:rgba(243,243,248,.68);--line:rgba(255,255,255,.18);--glass:rgba(255,255,255,.06);--grad:linear-gradient(135deg,#3f6bff,#8a3dff)}
html,body,.stApp,[data-testid="stApp"],[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#000!important}
body,.stApp{font-family:'Plus Jakarta Sans',system-ui,sans-serif;font-size:15.5px}
body *{color:var(--ink)!important}
a{color:#9db4ff!important}
::placeholder{color:rgba(243,243,248,.5)!important;opacity:1!important}
.stApp::before,.stApp::after{content:"";position:fixed;border-radius:50%;pointer-events:none;z-index:0;
background:radial-gradient(circle at 32% 28%,rgba(255,255,255,.38),transparent 34%),conic-gradient(from 210deg,#2a35ff,#8a2dff,#ff3fa4,#ff9a3d,#6a3cff,#2a35ff);filter:blur(14px) saturate(1.25);opacity:.85}
.stApp::before{width:460px;height:460px;right:-170px;top:-190px}
.stApp::after{width:520px;height:520px;left:-210px;bottom:-250px}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stAppDeployButton"]{display:none}
[data-testid="stMainBlockContainer"]{max-width:940px;padding:1.6rem 1.2rem 9rem;position:relative;z-index:2}
[data-testid="stBottom"],[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{background:transparent!important}
[data-testid="stBottomBlockContainer"]{max-width:940px;position:relative;z-index:2;padding-bottom:1rem}
[data-testid="stChatInput"]{background:rgba(255,255,255,.08)!important;border:1px solid rgba(255,255,255,.26)!important;border-radius:22px!important;-webkit-backdrop-filter:blur(18px);backdrop-filter:blur(18px)}
[data-testid="stChatInput"]>div,[data-testid="stChatInput"] textarea{background:transparent!important}
[data-testid="stChatInputFileUploadButton"],[data-testid="stChatInputSubmitButton"]{background:var(--grad)!important;border:0!important;border-radius:50%!important;box-shadow:0 0 18px rgba(100,110,255,.6)}
[data-testid="stChatInputFileUploadButton"]{width:40px!important;height:40px!important}
[data-testid="stChatMessage"]{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);border-radius:18px;padding:1rem 1.1rem;-webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px)}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){background:rgba(74,123,255,.15);border-color:rgba(120,150,255,.32)}
button[data-testid^="stBaseButton"]{background:rgba(255,255,255,.08)!important;border:1px solid var(--line)!important;border-radius:999px!important;font-weight:600}
button[data-testid^="stBaseButton"]:hover{border-color:#8aa2ff!important;background:rgba(255,255,255,.14)!important}
button[data-testid="stBaseButton-primary"],button[data-testid$="Active"]{background:var(--grad)!important;border-color:transparent!important}
[data-testid="stFileUploaderDropzone"]{background:rgba(255,255,255,.05)!important;border:1.5px dashed rgba(140,160,255,.6)!important;border-radius:18px!important;padding:22px!important}
[data-testid="stFileUploaderDropzone"] button{background:var(--grad)!important;border:0!important;padding:.55rem 1.5rem!important;font-weight:700!important;box-shadow:0 0 16px rgba(100,110,255,.5)}
[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="textarea"],[data-baseweb="select"]>div{background:rgba(255,255,255,.08)!important;border-color:var(--line)!important;border-radius:12px!important}
[data-baseweb="input"] input,[data-baseweb="base-input"] input{background:transparent!important}
[data-baseweb="popover"] div,[data-baseweb="menu"],ul[role="listbox"],[data-testid="stPopoverBody"]{background-color:#10101c!important}
[data-baseweb="popover"]>div,[data-testid="stPopoverBody"]{border:1px solid var(--line)!important;border-radius:14px!important}
li[role="option"]:hover,li[role="option"][aria-selected="true"]{background-color:rgba(255,255,255,.12)!important}
[data-testid="stAlert"]{background:rgba(255,255,255,.07)!important;border:1px solid var(--line)!important;border-radius:14px!important}
[data-testid="stAlert"]>div{background:transparent!important}
[data-testid="stExpander"] details,[data-testid="stStatus"] details{background:rgba(255,255,255,.04)!important;border:1px solid var(--line)!important;border-radius:14px!important}
[data-testid="stText"],pre,code{background:rgba(255,255,255,.06)!important;border-radius:10px}
div[role="dialog"]{background:rgba(12,12,24,.97)!important;border:1px solid rgba(255,255,255,.22);border-radius:24px}
.brand{display:flex;align-items:center}
.brand .bt,.brand .bf{border:1px solid rgba(130,110,255,.5);background:#121212;border-radius:999px;box-shadow:0 0 24px rgba(90,70,255,.18)}
.brand .bt{padding:16px 46px;font:700 26px 'Syne',sans-serif;letter-spacing:.01em}
.brand .neck{width:22px;height:16px;margin:0 -4px;background:#121212;border-top:1px solid rgba(130,110,255,.5);border-bottom:1px solid rgba(130,110,255,.5)}
.brand .bf{width:86px;height:62px;display:grid;place-items:center}
.brand svg{filter:drop-shadow(0 0 8px rgba(80,140,255,.7))}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 8px}
.chip{border:1px solid var(--line);background:var(--glass);border-radius:999px;padding:5px 13px;font-size:13px}
.chip i{font-style:normal;color:var(--mute)!important;margin-left:6px}
.chip.ok::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:#34d399;margin-right:8px;box-shadow:0 0 8px #34d399}
.hero{text-align:center;margin:7vh 0 3vh}
.hero h1{font:700 clamp(28px,4.2vw,44px)/1.12 'Syne',sans-serif;margin:0 0 10px;background:linear-gradient(90deg,#fff,#b9c8ff 50%,#ffd2b0);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;color:transparent!important}
.hero p,.uphead p,.step span{color:var(--mute)!important;margin:0}
.uphead{margin:26px 0 10px}
.uphead h3{font:700 24px 'Syne',sans-serif;margin:0 0 6px}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:20px 0}
.step{border:1px solid var(--line);background:var(--glass);border-radius:16px;padding:16px 18px}
.step b{display:block;font:700 15px 'Syne',sans-serif;margin-bottom:4px}
.tip{text-align:center;font-size:12.5px;color:var(--mute)!important;margin-top:6px}
.mascot{position:fixed;right:3vw;bottom:-6px;width:min(22vw,310px);z-index:1;pointer-events:none;filter:drop-shadow(0 0 40px rgba(70,90,255,.45));opacity:.95}
@media (max-width:1100px){.mascot{display:none}.brand .bt{padding:12px 26px;font-size:20px}.brand .bf{width:64px;height:48px}.steps{grid-template-columns:1fr}}
"""

MASCOT = """<svg class="mascot" viewBox="0 0 320 520" xmlns="http://www.w3.org/2000/svg">
<defs>
<radialGradient id="rb" cx="35%" cy="25%" r="85%"><stop offset="0" stop-color="#a9c4ff"/><stop offset=".45" stop-color="#4a74f5"/><stop offset="1" stop-color="#1c2fb5"/></radialGradient>
<radialGradient id="rv" cx="50%" cy="40%" r="70%"><stop offset="0" stop-color="#17173a"/><stop offset="1" stop-color="#050510"/></radialGradient>
<filter id="rg" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="3.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>
<rect x="112" y="400" width="38" height="130" rx="19" fill="url(#rb)"/><rect x="170" y="400" width="38" height="130" rx="19" fill="url(#rb)"/>
<rect x="38" y="292" width="46" height="140" rx="23" fill="url(#rb)" transform="rotate(10 61 300)"/><rect x="236" y="292" width="46" height="140" rx="23" fill="url(#rb)" transform="rotate(-10 259 300)"/>
<rect x="138" y="252" width="44" height="30" rx="10" fill="#1b2bb0"/>
<path d="M96 292q64-26 128 0l14 118q-78 30-156 0z" fill="url(#rb)"/>
<path d="M104 392q56 18 112 0" stroke="#8a6bff" stroke-width="5" fill="none" opacity=".8"/>
<ellipse cx="160" cy="334" rx="30" ry="19" fill="#fff" filter="url(#rg)"/>
<ellipse cx="46" cy="150" rx="26" ry="38" fill="#3a63ee"/><ellipse cx="274" cy="150" rx="26" ry="38" fill="#3a63ee"/>
<circle cx="46" cy="150" r="14" fill="#1b2bb0"/><circle cx="274" cy="150" r="14" fill="#1b2bb0"/>
<circle cx="160" cy="150" r="112" fill="url(#rb)"/>
<ellipse cx="160" cy="152" rx="90" ry="72" fill="url(#rv)" stroke="#9a7bff" stroke-width="4" filter="url(#rg)"/>
<g stroke="#ffe3cc" stroke-width="8" stroke-linecap="round" fill="none" filter="url(#rg)"><path d="M116 140q15-26 30 0"/><path d="M174 140q15-26 30 0"/><path d="M138 170q22 24 44 0"/></g>
</svg>"""

BRAND = """<div class="brand"><div class="bt">RAG Chatbot</div><div class="neck"></div><div class="bf"><svg width="38" height="38" viewBox="0 0 48 48"><defs><linearGradient id="ff" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#8cb8ff"/><stop offset="1" stop-color="#2e4bd6"/></linearGradient></defs><path d="M5 14a4 4 0 0 1 4-4h10l4 5h16a4 4 0 0 1 4 4v18a4 4 0 0 1-4 4H9a4 4 0 0 1-4-4z" fill="url(#ff)"/><path d="M5 22h38v15a4 4 0 0 1-4 4H9a4 4 0 0 1-4-4z" fill="#1f3bd0" opacity=".85"/></svg></div></div>"""


def inject_theme():
    st.markdown(f"<style>{CSS}</style>{MASCOT}", unsafe_allow_html=True)


# ======================================================================
# Sign-in popup
# ======================================================================
@st.dialog("Sign in with your API key")
def login_dialog():
    st.caption("Enter your OpenAI API key. It is verified first and kept only for this browser session.")
    key = st.text_input("OpenAI API key", type="password", placeholder="sk-...", label_visibility="collapsed")
    if st.button("Verify and continue", type="primary"):
        key = key.strip()
        if not key.startswith("sk-"):
            st.error("API keys start with “sk-”.")
            return
        try:
            with st.spinner("Checking your key…"):
                ids = [m.id for m in make_client(key).models.list()]
        except AuthenticationError:
            st.error("That API key was rejected. Check it and try again.")
            return
        except Exception as exc:
            st.error(f"Could not verify the key: {exc}")
            return
        models = chat_models(ids)
        st.session_state.update(
            api_key=key, models=models,
            normal_model=pick_model(models, NORMAL_PREFS), deep_model=pick_model(models, DEEP_PREFS),
        )
        st.rerun()


# ======================================================================
# User guide
# ======================================================================
GUIDE = """
### Quick start
1. **Sign in.** Paste your OpenAI API key in the popup. It is checked first and kept only for this session.
2. **Add documents.** Use the **Add your documents** box (drag and drop, or click **Browse files**, then **Index**), or press the glowing **＋** inside the chat box. You can add PDF, Word (DOCX), Excel (XLS, XLSX) and CSV files, as many as you like.
3. **Ask questions.** Type in the chat box and press Enter. Answers stream in and cite the file and the page, sheet or rows they came from.

### What you can ask
- **PDF and Word:** “Summarise the key points.” · “What does the contract say about termination?”
- **Excel and CSV:** “What is the total sales for the North region?” · “Which row has the highest value?”
- **Across files:** “Compare the Q1 sheet with the annual report.”

### Normal or DeepThink
- **🍃 Normal** is quick and uses the 4 most relevant passages. Good for simple lookups.
- **⚡ DeepThink** uses a stronger model and 8 passages, compares sources and checks arithmetic. Use it for analysis, comparisons and numbers.

### Reading an answer
Open **🔍 Sources** under any answer to see the exact passages it used. If the answer is not in your files the assistant says so instead of guessing.

### The ⚙️ menu
Change the Normal and DeepThink models, clear the chat, remove all documents, or sign out.

### Tips and limits
- Scanned PDFs or images of text have no readable text. Use a text-based PDF.
- Password-protected files cannot be read.
- Each file is limited to its first 1,500 parts to keep cost down. For very large spreadsheets, add only the sheets you need.
- You pay OpenAI for indexing and for answers. Re-adding the same file is skipped automatically.
- Ask specific questions and name the file or sheet when you have many documents.

### If something goes wrong
- **“API key was rejected”:** check the key, or create a new one in your OpenAI account.
- **“I couldn't find that in the documents”:** rephrase, or check that the right file is added (shown as chips under the title).
- **Error about quota or billing:** add credit to your OpenAI account.
"""


@st.dialog("User guide", width="large")
def guide_dialog():
    st.markdown(GUIDE)
    if st.button("Got it", type="primary"):
        st.rerun()


STEPS = (
    '<div class="steps">'
    '<div class="step"><b>1 · Sign in</b><span>Verify your OpenAI API key.</span></div>'
    '<div class="step"><b>2 · Add files</b><span>PDF, Word, Excel (XLS, XLSX) and CSV.</span></div>'
    '<div class="step"><b>3 · Ask</b><span>Get answers with page and sheet citations.</span></div>'
    '</div>'
)
SUGGESTIONS = ["Summarise my documents", "What are the key figures and dates?", "List the main topics"]


# ======================================================================
# Main app
# ======================================================================
def render_sources(sources):
    with st.expander("🔍 Sources"):
        for s in sources:
            st.markdown(f"**{s['source']} · {s['loc']}**")
            st.text(s["text"][:500])


def settings_popover():
    with st.popover("⚙️"):
        models = st.session_state.models
        st.selectbox("Normal model", list(dict.fromkeys([st.session_state.normal_model] + models)), key="normal_model")
        st.selectbox("DeepThink model", list(dict.fromkeys([st.session_state.deep_model] + models)), key="deep_model")
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()
        if st.button("Remove all documents"):
            st.session_state.kb = new_kb()
            st.rerun()
        if st.button("Sign out"):
            st.session_state.clear()
            st.rerun()


def add_note(added, problems):
    note = ("✅ Added: " + ", ".join(added) + ". " if added else "") + ("⚠️ " + "; ".join(problems) if problems else "")
    st.session_state.messages.append({"role": "assistant", "content": note.strip() or "No files were added."})


def uploader_block(client):
    def body():
        n = st.session_state.up_n
        files = st.file_uploader("Upload documents", type=FILE_TYPES, accept_multiple_files=True,
                                 key=f"up_{n}", label_visibility="collapsed")
        if files and st.button(f"Index {len(files)} file(s)", type="primary", key=f"idx_{n}"):
            with st.status("Reading your files…", expanded=True) as status:
                added, problems = ingest(client, files, status)
                status.update(label=f"Added {len(added)} file(s)" if added else "No files added",
                              state="complete" if added else "error")
            add_note(added, problems)
            st.session_state.up_n += 1
            st.rerun()

    if st.session_state.kb["files"]:
        with st.expander("＋ Add more documents"):
            body()
    else:
        st.markdown('<div class="uphead"><h3>Add your documents</h3><p>PDF · Word (DOCX) · Excel (XLS, XLSX) · CSV. Drag and drop, or click Browse files.</p></div>', unsafe_allow_html=True)
        body()


def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="📁", layout="wide", initial_sidebar_state="collapsed")
    inject_theme()

    if not st.session_state.get("api_key"):
        st.markdown(BRAND, unsafe_allow_html=True)
        st.markdown('<div class="hero"><h1>Chat with your documents</h1><p>Sign in with your OpenAI API key to begin.</p></div>', unsafe_allow_html=True)
        st.markdown(STEPS, unsafe_allow_html=True)
        st.button("🔑 Enter API key", type="primary")
        login_dialog()
        st.stop()

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("kb", new_kb())
    st.session_state.setdefault("up_n", 0)
    kb = st.session_state.kb
    client = make_client(st.session_state.api_key)
    pending = st.session_state.pop("pending_q", None)

    if not st.session_state.get("guide_seen"):
        st.session_state.guide_seen = True
        guide_dialog()

    left, help_col, gear = st.columns([8, 1, 1], vertical_alignment="center")
    left.markdown(BRAND, unsafe_allow_html=True)
    if help_col.button("❓", help="User guide"):
        guide_dialog()
    with gear:
        settings_popover()

    chips = f'<span class="chip ok">Signed in<i>{html.escape(st.session_state.normal_model)}</i></span>'
    chips += "".join(
        f'<span class="chip">{ICONS.get(n.rsplit(".", 1)[-1].lower(), "📄")} {html.escape(n)}<i>{v["chunks"]} parts</i></span>'
        for n, v in kb["files"].items()
    )
    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

    uploader_block(client)

    if not st.session_state.messages and not pending:
        if kb["files"]:
            st.markdown('<div class="hero"><h1>Ask anything about your files</h1><p>Try a suggestion or type your own question below.</p></div>', unsafe_allow_html=True)
            for col, q in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
                if col.button(q, key=f"sg_{q}"):
                    st.session_state.pending_q = q
                    st.rerun()
        else:
            st.markdown(STEPS, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    with st.bottom:
        mode = st.pills("Mode", MODES, default="Normal", required=True, key="mode",
                        format_func=MODE_LABELS.get, label_visibility="collapsed")
        prompt = st.chat_input("Ask anything…", accept_file="multiple", file_type=FILE_TYPES)
        st.markdown('<div class="tip">Press the glowing ＋ to attach PDF, Word, Excel or CSV files · Enter to send</div>', unsafe_allow_html=True)
    if not prompt and not pending:
        return

    text = (pending or (prompt.text if prompt else "") or "").strip()
    files = list(prompt.files or []) if prompt else []

    if files:
        with st.chat_message("assistant"):
            with st.status("Reading your files…", expanded=False) as status:
                added, problems = ingest(client, files, status)
                status.update(label=f"Added {len(added)} file(s)" if added else "No files added", state="complete" if added else "error")
        add_note(added, problems)
        if not text:
            st.rerun()

    if not text:
        return

    st.session_state.messages.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)

    with st.chat_message("assistant"):
        if not st.session_state.kb["chunks"]:
            reply = "I don't have any documents yet. Add a PDF, Word, Excel or CSV file first, using the upload box or the **＋** in the chat box."
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            return
        try:
            history = [m for m in st.session_state.messages[:-1] if m["role"] in ("user", "assistant")]
            previous = [m["content"] for m in history if m["role"] == "user"]
            query = f"{previous[-1]} {text}" if len(text.split()) < 6 and previous else text
            with st.spinner("Searching your documents…"):
                hits = retrieve(client, query, TOP_K[mode])
            context = "\n\n".join(f"[{h['source']} · {h['loc']}]\n{h['text']}" for h in hits)
            system = SYSTEM.replace("{context}", context) + (DEEP_EXTRA if mode == "DeepThink" else "")
            messages = [{"role": "system", "content": system}]
            messages += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
            messages.append({"role": "user", "content": text})
            model = st.session_state.deep_model if mode == "DeepThink" else st.session_state.normal_model
            answer = st.write_stream(stream_reply(client, model, messages))
            sources = [{"source": h["source"], "loc": h["loc"], "text": h["text"]} for h in hits]
            render_sources(sources)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            st.session_state.messages.pop()


if __name__ == "__main__":
    main()
