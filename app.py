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


def to_chunks(name, data, chunk_size=1000, chunk_overlap=150):
    chunks, part = [], 0
    for loc, text in extract(name, data):
        for piece in split_text(text, size=chunk_size, overlap=chunk_overlap):
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
            chunks = to_chunks(
                f.name, data,
                chunk_size=st.session_state.get("chunk_size", 1000),
                chunk_overlap=st.session_state.get("chunk_overlap", 150),
            )
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{color-scheme:dark;--ink:#edf4ff;--muted:#9caec8;--line:rgba(151,180,220,.2);--surface:#111d32;--soft:#0b1425;--primary:#38d6c2;--primary-dark:#20b6aa;--accent:#162f49}
html,body,.stApp,[data-testid="stApp"],[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#08111f!important}
body,.stApp{font-family:'DM Sans',system-ui,sans-serif;font-size:15px;color:var(--ink)!important}
.stApp{background:radial-gradient(ellipse at 12% 0%,rgba(31,87,130,.28),transparent 38%),radial-gradient(ellipse at 94% 15%,rgba(24,151,143,.14),transparent 30%),#08111f!important}
.stApp *{color:var(--ink)}
p,li,label,[data-testid="stCaptionContainer"],.stMarkdown{color:#d4dfef}
a{color:#55e3d2!important;text-decoration:none}
a:hover{text-decoration:underline}
::placeholder{color:#8194ae!important;opacity:1!important}
[data-testid="stHeader"]{background:rgba(8,17,31,.92)!important}
[data-testid="stAppDeployButton"]{display:none}
[data-testid="stMainBlockContainer"]{max-width:1080px;padding:2rem 1.5rem 9rem;position:relative;z-index:2}
[data-testid="stBottom"],[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{background:#08111f!important}
[data-testid="stBottomBlockContainer"]{max-width:1080px;position:relative;z-index:2;padding-bottom:1rem}
[data-testid="stChatInput"]{background:#101d30!important;border:1px solid rgba(104,148,185,.38)!important;border-radius:20px!important;box-shadow:0 12px 38px rgba(0,0,0,.22),0 0 0 1px rgba(55,214,194,.04)}
[data-testid="stChatInput"]>div,[data-testid="stChatInput"] textarea{background:transparent!important;color:#edf4ff!important}
[data-testid="stChatInput"] textarea{font-size:15px!important}
[data-testid="stChatInputFileUploadButton"],[data-testid="stChatInputSubmitButton"]{background:linear-gradient(135deg,#38d6c2,#258bd0)!important;border:0!important;border-radius:12px!important}
[data-testid="stChatInputFileUploadButton"] svg,[data-testid="stChatInputSubmitButton"] svg{color:#061522!important;fill:#061522!important}
[data-testid="stChatMessage"]{background:linear-gradient(145deg,rgba(20,35,57,.96),rgba(13,25,43,.96));border:1px solid rgba(126,159,199,.18);border-radius:18px;padding:1rem 1.15rem;box-shadow:0 8px 26px rgba(0,0,0,.1)}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){background:linear-gradient(145deg,rgba(22,58,78,.9),rgba(17,42,65,.95));border-color:rgba(56,214,194,.25)}
button[data-testid^="stBaseButton"]{background:#122139!important;color:#dce8f7!important;border:1px solid rgba(140,169,205,.25)!important;border-radius:12px!important;font-weight:600;transition:all .18s ease}
button[data-testid^="stBaseButton"]:hover{border-color:#38d6c2!important;background:#18314a!important;color:#fff!important;transform:translateY(-1px)}
button[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,#38d6c2,#288acb)!important;color:#071522!important;border-color:transparent!important;box-shadow:0 5px 18px rgba(43,198,188,.18)}
button[data-testid="stBaseButton-primary"]:hover{background:linear-gradient(135deg,#5be6d5,#38a6e2)!important;color:#071522!important}
button[data-testid="stBaseButton-primary"] *{color:#071522!important}
[data-testid="stFileUploaderDropzone"]{background:linear-gradient(145deg,rgba(17,33,54,.96),rgba(12,25,43,.96))!important;border:1.5px dashed rgba(56,214,194,.48)!important;border-radius:18px!important;padding:24px!important}
[data-testid="stFileUploaderDropzone"] *{color:#dce8f7!important}
[data-testid="stFileUploaderDropzone"] button{background:#173b4c!important;border:1px solid rgba(56,214,194,.38)!important;color:#66e8d8!important;padding:.55rem 1rem!important}
[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="textarea"],[data-baseweb="select"]>div{background:#101d30!important;border:1px solid rgba(140,169,205,.27)!important;border-radius:11px!important;color:#edf4ff!important}
[data-baseweb="input"] input,[data-baseweb="base-input"] input,[data-baseweb="textarea"] textarea{background:transparent!important;color:#edf4ff!important;-webkit-text-fill-color:#edf4ff!important}
[data-baseweb="select"] *{color:#edf4ff!important}
[data-baseweb="popover"],[data-baseweb="menu"],ul[role="listbox"],[data-testid="stPopoverBody"]{background:#101d30!important}
[data-baseweb="popover"]>div,[data-testid="stPopoverBody"]{border:1px solid rgba(140,169,205,.25)!important;border-radius:13px!important;box-shadow:0 14px 38px rgba(0,0,0,.35)}
li[role="option"]:hover,li[role="option"][aria-selected="true"]{background:#1b354b!important}
[data-testid="stAlert"]{background:#112239!important;border:1px solid rgba(140,169,205,.25)!important;border-radius:13px!important}
[data-testid="stAlert"] *{color:#e0ebf9!important}
[data-testid="stExpander"] details,[data-testid="stStatus"] details{background:#101d30!important;border:1px solid rgba(140,169,205,.2)!important;border-radius:14px!important}
[data-testid="stExpander"] summary,[data-testid="stStatus"] summary{color:#e0ebf9!important}
pre,code{background:#0b1728!important;color:#a8f1e7!important;border:1px solid rgba(140,169,205,.14);border-radius:8px}
/* Dialogs intentionally use a LIGHT surface for reliable contrast across Streamlit versions */
[data-testid="stDialog"],[data-testid="stDialog"]>div,[data-testid="stDialog"] [role="dialog"],div[role="dialog"]{background:#f8fbff!important;background-color:#f8fbff!important;color:#17263b!important;border:1px solid #d4deeb!important;border-radius:20px!important;box-shadow:0 24px 80px rgba(0,0,0,.48)!important}
[data-testid="stDialog"] [data-testid="stDialogHeader"],[data-testid="stDialog"] [data-testid="stDialogContent"],[data-testid="stDialog"] [data-testid="stVerticalBlock"],[data-testid="stDialog"] [data-testid="stMarkdownContainer"],[data-testid="stDialog"] .stMarkdown,div[role="dialog"] [data-testid="stMarkdownContainer"]{background:transparent!important;background-color:transparent!important;color:#17263b!important}
[data-testid="stDialog"] *,div[role="dialog"] *{color:#17263b!important;opacity:1!important}
[data-testid="stDialog"] h1,[data-testid="stDialog"] h2,[data-testid="stDialog"] h3,[data-testid="stDialog"] h4,[data-testid="stDialog"] p,[data-testid="stDialog"] li,[data-testid="stDialog"] label,[data-testid="stDialog"] span,[data-testid="stDialog"] [data-testid="stCaptionContainer"],div[role="dialog"] p,div[role="dialog"] li,div[role="dialog"] label{color:#17263b!important}
[data-testid="stDialog"] a,div[role="dialog"] a{color:#176b80!important;text-decoration:underline}
[data-testid="stDialog"] [data-baseweb="input"],[data-testid="stDialog"] [data-baseweb="base-input"],[data-testid="stDialog"] input,[data-testid="stDialog"] textarea,div[role="dialog"] input{background:#fff!important;background-color:#fff!important;color:#17263b!important;-webkit-text-fill-color:#17263b!important;border:1px solid #b8c6d8!important;border-radius:9px!important}
[data-testid="stDialog"] input::placeholder,div[role="dialog"] input::placeholder{color:#64748b!important;-webkit-text-fill-color:#64748b!important;opacity:1!important}
[data-testid="stDialog"] button,div[role="dialog"] button{color:#17263b!important}
[data-testid="stDialog"] button[data-testid="stBaseButton-primary"],[data-testid="stDialog"] button[data-testid="stBaseButton-primary"] *,div[role="dialog"] button[data-testid="stBaseButton-primary"],div[role="dialog"] button[data-testid="stBaseButton-primary"] *{background:#167e89!important;color:#fff!important;border-color:#167e89!important}
[data-testid="stDialog"] [data-testid="stDialogCloseButton"],div[role="dialog"] [data-testid="stDialogCloseButton"]{background:#eaf0f7!important;border:1px solid #cbd6e4!important}
[data-testid="stDialog"] [data-testid="stDialogCloseButton"] svg,div[role="dialog"] [data-testid="stDialogCloseButton"] svg{color:#17263b!important;fill:#17263b!important}
.brand{display:flex;align-items:center;gap:13px;padding:6px 0 14px}
.brand-mark{width:56px;height:56px;display:grid;place-items:center;border-radius:17px;background:linear-gradient(145deg,#17394b,#10273b);border:1px solid rgba(56,214,194,.42);box-shadow:0 8px 28px rgba(14,190,177,.12),inset 0 1px 0 rgba(255,255,255,.08)}
.brand-mark svg{width:39px;height:39px;filter:drop-shadow(0 0 7px rgba(56,214,194,.23))}
.brand-copy{display:flex;flex-direction:column;gap:2px}
.brand .bt{font:800 25px/1.1 'Manrope',sans-serif;letter-spacing:-.045em;color:#f2f7ff!important}
.brand .bt em{font-style:normal;color:#4fe0cf!important}
.brand .tagline{font-size:11px;font-weight:700;letter-spacing:.19em;text-transform:uppercase;color:#8ea7c2!important}
.brand .neck,.brand .bf{display:none}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 12px}
.chip{border:1px solid rgba(140,169,205,.22);background:rgba(17,33,54,.86);border-radius:999px;padding:6px 12px;font-size:12px;color:#dce8f7!important}
.chip i{font-style:normal;color:#91a8c3!important;margin-left:6px}
.chip.ok{background:rgba(24,119,101,.18);border-color:rgba(56,214,194,.28)}
.chip.ok::before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:#45dfad;margin-right:8px;box-shadow:0 0 9px rgba(69,223,173,.45)}
.hero{text-align:center;margin:5vh 0 3vh;padding:28px 18px 24px;border:1px solid rgba(121,164,200,.16);border-radius:24px;background:radial-gradient(ellipse at 50% 0%,rgba(37,121,144,.2),transparent 65%),linear-gradient(145deg,rgba(17,32,52,.84),rgba(10,22,39,.65));box-shadow:inset 0 1px 0 rgba(255,255,255,.025)}
.hero h1{font:800 clamp(30px,4.2vw,46px)/1.12 'Manrope',sans-serif;letter-spacing:-.05em;margin:0 0 12px;color:#f1f7ff!important}
.hero p,.uphead p,.step span{color:#a4b6cc!important;margin:0}
.uphead{margin:28px 0 12px}
.uphead h3{font:800 23px 'Manrope',sans-serif;letter-spacing:-.03em;color:#f0f6ff!important;margin:0 0 7px}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}
.step{position:relative;overflow:hidden;border:1px solid rgba(121,164,200,.2);background:linear-gradient(145deg,rgba(19,38,61,.96),rgba(12,26,44,.96));border-radius:17px;padding:20px 20px;box-shadow:0 8px 24px rgba(0,0,0,.1)}
.step:before{content:"";position:absolute;top:0;left:18px;right:18px;height:2px;background:linear-gradient(90deg,#38d6c2,#338ee0);opacity:.8}
.step b{display:block;font:700 14px 'Manrope',sans-serif;color:#eff8ff!important;margin-bottom:7px}
.step span{font-size:13px;line-height:1.6;display:block}
.tip{text-align:center;font-size:12px;color:#8298b3!important;margin-top:8px}
.mascot{display:none!important}
[data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3{color:#f0f6ff!important;font-family:'Manrope',sans-serif;letter-spacing:-.025em}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{line-height:1.7}
[data-testid="stPills"] button{border-radius:999px!important}
[data-testid="stPills"] button[aria-pressed="true"]{background:linear-gradient(135deg,#38d6c2,#288acb)!important;color:#071522!important;border-color:transparent!important}
/* Sidebar follows the same IrfanRAG navy / teal visual system as the main canvas */
[data-testid="stSidebar"],[data-testid="stSidebar"]>div:first-child{background:linear-gradient(180deg,#0d1b2d 0%,#08111f 100%)!important;border-right:1px solid rgba(56,214,194,.18)!important}
[data-testid="stSidebar"] [data-testid="stSidebarContent"]{background:transparent!important}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3{color:#edf4ff!important;font-family:'Manrope',sans-serif!important;letter-spacing:-.02em}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,[data-testid="stSidebar"] [data-testid="stCaptionContainer"],[data-testid="stSidebar"] label,[data-testid="stSidebar"] span{color:#c3d2e5!important}
[data-testid="stSidebar"] hr{border-color:rgba(140,169,205,.2)!important}
[data-testid="stSidebar"] [data-baseweb="input"],[data-testid="stSidebar"] [data-baseweb="select"]>div,[data-testid="stSidebar"] [data-testid="stSlider"]{background-color:#101d30!important;border-color:rgba(140,169,205,.28)!important}
[data-testid="stSidebar"] [data-baseweb="select"] *{color:#edf4ff!important}
[data-testid="stSidebar"] [data-baseweb="popover"],[data-testid="stSidebar"] [role="listbox"],[data-testid="stSidebar"] [data-baseweb="menu"]{background:#101d30!important}
[data-testid="stSidebar"] li[role="option"]:hover,[data-testid="stSidebar"] li[role="option"][aria-selected="true"]{background:#1b354b!important}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{background:#38d6c2!important;border-color:#38d6c2!important}
[data-testid="stSidebar"] [data-testid="stExpander"] details{background:#101d30!important;border:1px solid rgba(140,169,205,.22)!important;border-radius:12px!important}
[data-testid="stSidebar"] button[data-testid^="stBaseButton"]{background:#122139!important;color:#dce8f7!important;border:1px solid rgba(140,169,205,.25)!important}
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,#38d6c2,#288acb)!important;color:#071522!important}
@media(max-width:760px){[data-testid="stMainBlockContainer"]{padding:1.2rem .85rem 8rem}.brand .bt{font-size:22px}.brand-mark{width:48px;height:48px}.steps{grid-template-columns:1fr}.hero{margin:3vh 0 2vh;padding:22px 14px}.brand .tagline{font-size:9px}}
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

BRAND = """<div class="brand"><div class="brand-mark"><svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="RAG network logo"><path d="M13 8.5h14l8 8V35a4 4 0 0 1-4 4H13a4 4 0 0 1-4-4V12.5a4 4 0 0 1 4-4Z" stroke="#5CE5D4" stroke-width="2.2" stroke-linejoin="round"/><path d="M27 9v9h8" stroke="#5CE5D4" stroke-width="2.2" stroke-linejoin="round"/><path d="M15 25h7m-7 5h5" stroke="#8BB8FF" stroke-width="2" stroke-linecap="round"/><circle cx="30.5" cy="26" r="3.5" fill="#38D6C2"/><circle cx="27" cy="33" r="2.2" fill="#8BB8FF"/><path d="m28.5 29-1 2M27 33l-4-2m7.5-1 3 2" stroke="#8BB8FF" stroke-width="1.5" stroke-linecap="round"/><circle cx="34" cy="34" r="2.2" fill="#38D6C2"/></svg></div><div class="brand-copy"><div class="bt">Irfan<em>RAG</em></div><div class="tagline">Knowledge · Retrieval · Answers</div></div></div>"""


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

### The ⚙️ Settings menu
Use the visible **RAG Settings · Models · Retrieval** panel to choose Normal and DeepThink models, set chunk size and overlap, and control retrieved passages (1–20). Chunk settings apply to newly indexed files; remove and re-upload a file to re-index it.

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


def settings_panel():
    """Render model and RAG controls in Streamlit's left sidebar."""
    with st.sidebar:
        st.markdown("## ⚙️ RAG Settings")
        st.caption("Models · Chunking · Retrieval")
        st.divider()
        models = list(dict.fromkeys(st.session_state.get("models", [])))
        if not models:
            st.warning("No models available. Check your OpenAI API key and model access.")
        else:
            st.selectbox(
                "Normal mode model", models, key="normal_model",
                help="Model used for standard questions."
            )
            st.selectbox(
                "DeepThink model", models, key="deep_model",
                help="Model used for more detailed reasoning."
            )
        st.divider()
        st.markdown("### Document chunking")
        st.slider(
            "Chunk size (characters)", min_value=300, max_value=4000, step=100,
            key="chunk_size",
            help="Number of characters in each indexed text chunk."
        )
        max_overlap = max(0, st.session_state.chunk_size - 50)
        if st.session_state.chunk_overlap > max_overlap:
            st.session_state.chunk_overlap = min(150, max_overlap)
        st.slider(
            "Chunk overlap (characters)", min_value=0, max_value=max_overlap, step=25,
            key="chunk_overlap",
            help="Text repeated between neighbouring chunks to preserve context."
        )
        st.divider()
        st.markdown("### Retrieval")
        st.slider(
            "Retrieved passages", min_value=1, max_value=20, step=1,
            key="retrieved_passages",
            help="Number of relevant passages supplied to the model for each answer."
        )
        st.caption(
            f"Chunk size: {st.session_state.chunk_size:,} · "
            f"Overlap: {st.session_state.chunk_overlap:,} · "
            f"Passages: {st.session_state.retrieved_passages}"
        )
        st.caption("Chunk settings apply to newly indexed files. Re-upload files to re-index them.")
        with st.expander("More actions"):
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            if st.button("Remove all documents", use_container_width=True):
                st.session_state.kb = new_kb()
                st.rerun()
            if st.button("Sign out", use_container_width=True):
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
    st.set_page_config(page_title="IrfanRAG — Document Intelligence", page_icon="irfanrag-logo.svg", layout="wide", initial_sidebar_state="expanded")
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
    st.session_state.setdefault("chunk_size", 1000)
    st.session_state.setdefault("chunk_overlap", 150)
    st.session_state.setdefault("retrieved_passages", 4)
    kb = st.session_state.kb
    client = make_client(st.session_state.api_key)
    settings_panel()
    pending = st.session_state.pop("pending_q", None)

    if not st.session_state.get("guide_seen"):
        st.session_state.guide_seen = True
        guide_dialog()

    left, help_col = st.columns([9, 1], vertical_alignment="center")
    left.markdown(BRAND, unsafe_allow_html=True)
    if help_col.button("❓", help="User guide"):
        guide_dialog()
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
                hits = retrieve(client, query, st.session_state.retrieved_passages)
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
