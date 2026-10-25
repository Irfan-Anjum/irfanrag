# RAG Chatbot

Chat with your own documents. Sign in with an OpenAI API key, add PDF, Word, Excel (XLS/XLSX) or CSV files, and ask questions. Answers use only your files and cite the file and the page, sheet or rows they came from.

## Files

| File | Purpose |
|---|---|
| `app.py` | The whole application |
| `requirements.txt` | Python packages |
| `.streamlit/config.toml` | Dark theme (recommended, the app also forces readable colors by itself) |

## Run on your computer

1. Install Python 3.10 or newer.
2. In the project folder run: `pip install -r requirements.txt`
3. Start the app: `streamlit run app.py`
4. Your browser opens. Paste your OpenAI API key in the popup.

## Deploy on GitHub + Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py` and `requirements.txt` to its top level.
2. Create the file `.streamlit/config.toml` in the repository (GitHub: Add file, Create new file, type `.streamlit/config.toml` as the name, paste the contents).
3. Go to share.streamlit.io, sign in with GitHub, choose Create app, pick the repository and branch `main`, and set **Main file path** to `app.py`.
4. Press Deploy. Open the link when it is ready.

Do not put your API key in the repository. Every visitor enters their own key in the popup.

## User guide

**1. Sign in.** Paste your OpenAI API key in the popup. It is verified first and kept only for the current session.

**2. Add documents.** Two ways:
- Use the **Add your documents** box: drag files in, or click **Browse files**, then press **Index**.
- Press the glowing **＋** inside the chat box and attach files there.

Supported: `.pdf`, `.docx`, `.xls`, `.xlsx`, `.csv`. Added files appear as chips under the title. Use **＋ Add more documents** to add more later.

**3. Ask questions.** Type in the chat box and press Enter. Examples:
- PDF/Word: "Summarise the key points." "What does the contract say about termination?"
- Excel/CSV: "What is the total sales for the North region?" "Which row has the highest value?"
- Across files: "Compare the Q1 sheet with the annual report."

**4. Choose a mode** (above the chat box).
- 🍃 **Normal**: fast, uses the 4 most relevant passages.
- ⚡ **DeepThink**: stronger model, 8 passages, compares sources and checks arithmetic.

**5. Check the sources.** Open **🔍 Sources** under an answer to see the exact passages used. If the answer is not in your files, the assistant says so.

**6. Menu (⚙️).** Change the models, clear the chat, remove all documents, or sign out. The ❓ button reopens the in-app guide.

## Limits and tips

- Scanned PDFs (pictures of text) cannot be read. Use text-based PDFs.
- Password-protected files cannot be read.
- Each file is limited to its first 1,500 parts to keep cost reasonable. For huge spreadsheets, add only the sheets you need.
- You pay OpenAI for indexing and answering. The same file is never indexed twice.
- Name the file or sheet in your question when you have many documents.

## Troubleshooting

| Problem | Fix |
|---|---|
| "API key was rejected" | Check the key or create a new one in your OpenAI account |
| Quota or billing error | Add credit to your OpenAI account |
| "I couldn't find that in the documents" | Rephrase the question, or check the right file is added |
| Text hard to read | Make sure you are using this version of `app.py`, then hard-refresh the page (Ctrl+F5) |
| Old version still showing on Streamlit Cloud | Upload the new `app.py` to GitHub; the app redeploys by itself |
