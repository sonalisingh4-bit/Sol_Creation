# Solution Creation Tool

Upload a **question paper**, and export a complete **solution document**
(`.docx` + `.pdf`) with a marks-appropriate model answer for every question —
in the language you choose. Works for **any class (6–12), any board (CBSE /
ICSE / State), and NEET & JEE** papers; the primary content set is the
foundation library (Class 6–10, CBSE & ICSE). Answers are grounded in a book
knowledge base that is prepared centrally and downloaded automatically —
**faculty never see or manage knowledge-base files**; they only upload the
question paper.

Built with **FastAPI + HTMX** and **Google Gemini** (parsing, embeddings, answers).

## How it works

```
ADMIN (once per content update)                    FACULTY (every machine)
Google Drive folder of foundation books            clone repo + pip install
        │  scripts/build_kb.py                             │  python main.py
        ▼                                                  ▼
download → auto-tag by Board/Class/Subject         app auto-downloads the
→ chunk → Gemini embeddings → vector index         prebuilt index zip from
→ pack ONE zip → upload zip to Drive               Drive on first start
        └──────────── KB_INDEX_URL in .env ────────────────┘

Question paper ──▶ Gemini parses questions + marks ──▶ per question: retrieve
   board/class/subject-matched passages ──▶ marks-aware answer ──▶ .docx + .pdf
```

Why this design: the book files are far too large for GitHub. They stay on
**Google Drive (free)**; git only carries code. The heavy work (embedding) is
done once by the admin — faculty machines just download the ready-made index.

## Faculty setup (each machine)

```bash
# 1. install dependencies
.venv/Scripts/python -m pip install -r requirements.txt

# 2. configure
copy .env.example .env      # then edit .env:
#    GEMINI_API_KEY=...     key from https://aistudio.google.com/apikey
#    KB_INDEX_URL=...       index-zip link provided by the admin

# 3. run
python main.py              # http://127.0.0.1:8000
```

On first start the app downloads and unpacks the knowledge base automatically
in the background (a small notice shows until it finishes; the knowledge base
is otherwise invisible to faculty). Then: upload the paper, pick **Class /
Exam** (Class 6–12, NEET or JEE), **Board**, **Subject** and **Answer
language**, and click *Generate*. Every field except the paper is optional —
retrieval falls back gracefully — but picking the right class/board gives the
best-matched answers. NEET/JEE switch the answers into a brief MCQ style
(final option first, concise justification). You get **.docx** and **.pdf**
downloads when it finishes.

## Admin: build & publish the knowledge base

1. Keep the foundation books in one Google Drive folder, organised so each
   file's path names its board, class and subject — any of these layouts work:
   - `CBSE/Class 8/Science/ch04.pdf`
   - `ICSE/Class 9/Physics/Light.pdf`
   - `Maths/8th CBSE/chapter 2.pdf`

   Tags are read case-insensitively from the whole path: board (`CBSE`/`NCERT`,
   `ICSE`/`Selina`, `State Board`), class 6–12 (`Class 8`, `Grade 8`, `8th`,
   `Std VIII`, `Class XII`), subject (Mathematics, Science, Physics, Chemistry,
   Biology, Social Science/SST, English). A file with no board tag serves every
   board; no class tag means it is a fallback for all classes.

2. Share the folder: right-click → Share → **“Anyone with the link – Viewer”**
   (required — without this nothing can download it), and put the link in
   `.env` as `KB_DRIVE_FOLDER`.

3. Build:

   ```bash
   python scripts/build_kb.py --dry-run   # preview how every file is tagged
   python scripts/build_kb.py             # download → tag → embed → index → zip
   ```

   This produces `data/kb_foundation_index.zip`.

4. Upload that zip to Drive, share it as “Anyone with the link – Viewer”, and
   give the link to faculty for `KB_INDEX_URL` in their `.env`. Re-run steps
   3–4 whenever books change (re-runs only download new files); faculty
   delete `data/index` and restart to pick up a new index.

Notes:
- Without `DRIVE_API_KEY`, folder listing uses the keyless downloader, which
  Google caps at **50 files per subfolder** (chapterwise subfolders are fine).
  For bigger flat folders, create a free Google API key with the **Drive API**
  enabled and set `DRIVE_API_KEY` in `.env`.
- Scanned/image books are OCR'd via Gemini during the build.

## Multilingual & marks-aware

- **Marks-aware** — a 1-mark question gets a crisp sentence; a 10-mark question
  gets a structured long answer.
- **Multilingual** — English, Hindi, Bangla, Marathi, Gujarati, Kannada, Telugu,
  Tamil, Malayalam, Punjabi, Odia, Urdu (edit `LANGUAGES` in `app/config.py`).
- **Board/class-true retrieval** — a Class 8 ICSE Physics paper retrieves from
  Class 8 ICSE material first, then falls back to the subject's wider material
  only when nothing class-specific matches.
- **Scanned papers/images** — PDFs with no extractable text are OCR'd by Gemini.

## PDF export

The `.docx` is always produced. The `.pdf` is rendered from it using
**Microsoft Word** (`docx2pdf`) or **LibreOffice** (`soffice`), whichever is
installed. If neither is present you still get the `.docx`.

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** Your Gemini key. |
| `KB_INDEX_URL` | — | Drive share link of the prebuilt index zip (faculty). |
| `KB_DRIVE_FOLDER` | — | Drive folder of source books (admin, build_kb.py). |
| `DRIVE_API_KEY` | — | Optional: Drive API key for full folder listings (admin). |
| `GEMINI_PARSE_MODEL` | `gemini-2.5-flash` | Parsing the paper / OCR. |
| `GEMINI_GEN_MODEL` | `gemini-2.5-pro` | Writing the answers. |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | Embedding sources & queries. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `200` | Source chunking. |
| `TOP_K` | `6` | Passages retrieved per question. |

## Project layout

```
app/
  config.py         settings, boards/classes/subjects, languages
  drive_sync.py     Google Drive download + path→(board,class,subject) tagging
                    + auto-download of the prebuilt index at startup
  gemini_client.py  embeddings, generation, file upload (google-genai)
  extract.py        text extraction (pdf/docx/txt/image) + OCR fallback
  chunking.py       boundary-aware overlapping chunker
  vectorstore.py    persistent NumPy cosine-search index (board/class/subject tags)
  ingest.py         extract → chunk → embed → store one source
  paper_parser.py   Gemini parses paper → structured questions + marks
  solver.py         retrieve + marks-aware multilingual answer per question
  document.py       render SolvedPaper → .docx (+ pdf)
  pdf_convert.py    docx → pdf via Word or LibreOffice
  jobs.py           background generation jobs with live progress
  main.py           FastAPI routes
scripts/
  build_kb.py       ADMIN: Drive folder → tagged, embedded index → one zip
templates/  static/ HTMX UI
```

> Answers are AI-generated. Review them before use.
