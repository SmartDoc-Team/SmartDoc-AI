# SmartDoc AI - Flow Xu Ly Va Chuc Nang Ham

Tai lieu nay la ban tong hop de doc nhanh tren GitHub.
Muc tieu: nhin 1 file la hieu duoc he thong dang chay nhu the nao, module nao lam viec gi, va can tinh chinh o dau.

---

## Muc Luc

- [1. Tong Quan He Thong](#1-tong-quan-he-thong)
- [2. Flow End-to-End](#2-flow-end-to-end)
- [3. Module Va Chuc Nang](#3-module-va-chuc-nang)
- [4. Bien Moi Truong Quan Trong](#4-bien-moi-truong-quan-trong)
- [5. Input Output Cua He Thong](#5-input-output-cua-he-thong)
- [6. Diem Nghen Hieu Nang Va Cach Giam](#6-diem-nghen-hieu-nang-va-cach-giam)
- [7. Checklist Debug Nhanh](#7-checklist-debug-nhanh)

---

## 1. Tong Quan He Thong

SmartDoc AI gom 4 khoi chinh:

1. UI va dieu phoi: `app.py`, `smartdoc/ui.py`
2. Nap va tien xu ly tai lieu: `smartdoc/document_loaders.py`
3. RAG pipeline: `smartdoc/rag.py`
4. Ollama client (text/vision/stream): `smartdoc/ollama_client.py`

Khoi benchmark chunking nam o `smartdoc/benchmarking.py`.

---

## 2. Flow End-to-End

## 2.1 Khoi dong app

1. `app.py` goi `smartdoc.ui.main()`
2. `main()`:
   - setup Streamlit page
   - khoi tao session state
   - check Ollama (`health_check`)
3. Sidebar:
   - `chunk_size`, `chunk_overlap`
   - history
   - clear history
   - clear vector store

## 2.2 Upload va parse tai lieu

1. User upload 1 hoac nhieu file (`pdf`, `docx`)
2. UI luu file vao `data/`
3. Goi `load_document(path, progress_callback=...)`
4. Progress duoc day ra UI va `logs/ingestion.log`

### PDF extraction strategy

Thu tu fallback trong `extract_pdf_pages(...)`:

1. `pypdf` text extraction
2. `pdfplumber` text extraction
3. Fast OCR bang Tesseract (neu can)
4. Selective vision OCR (chi cho page OCR yeu)
5. Full vision OCR neu cac buoc tren khong dat

Render anh PDF:

1. PyMuPDF
2. pypdfium2
3. pdf2image

### DOCX extraction strategy

Thu tu fallback trong `extract_docx_text(...)`:

1. `python-docx`
2. `Docx2txtLoader`
3. Neu text yeu va bat cờ: OCR anh nhung trong DOCX

## 2.3 Hoi dap RAG

1. UI hop nhieu file thanh `combined_doc`
2. `RAGPipeline.answer_question(...)`:
   - rewrite follow-up (co dieu kien, tuy config)
   - split document thanh chunk
   - embed chunk
   - retrieval hybrid (FAISS + BM25 + RRF)
   - source balancing truoc rerank
   - cross-encoder rerank
   - build prompt (co source/page label)
   - stream token tu Ollama
   - persist vector artifacts

## 2.4 Streaming tra loi

1. UI goi `answer_question(..., stream=True, token_callback=...)`
2. Pipeline goi `ollama.generate_stream(...)`
3. Token render theo thoi gian thuc
4. Ket thuc: hien answer + confidence + citations

---

## 3. Module Va Chuc Nang

## 3.1 `app.py`

| Ham      | Vai tro                         |
| -------- | ------------------------------- |
| `main()` | Entry point, chay Streamlit app |

## 3.2 `smartdoc/config.py`

| Ham/Thanh phan                  | Vai tro                            |
| ------------------------------- | ---------------------------------- |
| `_load_env_file(env_path)`      | Nap bien moi truong tu `.env`      |
| `_env_bool(name, default)`      | Parse bool env flag                |
| `Settings`                      | Chua toan bo runtime config        |
| `Settings.ensure_directories()` | Tao `data`, `logs`, `vector_store` |

## 3.3 `smartdoc/ollama_client.py`

| Ham                                    | Vai tro                           |
| -------------------------------------- | --------------------------------- |
| `health_check()`                       | Kiem tra ket noi va model ton tai |
| `generate(prompt, images=None)`        | Non-stream generation             |
| `generate_stream(prompt, images=None)` | Streaming generation              |

## 3.4 `smartdoc/document_loaders.py`

### Data structure

| Lop              | Field                                         |
| ---------------- | --------------------------------------------- |
| `LoadedDocument` | `text`, `pages`, `source_name`, `source_type` |

### Utility

| Ham                                           | Vai tro                                        |
| --------------------------------------------- | ---------------------------------------------- |
| `_report(progress_callback, message)`         | Day progress ra UI/log                         |
| `normalize_text(text)`                        | Chuan hoa text (unicode/newline/control chars) |
| `_resolve_source_type(path)`                  | Nhan dien `pdf` hoac `docx`                    |
| `load_document(file_path, progress_callback)` | Dieu phoi parse theo loai file                 |

### PDF

| Ham                                                            | Vai tro                             |
| -------------------------------------------------------------- | ----------------------------------- |
| `_is_bad_pdf(pages)`                                           | Danh gia chat luong text extraction |
| `_render_pdf_images_*`                                         | Render PDF -> image                 |
| `_prepare_ocr_image(img, max_side)`                            | Resize truoc OCR                    |
| `_extract_pdf_pages_with_tesseract_plus_selective_vision(...)` | Fast OCR + selective vision         |
| `_extract_pdf_pages_with_vision(...)`                          | Vision OCR                          |
| `extract_pdf_pages(...)`                                       | Pipeline fallback tong              |

### DOCX

| Ham                                                          | Vai tro                             |
| ------------------------------------------------------------ | ----------------------------------- |
| `extract_docx_text(path, progress_callback)`                 | Fallback parser + OCR image neu can |
| `_extract_docx_text_with_python_docx(path)`                  | Parse paragraph + table             |
| `_extract_docx_text_with_loader(path)`                       | Parse bang LangChain loader         |
| `_extract_docx_embedded_image_text(path, progress_callback)` | OCR anh nhung trong DOCX            |
| `_heading_level_from_style(style_name)`                      | Rut heading level                   |
| `_table_to_markdown_rows(table)`                             | Chuyen table sang markdown          |

## 3.5 `smartdoc/rag.py`

### Data structure

| Lop                     | Field                                             |
| ----------------------- | ------------------------------------------------- |
| `RetrievalResult`       | `answer`, `chunks`, `sources`, `chunk_count`      |
| `CachedRetrievalAssets` | `chunk_docs`, `embeddings`, `index`, `bm25_index` |

### Chuc nang chinh

| Ham                                   | Vai tro                                     |
| ------------------------------------- | ------------------------------------------- |
| `answer_question(...)`                | Flow RAG full + streaming + persist         |
| `_should_rewrite_question(...)`       | Rewrite follow-up co dieu kien              |
| `_split_document(...)`                | Split chunk + metadata source/page          |
| `_document_cache_key(...)`            | Tao key theo document + chunk config        |
| `_prepare_retrieval_assets(...)`      | Tai su dung cache retrieval assets          |
| `_retrieve_documents(...)`            | Hybrid retrieve + rerank + source balancing |
| `_select_docs_diverse_by_source(...)` | Chon chunk da dang nguon                    |
| `_build_prompt(...)`                  | Prompt co source/page labels                |
| `_persist_index(...)`                 | Luu `.faiss`, `.metadata.json`, `.txt`      |

## 3.6 `smartdoc/ui.py`

| Ham                                         | Vai tro                                     |
| ------------------------------------------- | ------------------------------------------- |
| `_append_ingestion_log(message)`            | Ghi tien trinh ingest/OCR                   |
| `_uploaded_files_signature(uploaded_files)` | Hash bo file upload de cache                |
| `_merge_loaded_documents(target_docs)`      | Hop nhieu file thanh 1 dau vao RAG          |
| `_create_message(...)`                      | Chuan hoa chat message                      |
| `get_pipeline()`                            | Cache `RAGPipeline` bang Streamlit resource |
| `main()`                                    | Orchestrate upload, ingest, chat, streaming |

Luu y:

- Ho tro multi-file (`accept_multiple_files=True`)
- Co cache theo signature de tranh parse lai file khong doi
- Co cache version de tranh dung stale session data

## 3.7 `smartdoc/benchmarking.py`

| Ham/Lop                                             | Vai tro                                   |
| --------------------------------------------------- | ----------------------------------------- |
| `BenchmarkQuery`, `BenchmarkResult`, `BenchmarkRun` | Cac data model benchmark                  |
| `evaluate_configuration(...)`                       | Danh gia 1 cap `chunk_size/chunk_overlap` |
| `run_chunk_benchmark(...)`                          | Chay full benchmark grid                  |
| `format_report_markdown(...)`                       | Tao report markdown                       |

---

## 4. Bien Moi Truong Quan Trong

## Core LLM/RAG

- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `OLLAMA_TIMEOUT_SECONDS`
- `OLLAMA_TEMPERATURE`
- `OLLAMA_KEEP_ALIVE`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVAL_K`
- `MAX_CONTEXT_CHARS`

## Rewrite follow-up

- `REWRITE_FOLLOWUP_ENABLED`
- `REWRITE_FOLLOWUP_MAX_TOKENS`

## OCR chung

- `OCR_RENDER_DPI`
- `OCR_MAX_IMAGE_SIDE`
- `OCR_JPEG_QUALITY`
- `OCR_ENABLE_TESSERACT`
- `OCR_TESSERACT_LANG`
- `OCR_TESSERACT_MIN_CHARS`
- `OCR_VISION_FALLBACK_MAX_PAGES`

## DOCX image OCR

- `OCR_DOCX_IMAGE_ENABLED`
- `OCR_DOCX_IMAGE_MAX_ITEMS`

---

## 5. Input Output Cua He Thong

## Input

- File upload: PDF, DOCX
- Cau hoi cua user
- Cac bien moi truong trong `.env`

## Output

- Cau tra loi stream tren UI
- Citation/context theo chunk
- Log ingest/OCR: `logs/ingestion.log`
- Vector artifacts: `vector_store/*.faiss`, `*.metadata.json`, `*.txt`

---

## 6. Diem Nghen Hieu Nang Va Cach Giam

## Nghen chinh

1. OCR (nhat la vision OCR) ton thoi gian
2. Tai lieu dai -> chunk nhieu -> retrieval/rerank cham
3. Generation co the cham voi context lon

## Cach giam

1. Uu tien text extraction truoc OCR
2. Dung Tesseract + selective vision thay vi full vision OCR
3. Dieu chinh `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MAX_CONTEXT_CHARS`
4. Bat cache tai UI va retrieval assets
5. Giu model nong bang `OLLAMA_KEEP_ALIVE`

---

## 7. Checklist Debug Nhanh

Khi chat sai hoac tra loi "khong du thong tin":

1. Kiem tra `logs/ingestion.log` xem file da load thanh cong chua
2. Kiem tra `Context & Citations` co du source/page chua
3. Neu multi-file, dam bao context co chunk tu nhieu source
4. Thu giam `chunk_size` hoac tang `retrieval_k`
5. Neu file scan/anh, kiem tra OCR flags trong `.env`

---

## Tom Tat

Flow hien tai:

Upload -> Parse/OCR (text-first) -> Chunk -> Hybrid Retrieve -> Rerank -> Prompt -> Stream Answer -> Citation + Persist

Module trung tam:

- `smartdoc/document_loaders.py`
- `smartdoc/rag.py`
- `smartdoc/ui.py`
- `smartdoc/ollama_client.py`

Tai lieu nay uu tien tinh de doc va de bao tri khi day len GitHub.
