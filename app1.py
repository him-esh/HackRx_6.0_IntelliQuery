# app_hackrx.py
# FastAPI app for HackRx — Google Embeddings (text-embedding-004) + FAISS + Groq LLM (LLaMA3)
# Spec endpoint returns answers[]; added explain endpoint with decision/rationale/clauses.
# Supports PDF, DOCX, and Email (EML/MSG). Swagger "Authorize" lock enabled.

import os
import tempfile
import mimetypes
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Optional, Dict, Any

import requests
from dotenv import load_dotenv, find_dotenv

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# LangChain loaders & core
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredEmailLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Embeddings & LLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

# ----------------------------
# Environment bootstrap
# ----------------------------
dotenv_path = find_dotenv(usecwd=True)
if not dotenv_path:
    dotenv_path = str(Path(__file__).parent / ".env")
load_dotenv(dotenv_path, override=True)

print(f"[ENV] Loaded from: {dotenv_path}")
gemini_key_preview = os.getenv("GEMINI_API_KEY") or ""
print(f"[ENV] GEMINI_API_KEY: {gemini_key_preview[:4]+'****' if gemini_key_preview else 'NOT SET'}")

# ----------------------------
# Configuration / Environment
# ----------------------------

# REQUIRED HackRx token (exact from the problem statement)
REQUIRED_BEARER_TOKEN = "f90ea9778886527a4f2328e261c7e7af5db97024b150e84617606706d25d9b86"

# API Keys (read from env; set them before running)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Google embeddings
# Fix: look up GROQ correctly; keep backward compatibility with GROK_API_KEY if used earlier
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in environment.")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY (or GROK_API_KEY) is not set in environment.")

# Embedding & LLM model names
EMBEDDING_MODEL_NAME = "models/text-embedding-004"   # Google
GROQ_MODEL_NAME = "LLaMA3-8b-8192"                   # Valid Groq chat model

# Retrieval parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 6  # number of chunks to retrieve (post-diversification)
CONTEXT_CHAR_LIMIT = 3500
SNIPPET_CHAR_LIMIT = 800
CLAUSE_SNIPPET_LIMIT = 400

# ----------------------------
# FastAPI app
# ----------------------------
API_PREFIX = "/api/v1"
app = FastAPI(
    title="HackRx Retrieval System",
    version="1.1.0",
    swagger_ui_parameters={"persistAuthorization": True},
)

# Swagger "Authorize" support (adds lock button)
bearer_scheme = HTTPBearer()

@app.get("/")
def root():
    return {"message": "HackRx backend is running. See /docs for Swagger UI."}

# ----------------------------
# Request Models
# ----------------------------
class HackRxRequest(BaseModel):
    documents: str         # Blob URL to PDF/DOCX/Email
    questions: List[str]   # list of questions

# ----------------------------
# Helpers
# ----------------------------

def verify_token(credentials: HTTPAuthorizationCredentials):
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization token.")
    if credentials.credentials != REQUIRED_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token.")

def infer_suffix_from_url(url: str) -> str:
    """
    Try to infer the file extension from the URL path; default to '.bin' if unknown.
    """
    path = urlparse(url).path
    ext = Path(path).suffix
    if ext:
        return ext
    # Guess by content-type from name
    ctype, _ = mimetypes.guess_type(path)
    if ctype == "application/pdf":
        return ".pdf"
    return ".bin"

def download_to_temp(url: str) -> str:
    """
    Download a remote file (PDF/DOCX/EML/MSG) to a temporary path and return its file path.
    """
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to download file. HTTP {r.status_code}")
        suffix = infer_suffix_from_url(url)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(r.content)
        tmp.flush()
        tmp.close()
        return tmp.name
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading file: {str(e)}")

def load_docs_from_file(path: str) -> List[Document]:
    """
    Dispatch loader by file extension / mimetype.
    Supported: .pdf, .docx, .eml, .msg
    """
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    if ext == ".docx":
        return Docx2txtLoader(path).load()
    if ext in [".eml", ".msg"]:
        # Requires unstructured + extract-msg installed
        return UnstructuredEmailLoader(path, mode="elements").load()
    # Fallback for PDFs without extension
    ctype, _ = mimetypes.guess_type(path)
    if ctype == "application/pdf":
        return PyPDFLoader(path).load()
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or ctype or 'unknown'}")

def build_vectorstore_from_file(file_path: str) -> FAISS:
    """
    Load file -> split to chunks -> embed with Google text-embedding-004 -> FAISS vectorstore.
    """
    pages: List[Document] = load_docs_from_file(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
    )
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore

def get_retriever(vectorstore: FAISS, k: int = TOP_K):
    """
    Use MMR (diversified) retrieval to reduce duplicate chunks and improve answer quality.
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(20, 5 * k), "lambda_mult": 0.5}
    )

def build_answer_chain() -> StrOutputParser:
    """
    Construct a light-weight chain: Prompt -> Groq LLM -> String parser.
    The prompt restricts the model to answer ONLY from the provided context.
    """
    system_prompt = """You are an expert insurance policy assistant.
Answer the user's question ONLY from the provided policy context.
If the answer is not present in the context, reply exactly: "Not specified in the policy."

Return a concise, one or two sentence answer. Do not fabricate details.
"""
    template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"),
        ]
    )
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL_NAME,
        temperature=0.0,
        max_tokens=None,
    )
    return template | llm | StrOutputParser()

def build_decision_chain() -> StrOutputParser:
    """
    Explainable decision chain: returns compact JSON with decision/rationale based on context.
    """
    system = """You are a compliance-grade policy assistant.
Answer strictly from the provided context.
Return compact JSON with keys:
- decision: one of ["Yes","No","Partially","Not specified"]
- answer: one-sentence direct answer
- rationale: 1-2 sentences referencing the specific clauses
ONLY return JSON. No extra text."""
    template = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question:\n{question}\n\nContext:\n{context}\n\nJSON:")
    ])
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL_NAME, temperature=0.0)
    return template | llm | StrOutputParser()

def short(txt: str, n: int = CLAUSE_SNIPPET_LIMIT) -> str:
    txt = (txt or "").strip()
    return (txt[:n] + " ...") if len(txt) > n else txt

def build_context_blocks(docs: List[Document], max_chars: int = CONTEXT_CHAR_LIMIT) -> str:
    parts, used = [], 0
    for d in docs:
        s = (d.page_content or "").strip()
        if len(s) > SNIPPET_CHAR_LIMIT:
            s = s[:SNIPPET_CHAR_LIMIT] + " ..."
        p = d.metadata.get("page")
        block = (f"[Page {p}] " if p is not None else "") + s
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts) if parts else "(no relevant context)"

def answer_with_context(vectorstore: FAISS, question: str, k: int = TOP_K) -> str:
    """
    Retrieve top-k relevant chunks and answer with Groq LLM using a grounded prompt.
    """
    retriever = get_retriever(vectorstore, k)
    docs = retriever.get_relevant_documents(question)
    context = build_context_blocks(docs)
    chain = build_answer_chain()
    answer = chain.invoke({"question": question, "context": context}).strip()
    return answer

# ----------------------------
# Submission Endpoint (SPEC)
# ----------------------------
@app.post(f"{API_PREFIX}/hackrx/run")
async def hackrx_run(
    payload: HackRxRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    """
    Spec-compliant endpoint:
    - Validates Bearer token (via Swagger Authorize)
    - Downloads remote file (PDF/DOCX/Email)
    - Builds FAISS index with Google embeddings
    - Answers each question grounded in retrieved context
    - Returns: {"answers": ["...", "..."]}
    """
    verify_token(credentials)

    file_path = download_to_temp(payload.documents)
    try:
        vectorstore = build_vectorstore_from_file(file_path)
        answers = [answer_with_context(vectorstore, q, k=TOP_K) for q in payload.questions]
        return JSONResponse(content={"answers": answers})
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass

# ----------------------------
# Explainability Endpoint (Structured JSON)
# ----------------------------
@app.post(f"{API_PREFIX}/hackrx/run_explain")
async def hackrx_run_explain(
    payload: HackRxRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    """
    Returns explainable results per question:
    {
      "results": [
        {
          "question": "...",
          "decision": "Yes|No|Partially|Not specified",
          "answer": "one sentence",
          "rationale": "1-2 sentences",
          "clauses": [
            {"page": 5, "snippet": "..."},
            ...
          ]
        }
      ]
    }
    """
    verify_token(credentials)

    file_path = download_to_temp(payload.documents)
    try:
        vs = build_vectorstore_from_file(file_path)
        retriever = get_retriever(vs, k=TOP_K)
        chain = build_decision_chain()

        results: List[Dict[str, Any]] = []
        for q in payload.questions:
            docs = retriever.get_relevant_documents(q)
            context = build_context_blocks(docs)
            raw = chain.invoke({"question": q, "context": context}).strip()

            # Safe parse fallback
            try:
                import json
                parsed = json.loads(raw)
            except Exception:
                parsed = {
                    "decision": "Not specified",
                    "answer": "Not specified in the policy.",
                    "rationale": "Insufficient or ambiguous context.", 
                }

            results.append({
                "question": q,
                "decision": parsed.get("decision"),
                "answer": parsed.get("answer"),
                "rationale": parsed.get("rationale"),
                "clauses": [
                    {
                        "page": d.metadata.get("page"),
                        "snippet": short(d.page_content, CLAUSE_SNIPPET_LIMIT)
                    } for d in docs[:3]
                ]
            })
        return {"results": results}
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass
