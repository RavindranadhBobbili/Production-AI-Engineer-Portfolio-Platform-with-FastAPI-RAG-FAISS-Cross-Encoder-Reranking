from __future__ import annotations

import os, time, json, uuid, joblib, sqlite3
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


PORT = int(os.getenv("PORT", "8390"))
DB_PATH = "revised_hf_ai_engineer_final.db"
MODEL_PATH = "financial_sentiment_model.joblib"


try:
    import faiss
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_EMBEDDINGS = True
except Exception:
    HAS_EMBEDDINGS = False


def now() -> str:
    return datetime.utcnow().isoformat()


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS runs(
        id TEXT PRIMARY KEY,
        workflow TEXT,
        input_json TEXT,
        output_json TEXT,
        latency_ms REAL,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()


def log_run(workflow: str, input_data: Dict[str, Any], output_data: Dict[str, Any], start: float) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            workflow,
            json.dumps(input_data, default=str),
            json.dumps(output_data, default=str),
            round((time.time() - start) * 1000, 2),
            now(),
        )
    )
    conn.commit()
    conn.close()


class BuildRAGRequest(BaseModel):
    max_rows: int = 1000


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class TrainRequest(BaseModel):
    max_rows: int = 3000


class SentimentRequest(BaseModel):
    text: str


def load_qa_dataset(max_rows: int) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    errors = []

    try:
        ds = load_dataset("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source", split="train")
        docs = []

        for i, row in enumerate(ds):
            if i >= max_rows:
                break

            contexts = row.get("CONTEXTS", [])
            context = " ".join(contexts) if isinstance(contexts, list) else str(contexts)

            docs.append({
                "id": str(i),
                "title": row.get("QUESTION", ""),
                "content": context,
                "label": row.get("final_decision", ""),
                "dataset": "PubMedQA"
            })

        if docs:
            return docs, "PubMedQA", None

    except Exception as e:
        errors.append(f"PubMedQA failed: {e}")

    try:
        ds = load_dataset("rajpurkar/squad", split="train")
        docs = []

        for i, row in enumerate(ds):
            if i >= max_rows:
                break

            answers = row.get("answers", {})
            label = ""
            if isinstance(answers, dict) and answers.get("text"):
                label = answers["text"][0]

            docs.append({
                "id": str(i),
                "title": row.get("question", ""),
                "content": row.get("context", ""),
                "label": label,
                "dataset": "SQuAD fallback"
            })

        if docs:
            return docs, "SQuAD fallback", "; ".join(errors) if errors else None

    except Exception as e:
        errors.append(f"SQuAD failed: {e}")

    demo_docs = [
        {
            "title": "What is retrieval augmented generation?",
            "content": "Retrieval augmented generation combines document retrieval with language generation to answer questions using external evidence.",
            "label": "RAG"
        },
        {
            "title": "What is model evaluation?",
            "content": "Model evaluation measures relevance, groundedness, hallucination risk, latency, robustness, and task-specific performance before deployment.",
            "label": "evaluation"
        },
        {
            "title": "What is FAISS?",
            "content": "FAISS is a vector similarity search library used for semantic search, nearest-neighbor retrieval, and RAG systems.",
            "label": "FAISS"
        },
        {
            "title": "What is financial sentiment analysis?",
            "content": "Financial sentiment analysis classifies financial news or company text as positive, neutral, or negative.",
            "label": "sentiment"
        },
    ]

    docs = []
    for i, row in enumerate(demo_docs[:max_rows]):
        docs.append({
            "id": str(i),
            "title": row["title"],
            "content": row["content"],
            "label": row["label"],
            "dataset": "local_demo_fallback"
        })

    return docs, "local_demo_fallback", "; ".join(errors)


class RevisedRAGStore:
    def __init__(self) -> None:
        self.docs: List[Dict[str, Any]] = []
        self.mode = "not_built"
        self.dataset_used = "none"
        self.fallback_reason = None

        self.tfidf = TfidfVectorizer(stop_words="english", max_features=30000)
        self.tfidf_matrix = None

        self.embedding_model = None
        self.cross_encoder = None
        self.faiss_index = None

    def build_qa(self, max_rows: int = 1000) -> Dict[str, Any]:
        start = time.time()

        self.docs, self.dataset_used, self.fallback_reason = load_qa_dataset(max_rows)

        if not self.docs:
            raise ValueError("No QA documents loaded.")

        if HAS_EMBEDDINGS:
            try:
                self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

                texts = [d["content"] for d in self.docs]
                emb = self.embedding_model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False
                ).astype("float32")

                self.faiss_index = faiss.IndexFlatIP(emb.shape[1])
                self.faiss_index.add(emb)

                self.mode = "faiss_sentence_transformer_cross_encoder"

            except Exception as e:
                self.tfidf_matrix = self.tfidf.fit_transform([d["content"] for d in self.docs])
                self.mode = "tfidf_fallback_after_embedding_error"
                self.fallback_reason = (
                    f"{self.fallback_reason}; embedding fallback: {e}"
                    if self.fallback_reason else str(e)
                )
        else:
            self.tfidf_matrix = self.tfidf.fit_transform([d["content"] for d in self.docs])
            self.mode = "tfidf_fallback"

        out = {
            "status": "built",
            "dataset_used": self.dataset_used,
            "documents": len(self.docs),
            "retrieval_mode": self.mode,
            "fallback_reason": self.fallback_reason,
        }

        log_run("build_qa_rag", {"max_rows": max_rows}, out, start)
        return out

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.docs:
            raise ValueError("RAG index not built. Click Build QA RAG first.")

        top_k = max(1, min(top_k, len(self.docs)))

        if self.mode.startswith("faiss"):
            q_emb = self.embedding_model.encode(
                [query],
                normalize_embeddings=True
            ).astype("float32")

            scores, idx = self.faiss_index.search(q_emb, min(top_k * 5, len(self.docs)))

            candidates = []
            for score, i in zip(scores[0], idx[0]):
                if i == -1:
                    continue

                doc = self.docs[int(i)]
                candidates.append({
                    **doc,
                    "vector_score": float(score)
                })

            if self.cross_encoder and candidates:
                pairs = [(query, c["content"]) for c in candidates]
                rerank_scores = self.cross_encoder.predict(pairs)
            else:
                rerank_scores = [c["vector_score"] for c in candidates]

            reranked = [
                {**c, "rerank_score": float(rs)}
                for c, rs in zip(candidates, rerank_scores)
            ]

            reranked = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)

            return [
                {
                    "rank": rank + 1,
                    "title": r["title"],
                    "content": r["content"][:1500],
                    "label": r["label"],
                    "dataset": r["dataset"],
                    "vector_score": r["vector_score"],
                    "rerank_score": r["rerank_score"],
                }
                for rank, r in enumerate(reranked[:top_k])
            ]

        q = self.tfidf.transform([query])
        scores = cosine_similarity(q, self.tfidf_matrix).flatten()
        idx = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "rank": rank + 1,
                "title": self.docs[int(i)]["title"],
                "content": self.docs[int(i)]["content"][:1500],
                "label": self.docs[int(i)]["label"],
                "dataset": self.docs[int(i)]["dataset"],
                "tfidf_score": float(scores[int(i)])
            }
            for rank, i in enumerate(idx)
        ]

    def answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        start = time.time()

        retrieved = self.search(query, top_k)

        context = " ".join([r["content"] for r in retrieved])
        q_words = set(query.lower().split())
        c_words = set(context.lower().split())

        context_relevance = len(q_words & c_words) / max(len(q_words), 1)

        if retrieved:
            answer = (
                "Based on retrieved evidence: "
                + " ".join([r["content"] for r in retrieved[:2]])[:1800]
            )
            confidence = min(0.92, 0.45 + context_relevance)
        else:
            answer = "Insufficient evidence found."
            confidence = 0.15

        out = {
            "query": query,
            "answer": answer,
            "confidence": round(float(confidence), 3),
            "retrieved": retrieved,
            "context_relevance": round(float(context_relevance), 3),
            "hallucination_risk": round(1 - min(context_relevance, 1), 3),
            "retrieval_mode": self.mode,
            "dataset_used": self.dataset_used,
            "fallback_reason": self.fallback_reason,
        }

        log_run("rag_query", {"query": query, "top_k": top_k}, out, start)
        return out


class FinancialSentimentModel:
    label_map = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    def load_financial_dataset(self, max_rows: int) -> Tuple[pd.DataFrame, str, Optional[str]]:
        errors = []

        dataset_attempts = [
            ("zeroshot/twitter-financial-news-sentiment", None),
            ("nickmuchi/financial-classification", None),
        ]

        for dataset_name, config in dataset_attempts:
            try:
                if config:
                    ds = load_dataset(dataset_name, config, split="train")
                else:
                    ds = load_dataset(dataset_name, split="train")

                rows = []

                for i, row in enumerate(ds):
                    if i >= max_rows:
                        break

                    text = (
                        row.get("sentence")
                        or row.get("text")
                        or row.get("title")
                        or row.get("headline")
                        or row.get("content")
                        or ""
                    )

                    raw_label = row.get("label")

                    if isinstance(raw_label, str):
                        low = raw_label.lower()
                        if "neg" in low:
                            label = 0
                        elif "pos" in low:
                            label = 2
                        else:
                            label = 1
                    else:
                        label = int(raw_label)

                    if text.strip():
                        rows.append({"text": text, "label": label})

                df = pd.DataFrame(rows)

                if not df.empty and df["label"].nunique() >= 2:
                    return df, dataset_name, None

            except Exception as e:
                errors.append(f"{dataset_name} failed: {e}")

        demo_rows = [
            {"text": "The company reported strong revenue growth and improved profitability.", "label": 2},
            {"text": "The firm announced weaker margins and declining sales.", "label": 0},
            {"text": "The company said results were in line with expectations.", "label": 1},
            {"text": "Operating profit increased significantly during the quarter.", "label": 2},
            {"text": "The group expects challenging market conditions next year.", "label": 0},
            {"text": "Management did not change its previous financial guidance.", "label": 1},
            {"text": "Revenue increased and the company raised its outlook.", "label": 2},
            {"text": "Losses widened due to weak demand.", "label": 0},
            {"text": "The board maintained its dividend policy.", "label": 1},
            {"text": "Shares rose after the company beat earnings expectations.", "label": 2},
            {"text": "The company warned that profits may decline next quarter.", "label": 0},
            {"text": "The announcement had no material impact on guidance.", "label": 1},
        ]

        return pd.DataFrame(demo_rows[:max_rows]), "local_financial_demo_fallback", "; ".join(errors)

    def train(self, max_rows: int = 3000) -> Dict[str, Any]:
        start = time.time()

        df, dataset_used, fallback_reason = self.load_financial_dataset(max_rows)

        if df.empty or df["label"].nunique() < 2:
            raise ValueError("Not enough labeled financial sentiment data to train.")

        stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
        test_size = 0.2 if len(df) >= 20 else 0.4

        X_train, X_test, y_train, y_test = train_test_split(
            df["text"],
            df["label"],
            test_size=test_size,
            random_state=42,
            stratify=stratify
        )

        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                max_features=30000,
                ngram_range=(1, 2)
            )),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
        ])

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        labels_present = sorted(df["label"].unique().tolist())

        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "dataset_used": dataset_used,
            "fallback_reason": fallback_reason,
            "classification_report": classification_report(
                y_test,
                pred,
                labels=labels_present,
                target_names=[self.label_map.get(i, str(i)) for i in labels_present],
                output_dict=True,
                zero_division=0
            )
        }

        joblib.dump({"model": model, "metrics": metrics}, MODEL_PATH)

        out = {
            "status": "trained",
            "metrics": metrics,
            "model_path": MODEL_PATH
        }

        log_run("train_financial_sentiment", {"max_rows": max_rows}, out, start)
        return out

    def predict(self, text: str) -> Dict[str, Any]:
        start = time.time()

        if not os.path.exists(MODEL_PATH):
            raise ValueError("Model not trained. Train financial sentiment model first.")

        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]

        pred = int(model.predict([text])[0])

        probabilities = {}

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([text])[0]
            classes = model.named_steps["clf"].classes_
            probabilities = {
                self.label_map.get(int(cls), str(cls)): float(prob)
                for cls, prob in zip(classes, proba)
            }

        out = {
            "text": text,
            "prediction": self.label_map.get(pred, str(pred)),
            "probabilities": probabilities,
            "metrics": bundle["metrics"]
        }

        log_run("financial_sentiment_predict", {"text": text}, out, start)
        return out


def generate_artifacts() -> Dict[str, str]:
    files = {
        "requirements.txt": """fastapi
uvicorn
datasets==3.6.0
transformers
sentence-transformers
faiss-cpu
scikit-learn
pandas
numpy
joblib
nest_asyncio
""",
        "Dockerfile": """FROM python:3.11-slim
WORKDIR /app
COPY revised_hf_ai_engineer_platform_final.py /app/
COPY requirements.txt /app/
RUN pip install -r requirements.txt
EXPOSE 8381
CMD ["python", "revised_hf_ai_engineer_platform_final.py"]
""",
        "render.yaml": """services:
  - type: web
    name: production deployment final
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: python revised_hf_ai_engineer_platform_final.py
    autoDeploy: true
""",
    }

    for name, content in files.items():
        with open(name, "w", encoding="utf-8") as f:
            f.write(content.strip())

    return {k: "created" for k in files}


init_db()
rag = RevisedRAGStore()
sentiment = FinancialSentimentModel()

app = FastAPI(
    title="Generate the complete Python file",
    version="5.0.0"
)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<html>
<head>
<title> Platform deployment Final</title>
<style>
body{margin:0;background:#020617;color:#e5e7eb;font-family:Arial}
.header{padding:35px;text-align:center;background:#0f172a}
.header h1{color:#38bdf8}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:22px;padding:28px}
.card{background:#111827;padding:22px;border-radius:18px;border:1px solid #334155}
textarea,input,button{width:100%;padding:12px;margin-top:10px;border-radius:10px;border:0}
textarea,input{background:#020617;color:#e5e7eb;border:1px solid #475569}
button{background:linear-gradient(90deg,#2563eb,#06b6d4);color:white;font-weight:bold}
pre{background:#020617;padding:18px;border-radius:14px;max-height:600px;overflow:auto;color:#bbf7d0}
</style>
</head>
<body>
<div class="header">
<h1>Production deployment final</h1>
<p>Robust Dataset Fallback | FAISS RAG | CrossEncoder | Financial NLP | FastAPI</p>
</div>

<div class="grid">
<div class="card">
<h2>1. Build QA RAG</h2>
<input id="ragrows" value="1000">
<button onclick="buildRag()">Build QA RAG</button>
</div>

<div class="card">
<h2>2. Ask RAG</h2>
<textarea id="ragq">What is retrieval augmented generation?</textarea>
<button onclick="askRag()">Ask RAG</button>
</div>

<div class="card">
<h2>3. Train Financial Sentiment</h2>
<input id="trainrows" value="3000">
<button onclick="trainSentiment()">Train Financial Sentiment Model</button>
</div>

<div class="card">
<h2>4. Predict Financial Sentiment</h2>
<textarea id="senttext">The company reported strong revenue growth and improved profitability.</textarea>
<button onclick="predictSentiment()">Predict Sentiment</button>
</div>

<div class="card">
<h2>5. Deployment / Monitoring</h2>
<button onclick="artifacts()">Generate Files</button>
<button onclick="observe()">Observability</button>
<button onclick="reqmap()">Requirements Map</button>
</div>
</div>

<div style="padding:28px">
<h2>Output</h2>
<pre id="out">Ready...</pre>
</div>

<script>
async function show(r){out.textContent=JSON.stringify(await r.json(),null,2)}

async function buildRag(){
 show(await fetch('/rag/build-qa',{
  method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({max_rows:parseInt(ragrows.value)})
 }))
}

async function askRag(){
 show(await fetch('/rag/query',{
  method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({query:ragq.value,top_k:5})
 }))
}

async function trainSentiment(){
 show(await fetch('/sentiment/train',{
  method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({max_rows:parseInt(trainrows.value)})
 }))
}

async function predictSentiment(){
 show(await fetch('/sentiment/predict',{
  method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({text:senttext.value})
 }))
}

async function artifacts(){show(await fetch('/generate-artifacts',{method:'POST'}))}
async function observe(){show(await fetch('/observability'))}
async function reqmap(){show(await fetch('/requirements-map'))}
</script>
</body>
</html>
"""


@app.post("/rag/build-qa")
def build_qa(req: BuildRAGRequest):
    try:
        return rag.build_qa(req.max_rows)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/rag/build-pubmedqa")
def build_pubmedqa_alias(req: BuildRAGRequest):
    try:
        return rag.build_qa(req.max_rows)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/rag/query")
def rag_query(req: QueryRequest):
    try:
        return rag.answer(req.query, req.top_k)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/sentiment/train")
def train_sentiment(req: TrainRequest):
    try:
        return sentiment.train(req.max_rows)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/sentiment/predict")
def predict_sentiment(req: SentimentRequest):
    try:
        return sentiment.predict(req.text)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/observability")
def observability():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM runs ORDER BY created_at DESC LIMIT 100", conn)
    conn.close()

    summary = {"runs": int(len(df))}

    if not df.empty:
        summary.update({
            "avg_latency_ms": float(df["latency_ms"].mean()),
            "workflow_counts": df["workflow"].value_counts().to_dict()
        })

    return {
        "summary": summary,
        "recent_runs": df.to_dict(orient="records")
    }


@app.post("/generate-artifacts")
def artifacts():
    return generate_artifacts()


@app.get("/requirements-map")
def requirements_map():
    return {
        "fixed_issues": [
            "PubMedQA dataset-script error handled",
            "SQuAD HF URI error handled",
            "Financial PhraseBank dataset-script error avoided",
            "Script-free financial dataset attempts added",
            "Local fallback added",
            "Top-level await removed",
            "Jupyter-safe server helper added"
        ],
        "datasets": [
            "PubMedQA if compatible",
            "rajpurkar/squad fallback",
            "local QA fallback",
            "zeroshot/twitter-financial-news-sentiment if compatible",
            "nickmuchi/financial-classification if compatible",
            "local financial fallback"
        ],
        "rag": [
            "SentenceTransformer",
            "FAISS",
            "CrossEncoder",
            "TF-IDF fallback"
        ],
        "deployment": [
            "FastAPI",
            "Docker",
            "Render"
        ]
    }


import nest_asyncio
import uvicorn

nest_asyncio.apply()

config = uvicorn.Config(
    app=app,
    host="127.0.0.1",
    port=8390,
    reload=False,
    log_level="info"
)

