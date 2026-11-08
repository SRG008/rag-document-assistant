import os
import io
import time
import json
import math
import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pypdf import PdfReader
from docx import Document as DocxDocument
from dotenv import load_dotenv

load_dotenv()

# Enterprise RBAC Clearance Hierarchy (Higher number = higher clearance)
CLEARANCE_LEVELS = {
    "public": 1,
    "internal": 2,
    "confidential": 3,
    "restricted": 4,
}


class HybridFastRAGEngine:
    """
    FDE Enterprise Production RAG Engine v2.0:
    1. Pre-Retrieval RBAC & Metadata Filtering (Department + Clearance Level)
    2. True Hybrid Search: Dense Cosine Vectors + Sparse BM25 merged via Reciprocal Rank Fusion (RRF)
    3. Fast Precision Re-Ranker: Scores candidate chunks on exact phrase density, proximity, & query coverage
    4. Query Embedding Cache: Repeat questions skip embedding API latency entirely (< 0.5 ms)
    """

    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.meta_path = os.path.join(self.storage_dir, "chunks_meta.json")
        self.vec_path = os.path.join(self.storage_dir, "embeddings.npy")

        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._query_vec_cache: Dict[str, np.ndarray] = {}
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.meta_path) and os.path.exists(self.vec_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                self.embeddings = np.load(self.vec_path)
                # Ensure legacy v1 chunks get default RBAC metadata fields seamlessly
                for c in self.chunks:
                    c.setdefault("department", "general")
                    c.setdefault("clearance", "public")
            except Exception as e:
                print(f"[WARN] Could not load existing index, starting fresh: {e}")
                self.chunks = []
                self.embeddings = np.empty((0, 0), dtype=np.float32)

    def _save_index(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        if self.embeddings.size > 0:
            np.save(self.vec_path, self.embeddings)
        elif os.path.exists(self.vec_path):
            os.remove(self.vec_path)

    def clear_index(self):
        self.chunks = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self._query_vec_cache.clear()
        self._save_index()

    # ------------------------------------------------------------------
    # STEP 1: DOCUMENT PARSING (PDF, DOCX, TXT, MD)
    # ------------------------------------------------------------------
    def parse_file(self, filename: str, file_bytes: bytes) -> List[Dict[str, Any]]:
        ext = os.path.splitext(filename)[1].lower()
        pages = []

        if ext == ".pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                txt = re.sub(r"\s+", " ", txt).strip()
                if txt:
                    pages.append({"page": idx + 1, "text": txt})

        elif ext == ".docx":
            doc = DocxDocument(io.BytesIO(file_bytes))
            full_text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
            if full_text:
                pages.append({"page": 1, "text": full_text})

        elif ext in [".txt", ".md", ".csv", ".log"]:
            text = file_bytes.decode("utf-8", errors="ignore").strip()
            if text:
                pages.append({"page": 1, "text": text})
        else:
            raise ValueError(f"Unsupported file format: {ext}. Please upload PDF, DOCX, TXT, or MD.")

        return pages

    # ------------------------------------------------------------------
    # STEP 2: SENTENCE-AWARE CHUNKING WITH RBAC & METADATA TAGGING
    # ------------------------------------------------------------------
    def chunk_pages(
        self,
        filename: str,
        pages: List[Dict[str, Any]],
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        department: str = "general",
        clearance: str = "internal",
    ) -> List[Dict[str, Any]]:
        new_chunks = []
        department = department.lower().strip() or "general"
        clearance = clearance.lower().strip() or "internal"

        for page_obj in pages:
            page_num = page_obj["page"]
            text = page_obj["text"]

            sentences = re.split(r"(?<=[.!?])\s+", text)
            current_chunk = ""
            chunk_idx = 0

            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                    current_chunk = (current_chunk + " " + sentence).strip()
                else:
                    if current_chunk:
                        chunk_idx += 1
                        new_chunks.append(
                            self._build_chunk_record(
                                filename, page_num, chunk_idx, current_chunk, department, clearance
                            )
                        )
                    overlap_tail = (
                        current_chunk[-chunk_overlap:]
                        if len(current_chunk) > chunk_overlap
                        else current_chunk
                    )
                    current_chunk = (overlap_tail + " " + sentence).strip()

                    while len(current_chunk) > chunk_size * 1.5:
                        slice_part = current_chunk[:chunk_size]
                        chunk_idx += 1
                        new_chunks.append(
                            self._build_chunk_record(
                                filename, page_num, chunk_idx, slice_part, department, clearance
                            )
                        )
                        current_chunk = current_chunk[chunk_size - chunk_overlap :]

            if current_chunk:
                chunk_idx += 1
                new_chunks.append(
                    self._build_chunk_record(
                        filename, page_num, chunk_idx, current_chunk, department, clearance
                    )
                )

        return new_chunks

    def _build_chunk_record(
        self,
        filename: str,
        page: int,
        chunk_idx: int,
        text: str,
        department: str,
        clearance: str,
    ) -> Dict[str, Any]:
        chunk_hash = hashlib.md5(
            f"{filename}:{page}:{chunk_idx}:{department}:{clearance}:{text}".encode("utf-8")
        ).hexdigest()[:10]
        return {
            "id": chunk_hash,
            "source": filename,
            "page": page,
            "chunk_index": chunk_idx,
            "department": department,
            "clearance": clearance,
            "text": text,
            "tokens_est": max(1, len(text) // 4),
        }

    # ------------------------------------------------------------------
    # STEP 3: BATCH EMBEDDINGS WITH QUERY CACHE (GEMINI OR OPENAI)
    # ------------------------------------------------------------------
    def _embed_texts(self, texts: List[str], provider: str, api_key: str) -> np.ndarray:
        provider = provider.lower().strip()

        # Fast path: single-query cache hit
        if len(texts) == 1:
            cache_key = f"{provider}:{texts[0].strip().lower()}"
            if cache_key in self._query_vec_cache:
                return self._query_vec_cache[cache_key]

        if provider == "gemini":
            from google import genai
            client = genai.Client(api_key=api_key)
            embed_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
            response = client.models.embed_content(
                model=embed_model,
                contents=texts,
            )
            vectors = [e.values for e in response.embeddings]
            mat = np.array(vectors, dtype=np.float32)

        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            embed_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            response = client.embeddings.create(
                model=embed_model,
                input=texts,
            )
            vectors = [item.embedding for item in response.data]
            mat = np.array(vectors, dtype=np.float32)

        else:
            raise ValueError(f"Unknown AI_PROVIDER '{provider}'. Choose 'gemini' or 'openai'.")

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = mat / norms

        if len(texts) == 1:
            cache_key = f"{provider}:{texts[0].strip().lower()}"
            self._query_vec_cache[cache_key] = normed

        return normed

    def ingest_document(
        self,
        filename: str,
        file_bytes: bytes,
        provider: str,
        api_key: str,
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        department: str = "general",
        clearance: str = "internal",
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        pages = self.parse_file(filename, file_bytes)
        if not pages:
            raise ValueError("No readable text found in the uploaded document.")

        new_chunks = self.chunk_pages(
            filename, pages, chunk_size, chunk_overlap, department, clearance
        )
        texts = [c["text"] for c in new_chunks]

        batch_vecs = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch_vecs.append(self._embed_texts(texts[i : i + batch_size], provider, api_key))

        new_vecs = np.vstack(batch_vecs)

        if self.embeddings.size > 0 and self.embeddings.shape[1] != new_vecs.shape[1]:
            self.chunks = []
            self.embeddings = np.empty((0, 0), dtype=np.float32)

        self.chunks.extend(new_chunks)
        if self.embeddings.size == 0:
            self.embeddings = new_vecs
        else:
            self.embeddings = np.vstack([self.embeddings, new_vecs])

        self._save_index()
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "filename": filename,
            "department": department,
            "clearance": clearance,
            "pages_parsed": len(pages),
            "chunks_added": len(new_chunks),
            "total_chunks_in_index": len(self.chunks),
            "ingest_time_ms": elapsed_ms,
        }

    # ------------------------------------------------------------------
    # STEP 4: ENTERPRISE RBAC PRE-FILTER + BM25 + RRF + PRECISION RERANKER
    # ------------------------------------------------------------------
    def _compute_bm25_scores(self, query_terms: List[str], candidate_indices: List[int]) -> np.ndarray:
        """True Okapi BM25 Sparse Lexical Scoring over RBAC-authorized chunks."""
        if not query_terms or not candidate_indices:
            return np.zeros(len(candidate_indices), dtype=np.float32)

        tokenized_docs = []
        doc_lengths = []
        df: Dict[str, int] = {t: 0 for t in query_terms}

        for idx in candidate_indices:
            words = re.findall(r"[a-z0-9_\-]+", self.chunks[idx]["text"].lower())
            tokenized_docs.append(words)
            doc_lengths.append(len(words))
            unique_words = set(words)
            for t in query_terms:
                if t in unique_words:
                    df[t] += 1

        N = len(candidate_indices)
        avgdl = max(1.0, sum(doc_lengths) / N)
        k1 = 1.5
        b = 0.75

        scores = np.zeros(N, dtype=np.float32)
        for i, words in enumerate(tokenized_docs):
            dl = doc_lengths[i]
            score = 0.0
            for t in query_terms:
                f = words.count(t)
                if f > 0:
                    idf = math.log(1.0 + (N - df[t] + 0.5) / (df[t] + 0.5))
                    denom = f + k1 * (1.0 - b + b * (dl / avgdl))
                    score += idf * ((f * (k1 + 1.0)) / denom)
            scores[i] = score

        return scores

    def _precision_rerank_score(
        self, query: str, query_terms: List[str], chunk_text: str, rrf_score: float, vec_score: float
    ) -> float:
        """
        Stage-2 Cross-Feature Reranker:
        Evaluates query-chunk interaction (exact phrase continuity, bigram overlap,
        term coverage ratio, and numeric/code match) to promote the most accurate evidence to #1.
        """
        c_lower = chunk_text.lower()
        q_lower = query.lower().strip()

        # 1. Term coverage ratio (what % of unique query terms appear in this chunk?)
        if query_terms:
            matched_terms = sum(1 for t in query_terms if t in c_lower)
            coverage = matched_terms / len(query_terms)
        else:
            coverage = 0.0

        # 2. Bigram / Phrase continuity bonus (rewards adjacent query words appearing together)
        q_words = re.findall(r"[a-z0-9_\-]+", q_lower)
        bigram_bonus = 0.0
        if len(q_words) >= 2:
            bigrams = [f"{q_words[i]} {q_words[i+1]}" for i in range(len(q_words) - 1)]
            bigram_hits = sum(1 for bg in bigrams if bg in c_lower)
            bigram_bonus = 0.15 * (bigram_hits / len(bigrams))

        # 3. Exact substring match bonus for numbers/serials/clauses (e.g. "4.2b", "1040")
        codes = re.findall(r"\b\w*\d+\w*\b", q_lower)
        code_bonus = 0.0
        if codes:
            code_hits = sum(1 for cd in codes if cd in c_lower)
            code_bonus = 0.20 * (code_hits / len(codes))

        # Combine semantic vector score (55%), scaled RRF score (20%), and cross-feature signals (25%)
        final_score = (0.55 * float(vec_score)) + (8.0 * float(rrf_score)) + (0.20 * coverage) + bigram_bonus + code_bonus
        return round(final_score, 4)

    def retrieve(
        self,
        query: str,
        provider: str,
        api_key: str,
        top_k: int = 4,
        user_department: str = "all",
        user_clearance: str = "restricted",
        source_filter: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.chunks or self.embeddings.size == 0:
            return [], {"retrieval_ms": 0.0, "rerank_ms": 0.0, "rbac_allowed": 0, "rbac_blocked": 0}

        t0 = time.perf_counter()

        # --------------------------------------------------------------
        # STAGE 0: PRE-RETRIEVAL RBAC & METADATA FILTERING
        # --------------------------------------------------------------
        user_dept = (user_department or "all").lower().strip()
        user_level = CLEARANCE_LEVELS.get((user_clearance or "restricted").lower().strip(), 4)

        allowed_indices: List[int] = []
        for idx, c in enumerate(self.chunks):
            c_dept = c.get("department", "general").lower()
            c_level = CLEARANCE_LEVELS.get(c.get("clearance", "public").lower(), 1)

            # Clearance check: user's clearance level must be >= chunk's required level
            if c_level > user_level:
                continue

            # Department check: 'all' sees all departments; otherwise user sees their dept + 'general'
            if user_dept != "all" and c_dept not in (user_dept, "general"):
                continue

            # Optional filename filter
            if source_filter and source_filter != "all" and c.get("source") != source_filter:
                continue

            allowed_indices.append(idx)

        rbac_blocked = len(self.chunks) - len(allowed_indices)
        if not allowed_indices:
            return [], {
                "retrieval_ms": round((time.perf_counter() - t0) * 1000, 2),
                "rerank_ms": 0.0,
                "rbac_allowed": 0,
                "rbac_blocked": rbac_blocked,
            }

        # --------------------------------------------------------------
        # STAGE 1: HYBRID RETRIEVAL (DENSE VECTOR + SPARSE BM25 + RRF)
        # --------------------------------------------------------------
        q_vec = self._embed_texts([query], provider, api_key)[0]

        # Slice only the RBAC-authorized rows from the NumPy matrix
        sub_embeddings = self.embeddings[allowed_indices]
        semantic_scores = np.dot(sub_embeddings, q_vec)

        # Compute BM25 lexical scores across the authorized chunks
        stopwords = {"what", "where", "when", "which", "who", "how", "the", "and", "for", "are", "with", "this", "that", "from"}
        q_terms = [t for t in re.findall(r"[a-z0-9_\-]{2,}", query.lower()) if t not in stopwords]
        bm25_scores = self._compute_bm25_scores(q_terms, allowed_indices)

        # Reciprocal Rank Fusion (RRF): combines Dense Rank + BM25 Rank without scale bias
        dense_order = np.argsort(semantic_scores)[::-1]
        bm25_order = np.argsort(bm25_scores)[::-1]

        dense_ranks = np.empty(len(allowed_indices), dtype=np.int32)
        bm25_ranks = np.empty(len(allowed_indices), dtype=np.int32)
        for rank_pos, local_idx in enumerate(dense_order):
            dense_ranks[local_idx] = rank_pos + 1
        for rank_pos, local_idx in enumerate(bm25_order):
            bm25_ranks[local_idx] = rank_pos + 1

        rrf_k = 60.0
        rrf_scores = (1.0 / (rrf_k + dense_ranks)) + (1.0 / (rrf_k + bm25_ranks))

        # Select Top-(3 * top_k) candidates for Stage-2 Reranking
        candidate_pool_size = min(len(allowed_indices), max(top_k * 3, 12))
        top_local_candidates = np.argsort(rrf_scores)[::-1][:candidate_pool_size]
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)

        # --------------------------------------------------------------
        # STAGE 2: CROSS-FEATURE PRECISION RERANKING
        # --------------------------------------------------------------
        t_rerank = time.perf_counter()
        reranked_Pool = []
        for local_idx in top_local_candidates:
            global_idx = allowed_indices[int(local_idx)]
            chunk_obj = dict(self.chunks[global_idx])
            v_score = round(float(semantic_scores[int(local_idx)]), 4)
            b_score = round(float(bm25_scores[int(local_idx)]), 4)
            r_score = round(float(rrf_scores[int(local_idx)]), 5)

            rerank_score = self._precision_rerank_score(
                query, q_terms, chunk_obj["text"], r_score, v_score
            )
            chunk_obj["semantic_score"] = v_score
            chunk_obj["bm25_score"] = b_score
            chunk_obj["rrf_score"] = r_score
            chunk_obj["score"] = rerank_score
            reranked_Pool.append(chunk_obj)

        # Sort candidates by the Stage-2 Reranker score and take Top-K
        reranked_Pool.sort(key=lambda x: x["score"], reverse=True)
        final_results = reranked_Pool[:top_k]
        for r_idx, item in enumerate(final_results):
            item["rank"] = r_idx + 1

        rerank_ms = round((time.perf_counter() - t_rerank) * 1000, 2)

        return final_results, {
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "rbac_allowed": len(allowed_indices),
            "rbac_blocked": rbac_blocked,
        }

    # ------------------------------------------------------------------
    # STEP 5: GROUNDED ANSWER GENERATION WITH INLINE CITATIONS
    # ------------------------------------------------------------------
    def answer_question(
        self,
        query: str,
        provider: str,
        api_key: str,
        top_k: int = 4,
        user_department: str = "all",
        user_clearance: str = "restricted",
        source_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        retrieved_chunks, search_meta = self.retrieve(
            query=query,
            provider=provider,
            api_key=api_key,
            top_k=top_k,
            user_department=user_department,
            user_clearance=user_clearance,
            source_filter=source_filter,
        )

        if not self.chunks:
            return {
                "answer": "No documents are indexed yet. Please upload a PDF, DOCX, or TXT file first.",
                "citations": [],
                "metrics": {**search_meta, "llm_ms": 0, "total_ms": 0, "chunks_searched": 0},
            }

        if not retrieved_chunks:
            return {
                "answer": (
                    f"🔒 **RBAC Access Control Notice:** 0 chunks matched your current role permissions "
                    f"(Department: `{user_department.upper()}`, Clearance: `{user_clearance.upper()}`). "
                    f"`{search_meta['rbac_blocked']}` restricted chunks were filtered out before reaching the LLM."
                ),
                "citations": [],
                "metrics": {**search_meta, "llm_ms": 0, "total_ms": 0, "chunks_searched": 0},
            }

        context_blocks = []
        for c in retrieved_chunks:
            context_blocks.append(
                f"[Source {c['rank']}: {c['source']} | Page {c['page']} | Dept: {c['department'].upper()} | Clearance: {c['clearance'].upper()}]\n{c['text']}"
            )
        context_str = "\n\n---\n\n".join(context_blocks)

        system_prompt = (
            "You are a precision Forward Deployed Engineer (FDE) Enterprise RAG Assistant.\n"
            "Rules:\n"
            "1. Answer the user's question ONLY using the provided Context Sources below.\n"
            "2. Cite every fact inline using its source tag, e.g. [Source 1] or [Source 2].\n"
            "3. If the answer is not contained in the context, state clearly: "
            "'I cannot verify this from the authorized documents.' Never guess or hallucinate.\n"
            "4. Keep your response concise, structured with bullet points where helpful, and direct."
        )

        user_prompt = f"Context Sources:\n{context_str}\n\nUser Question: {query}"

        t_llm = time.perf_counter()
        provider = provider.lower().strip()

        if provider == "gemini":
            from google import genai
            client = genai.Client(api_key=api_key)
            chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
            resp = client.models.generate_content(
                model=chat_model,
                contents=f"{system_prompt}\n\n{user_prompt}",
            )
            answer_text = resp.text or "No response generated."

        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            answer_text = resp.choices[0].message.content or "No response generated."
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        llm_ms = round((time.perf_counter() - t_llm) * 1000, 1)
        total_ms = round((time.perf_counter() - t_start) * 1000, 1)

        return {
            "answer": answer_text,
            "citations": retrieved_chunks,
            "metrics": {
                "retrieval_ms": search_meta["retrieval_ms"],
                "rerank_ms": search_meta["rerank_ms"],
                "llm_ms": llm_ms,
                "total_ms": total_ms,
                "chunks_searched": search_meta["rbac_allowed"],
                "rbac_blocked": search_meta["rbac_blocked"],
            },
        }
