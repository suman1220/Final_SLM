# rag_engine.py
from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# -------- Optional deps (fail with friendly errors) --------
try:
    import numpy as np
except Exception as e:
    np = None
    _NP_ERR = e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None
    _ST_ERR = e

try:
    import fitz  # PyMuPDF
except Exception as e:
    fitz = None
    _FITZ_ERR = e

try:
    import faiss  # faiss-cpu
except Exception:
    faiss = None  # optional

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None  # optional

# -------- Tokenization helpers --------
_STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","by","is","are","am","was","were","be","being","been",
    "this","that","these","those","it","its","as","at","from","about","into","your","you","me","we","our","us","their"
}
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

def _toks(s):
    return [w for w in _TOKEN_RE.findall(s.lower()) if w not in _STOPWORDS and len(w) > 2]


def _normalize_spaces(s):
    return re.sub(r"[ \t]+", " ", s.strip())


def _fingerprint(s):
    norm = _normalize_spaces(s).lower()
    return hashlib.blake2s(norm.encode("utf-8"), digest_size=12).hexdigest()


def _split_sentences(text):
    parts = re.split(r"(?<=[\.\!\?।।؟])\s+", text)
    return [_normalize_spaces(p) for p in parts if p and _normalize_spaces(p)]


def _looks_like_heading(line):
    ln = line.strip()
    if not ln:
        return False
    if re.match(r"^\s*(\d+(\.\d+)*[\.\)]\s+)", ln):
        return True
    if len(ln) <= 80 and (ln.endswith(":") or ln.isupper() or len(ln.split()) <= 10):
        return not re.match(r"^(table|figure|page|section)$", ln.strip().lower())
    return False


def _is_bullet(line):
    return bool(re.match(r"^\s*(?:[-*•·◦‣]|(\d+[\.\)]))\s+", line))


def _is_table_block(lines):
    pipey = sum(1 for l in lines if l.count("|") >= 2)
    multisp = sum(1 for l in lines if re.search(r"\S(\s{2,})\S", l))
    return (pipey >= 2) or (multisp >= 3)


@dataclass
class RAGConfig:
    docs_dir: str = "./data/documents"   # ← ye line add karo
    allowed_ext: tuple = ("pdf",)
    chunk_size: int = 300
    chunk_overlap: int = 60
    sents_per_chunk: int = 1
    context_budget: int = 150
    k: int = 4
    bm25_fallback: bool = True
    rag_lex_min: int = 2
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class _NumpyFlatIP:
    def __init__(self, dim):
        self.dim = dim
        self.x = np.empty((0, dim), dtype=np.float32)
    @property
    def ntotal(self):
        return int(self.x.shape[0])
    def add(self, vecs):
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError("Bad vector shape")
        self.x = np.vstack([self.x, vecs])
    def search(self, q, k):
        if self.ntotal == 0:
            return np.zeros((q.shape[0], k), dtype=np.float32), np.full((q.shape[0], k), -1, dtype=np.int64)
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        xn = self.x / (np.linalg.norm(self.x, axis=1, keepdims=True) + 1e-8)
        sims = qn @ xn.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        D = np.take_along_axis(sims, idx, axis=1)
        return D.astype(np.float32), idx.astype(np.int64)


class RAGEngine:

    def __init__(self, config: RAGConfig, token_len_fn=None):
        self.cfg = config
        self.doc_store= []
        self.bm25 = None
        self.emb_model= None
        self.emb_dim= None
        self.index = None
        self.faiss_metric = "ip"
        # token_len callable comes from main (llama tokenizer); fallback ~0.26 chars
        self.token_len = token_len_fn or (lambda s: max(1, int(len(s) * 0.26)))

    # ----- Setup / dependencies -----
    def _require(self):
        if np is None:
            raise RuntimeError(f"NumPy not available: {_NP_ERR}")
        if SentenceTransformer is None:
            raise RuntimeError(f"sentence-transformers not available: {_ST_ERR}")
        if fitz is None:
            raise RuntimeError(f"PyMuPDF not available: {_FITZ_ERR}")

    def _ensure_embedding_stack(self):
        if self.emb_model is None:
            self.emb_model = SentenceTransformer(self.cfg.model_name, device="cpu")
            self.emb_dim = self.emb_model.get_sentence_embedding_dimension()
        if self.index is None:
            if faiss is not None:
                try:
                    self.index = faiss.IndexFlatIP(self.emb_dim)
                    self.faiss_metric = "ip"
                except Exception:
                    self.index = _NumpyFlatIP(self.emb_dim)
                    self.faiss_metric = "ip"
            else:
                self.index = _NumpyFlatIP(self.emb_dim)
                self.faiss_metric = "ip"

    # ----- Embeddings -----
    def _embed_texts(self, texts):
        self._ensure_embedding_stack()
        vecs = self.emb_model.encode(
            texts, batch_size=16, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True, device="cpu"
        )
        return np.asarray(vecs, dtype=np.float32)

    # ----- PDF ingestion -----
    def _extract_pdf(self, path: Path):
        out = []
        with fitz.open(str(path)) as doc:
            for i in range(len(doc)):
                page = doc.load_page(i)
                text = (page.get_text("text") or "").strip()
                if text: out.append((i + 1, text))
        return out

    def _sectionize(self, text):
        lines = [l.rstrip() for l in text.splitlines()]
        blocks, buf, mode = [], [], "paragraph"
        def flush():
            nonlocal buf, mode
            if not buf: return
            content = "\n".join(buf).strip()
            if content:
                blocks.append({"type": mode, "text": content})
            buf, mode = [], "paragraph"
        i = 0
        while i < len(lines):
            ln = lines[i]
            if _looks_like_heading(ln):
                flush(); blocks.append({"type": "heading", "text": _normalize_spaces(ln)}); i += 1; continue
            if _is_bullet(ln):
                if mode != "list": flush(); mode = "list"
                buf.append(ln); i += 1
                while i < len(lines) and (_is_bullet(lines[i]) or not lines[i].strip()):
                    if lines[i].strip(): buf.append(lines[i])
                    i += 1
                continue
            look = lines[i:i+8]
            if _is_table_block(look):
                flush()
                tbuf = [lines[i]]; i += 1
                while i < len(lines) and (_is_table_block([lines[i]]) or lines[i].strip()):
                    tbuf.append(lines[i]); i += 1
                blocks.append({"type": "table", "text": "\n".join(tbuf)})
                continue
            if mode != "paragraph":
                flush(); mode = "paragraph"
            buf.append(ln); i += 1
        flush()
        return blocks

    def _chunk_block_text(self, text ,max_tokens, overlap_tokens):
        sents = _split_sentences(text)
        chunks, cur, cur_tok = [], [], 0
        def add_chunk(slice_sents: List[str]):
            if slice_sents:
                ch = " ".join(slice_sents).strip()
                if ch: chunks.append(ch)
        i = 0
        while i < len(sents):
            s = sents[i]
            t = self.token_len(s)
            if cur_tok + t <= max_tokens:
                cur.append(s); cur_tok += t; i += 1; continue
            if not cur:
                seg = s
                if len(seg) > 300: seg = seg[:280]
                add_chunk([seg]); i += 1
            else:
                add_chunk(cur)
                back_toks, back = 0, []
                j = len(cur) - 1
                while j >= 0 and back_toks < overlap_tokens:
                    back.insert(0, cur[j])
                    bt = self.token_len(cur[j])
                    back_toks += bt; j -= 1
                cur = back[:]
                cur_tok = sum(self.token_len(x) for x in cur)
        add_chunk(cur)
        return chunks

    def _smart_chunk_page(self, text, page_no, source_name,max_tokens, overlap_tokens):
        blocks = self._sectionize(text)
        chunks, seen = [], set()
        section_path: List[str] = []
        for b in blocks:
            btype = b["type"]; btext = _normalize_spaces(b["text"])
            if btype == "heading":
                section_path.append(btext); section_path = section_path[-2:]; continue
            if btype == "table":
                lines = [l for l in btext.splitlines() if l.strip()]
                if len(lines) <= 40:
                    ctext = "\n".join(lines)
                    fp = _fingerprint(ctext)
                    if fp in seen: continue
                    seen.add(fp)
                    chunks.append({"text": ctext, "meta": {"source": source_name, "page": page_no, "type": "table",
                                                           "section": " › ".join(section_path) if section_path else None}})
                else:
                    w = 30
                    for k in range(0, len(lines), w - 5):
                        part = "\n".join(lines[k:k+w])
                        fp = _fingerprint(part)
                        if fp in seen: continue
                        seen.add(fp)
                        chunks.append({"text": part, "meta": {"source": source_name, "page": page_no, "type": "table",
                                                              "section": " › ".join(section_path) if section_path else None,
                                                              "row_window": f"{k+1}-{min(len(lines), k+w)}"}})
                continue
            max_tok = max_tokens + (40 if btype == "list" else 0)
            parts = self._chunk_block_text(btext, max_tokens=max_tok, overlap_tokens=self.cfg.chunk_overlap)
            for part in parts:
                if len(part) < 10: continue
                fp = _fingerprint(part)
                if fp in seen: continue
                seen.add(fp)
                chunks.append({"text": part, "meta": {"source": source_name, "page": page_no, "type": btype,
                                                      "section": " › ".join(section_path) if section_path else None}})
        return chunks

    def _process_pdf_to_chunks(self, file_path, source_name):
        pages = self._extract_pdf(file_path)
        out = []
        for page_no, text in pages:
            out.extend(self._smart_chunk_page(text, page_no, source_name, self.cfg.chunk_size, self.cfg.chunk_overlap))
        return out

    # ----- Indexing -----
    def _allowed_file(self, name):
        return "." in name and name.rsplit(".", 1)[1].lower() in self.cfg.allowed_ext

    def add_chunks_to_index(self, chunks: List[Dict[str, Any]]):
        if not chunks: return
        texts = [c["text"] for c in chunks]
        vecs = self._embed_texts(texts)
        self._ensure_embedding_stack()
        base = len(self.doc_store)
        for i, c in enumerate(chunks):
            c.setdefault("meta", {})
            c["meta"]["gid"] = base + i
        self.index.add(vecs)
        self.doc_store.extend(chunks)
        if BM25Okapi is not None and self.cfg.bm25_fallback:
            try:
                tokens = [d["text"].lower().split() for d in self.doc_store]
                self.bm25 = BM25Okapi(tokens)
            except Exception:
                self.bm25 = None

    def index_pdf_file(self, pdf_path):
        self._require()
        p = Path(pdf_path)
        if not p.exists():
            raise FileNotFoundError(p)
        if not self._allowed_file(p.name):
            return 0
        chunks = self._process_pdf_to_chunks(p, p.name)
        self.add_chunks_to_index(chunks)
        return len(chunks)

    def index_pdfs_in_dir(self, dir_path):
        self._require()
        dp = Path(dir_path or self.cfg.docs_dir)
        if not dp.exists():
            raise FileNotFoundError(f"RAG docs directory not found: {dp}")
        count = 0
        for f in dp.iterdir():
            if f.is_file() and self._allowed_file(f.name):
                chunks = self._process_pdf_to_chunks(f, f.name)
                self.add_chunks_to_index(chunks)
                count += 1
        return count

    # ----- Retrieval -----
    def _lexical_overlap_count(self, q, txt):
        qset = set(_toks(q))
        if not qset: return 0
        tset = set(_toks(txt))
        return len(qset & tset)

    def search_with_score(self, query,k):
        k = int(k or self.cfg.k)
        hits_idx= []
        top_sim = 0.0
        hit_count = 0
        if self.index is not None and len(self.doc_store) > 0 and getattr(self.index, "ntotal", 0) > 0:
            qv = self._embed_texts([query])
            kk = max(k, 2)
            D, I = self.index.search(qv, kk)
            idxs = [int(i) for i in I[0] if 0 <= int(i) < len(self.doc_store)]
            hit_count = len(idxs)
            if hit_count > 0:
                d0 = float(D[0][0])
                if self.faiss_metric == "ip":
                    top_sim = max(0.0, min(1.0, d0))
                else:
                    top_sim = max(0.0, min(1.0, 1.0 - d0/2.0))
            hits_idx.extend(idxs)
        if BM25Okapi is not None and self.cfg.bm25_fallback and self.bm25 is not None:
            scores = self.bm25.get_scores(query.lower().split())
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            hits_idx.extend(top_idx)
        seen, order = set(), []
        for i in hits_idx:
            if i not in seen:
                order.append(i); seen.add(i)
        order = order[:k]
        return [self.doc_store[i] for i in order], float(top_sim), int(hit_count)

    def distill_sentences(self, query: str, text: str, keep: Optional[int] = None) -> str:
        keep = keep or self.cfg.sents_per_chunk
        sents = _split_sentences(text)[:60]
        if not sents: return text[:400]
        if BM25Okapi is not None:
            try:
                tokenized = [s.lower().split() for s in sents]
                bm_local = BM25Okapi(tokenized)
                scores = bm_local.get_scores(query.lower().split())
                top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:keep]
                return "\n".join(sents[i] for i in top_idx)
            except Exception:
                pass
        q = set(_toks(query))
        scored = sorted(sents, key=lambda s: len(q & set(_toks(s))), reverse=True)
        return "\n".join(scored[:keep])

    def assemble_context(self, query, hits):
        ctx_parts, used, chosen = [], 0, []
        for h in hits:
            meta = h.get("meta", {})
            distilled = self.distill_sentences(query, h["text"], keep=self.cfg.sents_per_chunk)
            piece = f"[{meta.get('source','?')} p.{meta.get('page','?')} | {meta.get('section') or ''}]\n{distilled}"
            tl = self.token_len(piece)
            if used + tl > self.cfg.context_budget:
                continue
            ctx_parts.append(piece); chosen.append(meta); used += tl
            if used >= self.cfg.context_budget:
                break
        return "\n\n".join(ctx_parts), chosen

    # ----- Simple decision helper -----
    def should_use_rag(self, question ,hits):
        if not hits:
            return False
        lex = self._lexical_overlap_count(question, hits[0]["text"])
        return lex >= self.cfg.rag_lex_min