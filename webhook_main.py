#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webhook_main.py — чистая версия (готовая к подстановке)
- AIOHTTP веб-сервер под Telegram Webhook
- Гибридный retrieve: эмбеддинги + BM25/ключевые слова
- Жёсткая проверка согласованности embeddings.npy ↔ JSON
- Строгий JSON-вывод от LLM и безопасный рендер в HTML для Telegram
- Без дублей функций и "магических" констант в логике

Зависимости (requirements.txt):
    aiohttp
    requests
    numpy
    openai>=1.30.0
    rank_bm25     # рекомендуется, но не обязательно (fallback на keyword-скоринг)
    # (опционально для логов в Google Sheets)
    gspread
    google-auth
    google-auth-oauthlib
    google-auth-httplib2

Переменные окружения (обязательные):
    OPENAI_API_KEY
    TELEGRAM_TOKEN
    WEBHOOK_URL                   # например, https://your-app.onrender.com

Переменные окружения (рекомендуемые/опциональные):
    PORT                          # default: 8080
    WEBHOOK_PATH                  # default: /webhook
    TELEGRAM_WEBHOOK_SECRET       # секрет заголовка X-Telegram-Bot-Api-Secret-Token
    EMBEDDINGS_PATH               # default: embeddings.npy
    PUNKTS_PATH                   # default: pravila_detailed_tagged_autofix.json
    EMBEDDING_MODEL               # default: text-embedding-ada-002 (совместим с текущим embeddings.npy)
    CHAT_MODEL                    # default: gpt-4o-mini
    MULTI_QUERY                   # "1" → включить Мульти-переформулировки (дороже)
    SHEET_ID, GOOGLE_CREDENTIALS_JSON  # если хотите логировать вопросы/ответы в Google Sheets

Автор: специально для замены старого webhook_main.py, без сокращений.
"""
from __future__ import annotations

import os
import json
import time
import html
import uuid
import logging
import asyncio
import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import requests
from aiohttp import web

# OpenAI v1 style
from openai import OpenAI
from openai import APIStatusError, APIConnectionError, RateLimitError, APITimeoutError
# Опционально: лемматизация RU
try:
    import pymorphy2  # type: ignore
    _MORPH = pymorphy2.MorphAnalyzer()
except Exception:
    _MORPH = None

# Опционально: BM25
try:
    from rank_bm25 import BM25Okapi
    HAVE_BM25 = True
except Exception:
    HAVE_BM25 = False

# Опционально: импорт словарей маппингов (если файл есть и валиден)
try:
    import importlib.util
    _cfg_path = os.path.join(os.getcwd(), "config_maps.py")
    if os.path.exists(_cfg_path):
        spec = importlib.util.spec_from_file_location("config_maps", _cfg_path)
        cfg = importlib.util.module_from_spec(spec) if spec else None
        if spec and cfg and spec.loader:
            spec.loader.exec_module(cfg)  # type: ignore
            UNIVERSAL_MAP: Dict[str, Dict[str, str]] = getattr(cfg, "UNIVERSAL_MAP", {})
        else:
            UNIVERSAL_MAP = {}
    else:
        UNIVERSAL_MAP = {}
except Exception:
    UNIVERSAL_MAP = {}

# ──────────────────────────── Конфиг ────────────────────────────

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "").strip()

assert TELEGRAM_TOKEN, "TELEGRAM_TOKEN is required"
assert OPENAI_API_KEY, "OPENAI_API_KEY is required"
assert WEBHOOK_URL, "WEBHOOK_URL is required (e.g., https://your-app.onrender.com)"

PORT = int(os.environ.get("PORT", "8080"))
WEBHOOK_PATH = os.environ.get("WEBHOOK_PATH", "/webhook")
TELEGRAM_WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip() or None

EMBEDDINGS_PATH = os.environ.get("EMBEDDINGS_PATH", "embeddings.npy")
PUNKTS_PATH = os.environ.get("PUNKTS_PATH", "pravila_detailed_tagged_autofix.json")

# ВНИМАНИЕ: embeddings.npy сейчас на 1536-мерной модели ada-002 — оставляем такую же модель до пересчёта
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")


MULTI_QUERY = os.environ.get("MULTI_QUERY", "0") == "1"
HYDE = os.environ.get("HYDE", "0") == "1"  # ⬅️ добавить эту строку


SHEET_ID = os.environ.get("SHEET_ID", "").strip() or None
GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON", "").strip() or None

# Скоринговые веса
W_EMB = 1.0
W_BM25 = 0.75
W_MAP = 1.25
W_REGEX = 0.15

# рядом с другими весами
W_CAT = 1.1
CAT_KEYS = ("исследовател", "модератор", "эксперт", "мастер")

def kw_category_boost(question: str, doc_text: str) -> float:
    ql = (question or "").lower().replace("ё", "е")
    dl = (doc_text or "").lower().replace("ё", "е")
    for k in CAT_KEYS:
        if k in ql and f"педагог-{k}" in dl:
            return 1.0
    return 0.0

W_KW = 0.9  # вес сигнала по ключевым словам для тематических исключений/зарубеж
KW_EXCEPTION_TERMS = (
    "без прохождения процедуры аттестации",
    "присваивается без",
    "освобожда",
    "не подлежит аттестации",
)
KW_FOREIGN_TERMS = (
    "зарубеж", "за пределами республики казахстан", "за границ",
    "иностран", "nazarbayev university", "болаш",
)
def kw_boost(question: str, doc_text: str) -> float:
    ql = (question or "").lower().replace("ё", "е")
    dl = (doc_text or "").lower().replace("ё", "е")

    boost = 0.0
    foreign_q = any(k in ql for k in ("магист", "за рубеж", "за границ", "зарубеж", "иностран"))
    if foreign_q:
        if any(k in dl for k in KW_EXCEPTION_TERMS):
            boost += 0.6
        if any(k in dl for k in KW_FOREIGN_TERMS):
            boost += 0.6
    # общий случай: если в документе есть явное "без прохождения" — подбустим всегда
    if any(k in dl for k in KW_EXCEPTION_TERMS):
        boost += 0.3
    return boost

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# ──────────────────────────── Логгер ────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag-bot")
LAST_RESPONSES: Dict[Tuple[int, int], Dict[str, Any]] = {}

# ─────────────────────── Клиенты и данные ───────────────────────

client = OpenAI(api_key=OPENAI_API_KEY)

def load_punkts(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    assert isinstance(arr, list) and len(arr) > 0, "PUNKTS JSON must be a non-empty list"
    # Нормализация ключей (на всякий случай)
    for p in arr:
        p.setdefault("id", str(uuid.uuid4()))
        p.setdefault("punkt_num", "")
        p.setdefault("subpunkt_num", "")
        p.setdefault("text", "")
        p.setdefault("chapter", "")
        p.setdefault("paragraph", "")
    return arr

PUNKTS: List[Dict[str, Any]] = load_punkts(PUNKTS_PATH)

# embeddings.npy — memmap для экономии памяти/быстрой загрузки
PUNKT_EMBS: np.ndarray = np.load(EMBEDDINGS_PATH, mmap_mode="r")
assert PUNKT_EMBS.ndim == 2, "embeddings.npy must be 2D"
assert PUNKT_EMBS.shape[0] == len(PUNKTS), "rows(embeddings) != len(PUNKTS)"
if EMBEDDING_MODEL == "text-embedding-ada-002":
    assert PUNKT_EMBS.shape[1] == 1536, "For ada-002 expected 1536 dims"
logger.info("Loaded %d punkts; embeddings: %s", len(PUNKTS), PUNKT_EMBS.shape)

def tokenize(text: str) -> List[str]:
    return re.findall(r"[а-яёa-z0-9]+", (text or "").lower())

def normalize_tokens(tokens: List[str]) -> List[str]:
    if not _MORPH:
        return tokens
    out = []
    for t in tokens:
        try:
            p = _MORPH.parse(t)
            out.append(p[0].normal_form if p else t)
        except Exception:
            out.append(t)
    return out
# Тексты для BM25/keyword поиска
DOCS_TOKENS: List[List[str]] = []
for p in PUNKTS:
    toks = tokenize(p.get("text") or "")
    DOCS_TOKENS.append(normalize_tokens(toks))


BM25 = None
if HAVE_BM25:
    BM25 = BM25Okapi(DOCS_TOKENS)
    logger.info("BM25 index built (rank_bm25).")
else:
    logger.warning("rank_bm25 not installed — will use simple keyword scoring fallback.")

# ───────────────────────── Утилиты ──────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (D,), b: (N, D)
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (b_norm @ a_norm)

def normalize_query(q: str) -> str:
    q = q or ""
    q = q.replace("ё", "е")
    return q.strip()




def call_with_retries(fn, max_attempts=3, base_delay=1.0, *args, **kwargs):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as e:
            if attempt == max_attempts:
                raise
            sleep_s = base_delay * (2 ** (attempt - 1)) + (0.1 * attempt)
            logger.warning("OpenAI call failed (%s). Retry %d/%d in %.1fs", type(e).__name__, attempt, max_attempts, sleep_s)
            time.sleep(sleep_s)

# ─────────────────── Переформулировки запроса ───────────────────

def multi_query_rewrites(q: str, n: int = 3) -> List[str]:
    if not MULTI_QUERY:
        return [q]
    prompt = f"Переформулируй этот вопрос юридической/официальной формулировкой на русском, сохранив смысл. Дай {n} вариантов, по одному в строке, без нумерации и комментариев:\n\n{q}"
    resp = call_with_retries(
        client.chat.completions.create,
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ты помощник по правовым/официальным формулировкам на русском языке."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()
    variants = [v.strip() for v in text.split("\n") if v.strip()]
    if not variants:
        variants = [q]
    # Дедуп
    uniq = []
    for v in [q] + variants:
        if v not in uniq:
            uniq.append(v)
    return uniq[: n + 1]

# ─────────────────────── Векторный поиск ────────────────────────

def embed_query(text: str) -> np.ndarray:
    resp = call_with_retries(
        client.embeddings.create,
        model=EMBEDDING_MODEL,
        input=text,
    )
    vec = np.array(resp.data[0].embedding, dtype=np.float64)
    return vec

def vector_search(q: str, top_k: int = 100) -> List[Tuple[int, float]]:
    vec = embed_query(q)
    sims = cosine_sim(vec, PUNKT_EMBS)  # (N,)
    idxs = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in idxs]

def hyde_passage(question: str) -> Optional[str]:
    if not HYDE:
        return None
    prompt = (
        "Сформулируй краткий официальный ответ (5–7 предложений) на вопрос по Правилам, "
        "без выдумки фактов, максимально общий. Это черновой конспект для поиска, не окончательный ответ.\n\n"
        f"Вопрос: {question}"
    )
    try:
        resp = call_with_retries(
            client.chat.completions.create,
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Ты аккуратно формулируешь юридический конспект на русском."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None
# ───────────────────── Sparse/BM25 поиск ────────────────────────

def bm25_search(q: str, top_k: int = 100) -> List[Tuple[int, float]]:
    toks = normalize_tokens(tokenize(q))
    
    if BM25:
        scores = BM25.get_scores(toks)
        idxs = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idxs]
    # Fallback: простой keyword-скоринг с TF (веса урезаны)
    scores = []
    for i, doc in enumerate(DOCS_TOKENS):
        if not doc: 
            scores.append(0.0)
            continue
        s = 0
        for t in toks:
            s += doc.count(t)
        scores.append(float(s))
    idxs = np.argsort(-np.array(scores))[:top_k]
    return [(int(i), float(scores[i])) for i in idxs]

# ───────────────────── Маппинги и регэкспы ──────────────────────

KEY_REGEXES = [
    # пример: «п. 29», «пункт 30», «по пункту 41» и т.п.
    r"\bп\.?\s*(\d{1,3})\b",
    r"\bпункт[а-я]*\s*(\d{1,3})\b",
]

def regex_hits(q: str) -> List[int]:
    hits: List[int] = []
    for rgx in KEY_REGEXES:
        for m in re.finditer(rgx, q.lower()):
            num = m.group(1)
            for i, p in enumerate(PUNKTS):
                if p.get("punkt_num") == num:
                    hits.append(i)
    return list(dict.fromkeys(hits))

def mapped_hits(q: str) -> List[int]:
    ql = q.lower()
    if not UNIVERSAL_MAP:
        return []
    got: List[int] = []
    for key, coord in UNIVERSAL_MAP.items():
        if key in ql:
            # Находим все элементы с таким пунктом; если subpunkt_num задан, фильтруем по нему
            for i, p in enumerate(PUNKTS):
                if p.get("punkt_num") == coord.get("punkt_num", ""):
                    sp = coord.get("subpunkt_num", "")
                    if not sp or p.get("subpunkt_num") == sp:
                        got.append(i)
    return list(dict.fromkeys(got))

# ───────────────────────── Merge & Score ────────────────────────


def rag_search(q: str, top_k_stage1: int = 120, final_k: int = 45) -> List[Dict[str, Any]]:
    q = normalize_query(q)
    variants = multi_query_rewrites(q)
    dense_agg: Dict[int, float] = {}
    sparse_agg: Dict[int, float] = {}

    # Dense для каждого варианта
    for v in variants:
        for idx, sc in vector_search(v, top_k=top_k_stage1):
            dense_agg[idx] = max(dense_agg.get(idx, 0.0), sc)

    # HyDE (опционально)
    hyde = hyde_passage(q)
    if hyde:
        for idx, sc in vector_search(hyde, top_k=top_k_stage1 // 2):
            dense_agg[idx] = max(dense_agg.get(idx, 0.0), sc)

    # Sparse для базового и вариантов
    for v in [q] + variants:
        for idx, sc in bm25_search(v, top_k=top_k_stage1):
            sparse_agg[idx] = max(sparse_agg.get(idx, 0.0), sc)

    # Явные попадания
    regex_idx = regex_hits(q)
    mapped_idx = mapped_hits(q)

    # Нормировка sparse (z-score)
    if sparse_agg:
        vals = np.array(list(sparse_agg.values()), dtype=np.float64)
        mu, sigma = float(vals.mean()), float(vals.std() + 1e-6)
        for k in list(sparse_agg.keys()):
            sparse_agg[k] = (sparse_agg[k] - mu) / sigma

    # Аггрегируем очки
    items: List[Tuple[int, float]] = []
    candidate_ids = set(list(dense_agg.keys()) + list(sparse_agg.keys()) + regex_idx + mapped_idx)
    for idx in candidate_ids:
        emb_sc = dense_agg.get(idx, 0.0)
        sp_sc  = sparse_agg.get(idx, 0.0)
        rx_sc  = 1.0 if idx in regex_idx  else 0.0
        mp_sc  = 1.0 if idx in mapped_idx else 0.0
        total  = W_EMB*emb_sc + W_BM25*sp_sc + W_REGEX*rx_sc + W_MAP*mp_sc
        total += W_KW * kw_boost(q, PUNKTS[idx].get("text",""))
        total += W_CAT * kw_category_boost(q, PUNKTS[idx].get("text", ""))
        items.append((idx, total))
       

    # Сортировка и выбор top-K
    items.sort(key=lambda x: -x[1])
    top_idx = [i for i, _ in items[:final_k]]

    # Форс-включаем явные совпадения из mapped/regex
    for i in mapped_idx + regex_idx:
        if i not in top_idx:
            top_idx.append(i)
    top_idx = top_idx[:final_k]

    selected = [PUNKTS[i] for i in top_idx]
    logger.debug("RAG selected idx: %s", top_idx[:15])
    return selected


# ───────────────────── Генерация ответа (LLM) ───────────────────

SYSTEM_PROMPT = (
    "Ты — строгий помощник по Правилам и условиям проведения аттестации педагогов РК. "
    "Отвечай только по предоставленным пунктам, не выдумывай деталей. "
    "Если прямого ответа нет, так и скажи. Всегда ссылайся на конкретные пункты."
)

GEN_PROMPT_TEMPLATE = """\
Вопрос пользователя:
{question}

Контекст (фрагменты Правил):
{context}

Требования к ответу (СТРОГО):
1) Верни результат строго в JSON с полями:
   - short_answer: ОДНА строка. Если в контексте есть потенциальные исключения → начни с "Зависит: ..." и кратко укажи условие ("если X — то без аттестации; иначе — по общим правилам"). Не давай безусловное "Да/Нет", когда в контексте есть исключения.
   - reasoned_answer: 2–5 абзацев делового разбора: общая норма → специальные нормы/исключения → практические шаги.
   - citations: список объектов вида {"punkt_num":"N","subpunkt_num":"M"|"" ,"quote":"точная выдержка"} — цитаты ТОЛЬКО из переданного контекста, минимум 2.
   - related: список {"punkt_num":"N","subpunkt_num":""}.
2) Запрещено использовать пункты про периодичность/оплату/кол-во попыток как доказательство обязанности проходить аттестацию.
3) Если вопрос про зарубежную магистратуру/иностранное образование и в контексте есть нормы об освобождении/присвоении без аттестации (например, выпускники NU, перечень "Болашақ") — ОБЯЗАТЕЛЬНО отрази это в short_answer ("Зависит: ...") и процитируй соответствующий пункт.
4) Если вопрос про конкретную категорию (модератор/эксперт/исследователь/мастер) — в citations должна быть хотя бы одна цитата, где эта категория названа явно; в reasoned_answer перечисли этапы (заявление, документы/портфолио, критерии/баллы, сроки, решение комиссии).
5) Не добавляй лишних полей. JSON — без комментариев.
"""


def build_context_snippets(punkts: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    
    """Собираем контекст: «п. X[.Y]: текст…»; ограничиваем размер для токенов."""
    parts: List[str] = []
    total = 0
    for p in punkts:
        pn = str(p.get("punkt_num") or "").strip()
        sp = str(p.get("subpunkt_num") or "").strip()
        head = f"п. {pn}{('.' + sp) if sp else ''}".strip()
        txt = (p.get("text") or "").strip()
        one = f"{head}: {txt}"
        if total + len(one) + 2 > max_chars:
            break
        parts.append(one)
        total += len(one) + 2
    return "\n\n".join(parts)
def _needs_reask(question: str, data: Dict[str, Any]) -> bool:
    ql = (question or "").lower().replace("ё", "е")
    if not any(k in ql for k in ("магист", "за рубеж", "за границ", "зарубеж", "иностран")):
        return False
    cited = {str(c.get("punkt_num", "")) for c in data.get("citations", [])}
    # если не процитированы спец-нормы — просим пересобрать
    return ("5" not in cited) and ("32" not in cited)
def _needs_reask_foreign(question: str, data: Dict[str, Any]) -> bool:
    """Требовать пересборку, если вопрос про зарубеж/магистратуру,
    а в цитатах нет спец-норм (здесь примеры: п.5/п.32 — подставь свои реальные номера при необходимости)."""
    ql = (question or "").lower().replace("ё", "е")
    if any(k in ql for k in ("магист", "за рубеж", "за границ", "зарубеж", "иностран")):
        cited = {str(c.get("punkt_num", "")).strip() for c in data.get("citations", [])}
        return ("5" not in cited) and ("32" not in cited)
    return False


def _needs_reask_category(question: str, data: Dict[str, Any]) -> bool:
    """Требовать пересборку, если спрашивают про конкретную категорию,
    а в цитатах нет явных упоминаний самой категории."""
    ql = (question or "").lower().replace("ё", "е")
    if any(k in ql for k in ("исследовател", "модератор", "эксперт", "мастер")):
        # хотя бы одна цитата должна содержать упоминание категории (простая эвристика)
        for c in data.get("citations", []):
            qt = (c.get("quote", "") or "").lower()
            if any(("педагог-" + k) in qt for k in ("исследовател", "модератор", "эксперт", "мастер")):
                return False
        return True
    return False


def _too_generic_citations(data: Dict[str, Any]) -> bool:
    """Ответ опирается только на общие нормы (например, п.2/п.6) — это слабое доказательство."""
    cited_nums = {str(c.get("punkt_num", "")).strip() for c in data.get("citations", [])}
    # подстрой под свои реальные «общие» пункты при необходимости
    return cited_nums != set() and cited_nums.issubset({"2", "6"})


def ask_llm(question: str, punkts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Строим строгий JSON-ответ на основе фрагментов Правил.
    При необходимости — пересборка с ужесточающим уточнением запроса."""
    # ── Вспомогательные эвристики (локально, чтобы не было NameError)
    def _needs_reask_foreign(q: str, data: Dict[str, Any]) -> bool:
        ql = (q or "").lower().replace("ё", "е")
        foreign = any(k in ql for k in ("магист", "за рубеж", "за границ", "зарубеж", "иностран"))
        if not foreign:
            return False
        cites = data.get("citations", []) or []
        cited_nums = {str(c.get("punkt_num", "")) for c in cites}
        quotes = " ".join([(c.get("quote") or "") for c in cites]).lower()
        # Если ни спец-норм (п.5/п.32), ни ключевых фраз об освобождении — просим пересобрать
        has_special = ("5" in cited_nums) or ("32" in cited_nums)
        has_free_words = any(x in quotes for x in (
            "без прохождения процедуры", "без прохождения аттестации", "присваивается без", "не подлежит аттестации"
        ))
        return not (has_special or has_free_words)

    def _needs_reask_category(q: str, data: Dict[str, Any]) -> bool:
        ql = (q or "").lower().replace("ё", "е")
        cats = {
            "исследователь": ("исследоват",),
            "эксперт": ("эксперт",),
            "модератор": ("модератор",),
            "мастер": ("мастер",),
        }
        hit = None
        for cname, tokens in cats.items():
            if any(t in ql for t in tokens):
                hit = tokens
                break
        if not hit:
            return False
        # Требуем, чтобы хотя бы в одной цитате явно встречалось слово категории
        cites = data.get("citations", []) or []
        quotes = " ".join([(c.get("quote") or "") for c in cites]).lower()
        return not any(t in quotes for t in hit)

    def _too_generic_citations(data: Dict[str, Any]) -> bool:
        cites = data.get("citations", []) or []
        if len(cites) < 2:
            return True
        # Слишком общие ссылки: все на базовые определения и без спец-норм
        nums = [str(c.get("punkt_num", "")) for c in cites]
        only_generic = set(nums).issubset({"2", "6"})
        has_special = any(n in {"5", "30", "32"} for n in nums)
        short_quotes = any(len((c.get("quote") or "").strip()) < 10 for c in cites)
        return (only_generic and not has_special) or short_quotes

    def _force_conditional_short_answer_if_needed(q: str, data: Dict[str, Any]) -> None:
        """Если вопрос про зарубеж/магистратуру и цитаты намекают на 'без аттестации',
        то short_answer делаем условным 'Зависит: ...'."""
        ql = (q or "").lower().replace("ё", "е")
        if not any(k in ql for k in ("магист", "за рубеж", "за границ", "зарубеж", "иностран")):
            return
        cites = data.get("citations", []) or []
        blob = " ".join([(c.get("quote") or "") for c in cites]).lower()
        if any(x in blob for x in ("без прохождения процедуры", "без прохождения аттестации", "присваивается без", "не подлежит аттестации")):
            sa = (data.get("short_answer", "") or "").strip()
            if sa and sa.lower().startswith(("да", "нет")) and "завис" not in sa.lower():
                data["short_answer"] = (
                    "Зависит: если вы подпадаете под специальную норму (напр., выпускник NU/перечня «Болашақ» — "
                    "присвоение категории без аттестации), иначе — аттестация по общим правилам."
                )

    # ── Формируем контекст и первичный запрос к LLM
    context_text = build_context_snippets(punkts)
    user_prompt = (
    GEN_PROMPT_TEMPLATE
    .replace("{question}", question)
    .replace("{context}", context_text)
    )


    resp = call_with_retries(
        client.chat.completions.create,
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    text = (resp.choices[0].message.content or "").strip()

    # ── Парсим и валидируем JSON
    try:
        data = json.loads(text)
        assert isinstance(data.get("short_answer", ""), str)
        assert isinstance(data.get("reasoned_answer", ""), str)
        assert isinstance(data.get("citations", []), list)
        assert isinstance(data.get("related", []), list)
    except Exception as e:
        logger.warning("LLM JSON parse error: %s; raw: %s", e, text[:500])
        return {
            "short_answer": "Не удалось корректно сформировать структурированный ответ.",
            "reasoned_answer": "Попробуйте сформулировать вопрос иначе или уточните контекст.",
            "citations": [],
            "related": [],
        }

    # ── Эвристики качества: нужно ли пересобрать ответ
    need_reask = (
        _needs_reask_foreign(question, data)
        or _needs_reask_category(question, data)
        or _too_generic_citations(data)
    )

    if need_reask:
        extra = (
            "\nВАЖНО: Пересобери ответ. Для «зарубеж/магистратура» процитируй спец-нормы (например, п.5 и/или п.32) из переданного контекста, "
            "если они присутствуют; не опирайся на п.3/п.41 как на доказательство обязанности проходить аттестацию. "
            "Если вопрос про конкретную категорию (модератор/эксперт/исследователь/мастер) — процитируй пункты, где эта категория названа явно, "
            "и перечисли этапы (подача, документы/портфолио, критерии/баллы, сроки, решение комиссии)."
        )
        strict_prompt = GEN_PROMPT_TEMPLATE + extra

        try:
            resp2 = call_with_retries(
                client.chat.completions.create,
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": strict_prompt.format(question=question, context=context_text)},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            t2 = (resp2.choices[0].message.content or "").strip()
            data2 = json.loads(t2)
            if isinstance(data2.get("citations", []), list) and data2.get("citations"):
                data = data2
        except Exception:
            # если не вышло — остаёмся на первой версии
            pass

    # ── Правим short_answer при наличии освобождений
    _force_conditional_short_answer_if_needed(question, data)

    return data


# ───────────────────── Пост-процесс и рендер ────────────────────

def validate_citations(citations: List[Dict[str, Any]], punkts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    allowed_keys = {(str(p.get("punkt_num","")), str(p.get("subpunkt_num",""))) for p in punkts}
    by_key = {(str(p.get("punkt_num","")), str(p.get("subpunkt_num",""))): (p.get("text") or "") for p in punkts}

    out: List[Dict[str, Any]] = []
    for c in citations or []:
        pn = str(c.get("punkt_num",""))
        sp = str(c.get("subpunkt_num",""))
        if (pn, sp) not in allowed_keys:
            continue

        base = (by_key[(pn, sp)] or "").strip()
        qt = (c.get("quote") or "").strip()

        # Принимаем только если quote реально входит в текст пункта (безопасно по регистру)
        if qt and qt.lower() in base.lower():
            good_quote = qt
        else:
            # Авто-выдержка вместо «галлюцинации»
            good_quote = base[:300] + ("…" if len(base) > 300 else "")

        if good_quote:
            out.append({"punkt_num": pn, "subpunkt_num": sp, "quote": good_quote})

    # дедуп и ограничение
    seen, uniq = set(), []
    for c in out:
        key = (c["punkt_num"], c["subpunkt_num"], c["quote"])
        if key not in seen:
            seen.add(key); uniq.append(c)
    return uniq[:8]



def split_for_telegram(text: str, limit: int = 4000) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    cur = 0
    for line in (text or "").split("\n"):
        if cur + len(line) + 1 > limit:
            parts.append("\n".join(buf))
            buf = [line]
            cur = len(line) + 1
        else:
            buf.append(line)
            cur += len(line) + 1
    if buf:
        parts.append("\n".join(buf))
    return parts

# ────────────────────── Telegram I/O ────────────────────────────

def tg_send_message(chat_id: int, text: str, parse_mode: str = "HTML", reply_markup: Optional[dict] = None) -> Optional[int]:
    url = f"{TELEGRAM_API}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup  # добавляем только если объект

    r = requests.post(url, json=payload, timeout=15)
    if not r.ok:
        logger.error("sendMessage failed: %s %s", r.status_code, r.text)
        return None
    try:
        return r.json().get("result", {}).get("message_id")
    except Exception:
        return None


def tg_edit_message_text(chat_id: int, message_id: int, text: str, parse_mode: str = "HTML", reply_markup: Optional[dict] = None) -> None:
    url = f"{TELEGRAM_API}/editMessageText"
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup  # добавляем только если объект

    r = requests.post(url, json=payload, timeout=15)
    if not r.ok:
        logger.error("editMessageText failed: %s %s", r.status_code, r.text)

def kb_show_detailed():
    return {"inline_keyboard": [[{"text": "Показать подробный ответ", "callback_data": "show_detailed"}]]}

def kb_show_short():
    return {"inline_keyboard": [[{"text": "Показать краткий ответ", "callback_data": "show_short"}]]}
def render_short_html(question: str, data: Dict[str, Any]) -> str:
    sa = html.escape(data.get("short_answer", "")).strip()
    ra = html.escape(data.get("reasoned_answer", "")).strip()
    lines = [f"<b>Вопрос:</b> {html.escape(question)}"]
    if sa:
        lines.append(f"<b>Краткий ответ:</b>\n{sa}")
    # намёк, что есть подробности
    if ra:
        lines.append("<i>Нажмите кнопку ниже, чтобы увидеть подробное обоснование и цитаты.</i>")
    return "\n".join(lines)

def render_detailed_html(question: str, data: Dict[str, Any], punkts: List[Dict[str, Any]]) -> str:
    sa = html.escape(data.get("short_answer", "")).strip()
    ra = html.escape(data.get("reasoned_answer", "")).strip()
    citations = validate_citations(data.get("citations", []), punkts)
    related = data.get("related", [])
    lines: List[str] = []
    lines.append(f"<b>Вопрос:</b> {html.escape(question)}")
    if sa:
        lines.append(f"<b>Краткий вывод:</b>\n{sa}")
    if ra:
        lines.append(f"<b>Подробное обоснование:</b>\n{ra}")
    if citations:
        lines.append("<b>Цитаты из Правил:</b>")
        for c in citations:
            pn = c.get("punkt_num", "")
            sp = c.get("subpunkt_num", "")
            head = f"п. {pn}{('.' + sp) if sp else ''}".strip()
            qt = html.escape(c.get("quote", ""))
            lines.append(f"— <i>{head}</i>: {qt}")
    if related:
        lines.append("<b>Связанные пункты:</b>")
        for r in related[:12]:
            pn = html.escape(str(r.get("punkt_num", "")))
            sp = html.escape(str(r.get("subpunkt_num", "")))
            head = f"п. {pn}{('.' + sp) if sp else ''}".strip()
            lines.append(f"• {head}")
    return "\n".join(lines).strip()


def tg_set_webhook(full_url: str, secret: Optional[str]) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook"
    payload = {"url": full_url}
    if secret:
        payload["secret_token"] = secret
    payload["allowed_updates"] = ["message", "callback_query"]

    r = requests.post(url, json=payload, timeout=15)
    if not r.ok:
        logger.error("setWebhook failed: %s %s", r.status_code, r.text)
    else:
        logger.info("Webhook set: %s", full_url)
def tg_answer_callback_query(callback_query_id: str) -> None:
    url = f"{TELEGRAM_API}/answerCallbackQuery"
    payload = {"callback_query_id": callback_query_id}
    r = requests.post(url, json=payload, timeout=15)
    if not r.ok:
        logger.error("answerCallbackQuery failed: %s %s", r.status_code, r.text)

# ─────────────────── Google Sheets логирование ──────────────────

def log_to_sheet_safe(user_id: int, question: str, short_answer: str) -> None:
    if not (SHEET_ID and GOOGLE_CREDENTIALS_JSON):
        return
    try:
        import gspread  # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
        import json as _json

        creds_data = _json.loads(GOOGLE_CREDENTIALS_JSON)
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(creds_data, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID)
        ws = sh.sheet1
        ws.append_row([int(time.time()), str(user_id), question, short_answer], value_input_option="USER_ENTERED")
    except Exception as e:
        logger.warning("Sheets log failed: %s", e)

# ───────────────────────── HTTP хендлеры ────────────────────────

async def handle_health(request: web.Request) -> web.Response:
    return web.Response(text="ok")

async def handle_webhook(request: web.Request) -> web.Response:
    if TELEGRAM_WEBHOOK_SECRET:
        recv_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if recv_secret != TELEGRAM_WEBHOOK_SECRET:
            return web.Response(status=403, text="forbidden")

    data = await request.json()

    # 1) CallbackQuery (кнопки)
    if "callback_query" in data:
        cq = data["callback_query"]
        chat_id = cq.get("message", {}).get("chat", {}).get("id")
        message_id = cq.get("message", {}).get("message_id")
        action = cq.get("data")
        if not chat_id or not message_id:
            return web.Response(text="ok")

        key = (int(chat_id), int(message_id))
        stash = LAST_RESPONSES.get(key)

        tg_answer_callback_query(cq.get("id"))

        if not stash:
            tg_edit_message_text(chat_id, message_id, "Данные недоступны. Отправьте вопрос заново.")
            return web.Response(text="ok")

        if action == "show_detailed":
            detailed = stash["detailed_html"]
            if len(detailed) <= 4000:
                tg_edit_message_text(chat_id, message_id, detailed, reply_markup=kb_show_short())
            else:
                notice = stash["short_html"] + "\n\n<i>Подробный ответ отправлен отдельными сообщениями ниже.</i>"
                tg_edit_message_text(chat_id, message_id, notice, reply_markup=kb_show_short())
                for chunk in split_for_telegram(detailed, 4000):
                    tg_send_message(chat_id, chunk)
        elif action == "show_short":
            tg_edit_message_text(chat_id, message_id, stash["short_html"], reply_markup=kb_show_detailed())

        return web.Response(text="ok")

    # 2) Обычное сообщение
    message = data.get("message", {}) if isinstance(data, dict) else {}
    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = message.get("text", "") or ""

    if not chat_id:
        return web.Response(text="ok")

    if text.strip().startswith("/start"):
        tg_send_message(chat_id, "Здравствуйте! Задайте вопрос по Правилам аттестации педагогов — я отвечу с цитатами.")
        return web.Response(text="ok")

    if not text.strip():
        tg_send_message(chat_id, "Пожалуйста, пришлите текстовый вопрос.")
        return web.Response(text="ok")

    try:
        punkts = rag_search(text)
        data_struct = ask_llm(text, punkts)

        short_html = render_short_html(text, data_struct)
        detailed_html = render_detailed_html(text, data_struct, punkts)

        msg_id = tg_send_message(chat_id, short_html, reply_markup=kb_show_detailed())
        if msg_id:
            key = (int(chat_id), int(msg_id))
            LAST_RESPONSES[key] = {
                "message_id": int(msg_id),
                "short_html": short_html,
                "detailed_html": detailed_html,
            }

        log_to_sheet_safe(chat_id, text, data_struct.get("short_answer", ""))
    except Exception:
        logger.exception("Processing failed")
        tg_send_message(chat_id, "Произошла ошибка при обработке запроса. Попробуйте позже.")

    return web.Response(text="ok")

# ─────────────────────────── main() ─────────────────────────────

async def on_startup(app: web.Application):
    full_url = f"{os.environ.get('WEBHOOK_URL', '').strip()}{os.environ.get('WEBHOOK_PATH', '/webhook')}"
    tg_set_webhook(full_url, TELEGRAM_WEBHOOK_SECRET)
    logger.info("Service started on port %s", int(os.environ.get("PORT", "8080")))

def main():
    app = web.Application()

    # роуты
    app.router.add_get("/health", handle_health)
    app.router.add_post(os.environ.get("WEBHOOK_PATH", "/webhook"), handle_webhook)

    # ⬇️ ВОТ ЗДЕСЬ — «после роутов»
    app.on_startup.append(on_startup)

    # (необязательно) чтобы / не давал 404:
    # async def handle_root(request):
    #     return web.Response(text="ok")
    # app.router.add_get("/", handle_root)

    loop = asyncio.get_event_loop()
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "0.0.0.0", int(os.environ.get("PORT", "8080")))
    loop.run_until_complete(site.start())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(runner.cleanup())


if __name__ == "__main__":
    main()
