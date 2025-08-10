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
W_CAT = 1.8
CAT_KEYS = ("исследовател", "модератор", "эксперт", "мастер")


def kw_category_boost(question: str, doc_text: str) -> float:
    ql = (question or "").lower().replace("ё", "е")
    dl = (doc_text or "").lower().replace("ё", "е")
    if not any(k in ql for k in CAT_KEYS):
        return 0.0
    # принимаем дефис/длинное тире/среднее тире/пробел, и также просто наличие корня
    variants = ("педагог-", "педагог —", "педагог –", "педагог ")
    for k in CAT_KEYS:
        if k in ql:
            if k in dl:
                return 1.0
            if any((v + k) in dl for v in variants):
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
KW_PERIOD_TERMS = ("каждые пять лет", "не реже одного раза в пять лет", "периодичност", "оплат", "стоимост", "количеств", "попыток")

def kw_boost(question: str, doc_text: str) -> float:
    ql = (question or "").lower().replace("ё", "е")
    dl = (doc_text or "").lower().replace("ё", "е")

    boost = 0.0
    is_category_q = any(k in ql for k in CAT_KEYS)

    foreign_q = any(k in ql for k in ("магист", "за рубеж", "за границ", "зарубеж", "иностран"))
    if foreign_q and not is_category_q:
        if any(k in dl for k in KW_EXCEPTION_TERMS):
            boost += 0.6
        if any(k in dl for k in KW_FOREIGN_TERMS):
            boost += 0.6

    # общий случай: «без прохождения...» слегка подбустим,
    # НО если вопрос про категорию — наоборот, чуть прижмём такие пункты
    has_exception_phrase = any(k in dl for k in KW_EXCEPTION_TERMS)
    if has_exception_phrase:
        boost += 0.3
        if is_category_q:
            boost -= 0.6  # штраф для льгот, когда спрашивают про категорию
    # Анти-шум: если вопрос про категорию или про зарубеж, а документ про периодичность/оплату — слегка штрафуем
    if (is_category_q or foreign_q) and any(k in dl for k in KW_PERIOD_TERMS):
        boost -= 0.4

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
        model=CHAT_MODEL,  # вместо "gpt-4o-mini"
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
    """
    Гибридный retrieve: эмбеддинги + BM25 + маппинги/регэкспы + тематические бусты.
    Плюс форс-добавление пунктов, где явно встречается запрошенная категория
    (исследователь/модератор/эксперт/мастер), чтобы LLM мог процитировать их.
    """
    q = normalize_query(q)
    ql = q.lower().replace("ё", "е")

    variants = multi_query_rewrites(q)
    dense_agg: Dict[int, float] = {}
    sparse_agg: Dict[int, float] = {}

    # Dense (эмбеддинги) для базового и вариантов
    for v in variants:
        for idx, sc in vector_search(v, top_k=top_k_stage1):
            dense_agg[idx] = max(dense_agg.get(idx, 0.0), sc)

    # HyDE (если включен)
    hyde = hyde_passage(q)
    if hyde:
        for idx, sc in vector_search(hyde, top_k=top_k_stage1 // 2):
            dense_agg[idx] = max(dense_agg.get(idx, 0.0), sc)

    # Sparse (BM25/ключевые слова)
    for v in [q] + variants:
        for idx, sc in bm25_search(v, top_k=top_k_stage1):
            sparse_agg[idx] = max(sparse_agg.get(idx, 0.0), sc)

    # Явные попадания (регэксп/маппинг)
    regex_idx = regex_hits(q)
    mapped_idx = mapped_hits(q)

    # Нормируем sparse (z-score), чтобы не «перебивал» эмбеддинги на длинных текстах
    if sparse_agg:
        vals = np.array(list(sparse_agg.values()), dtype=np.float64)
        mu = float(vals.mean())
        sigma = float(vals.std() + 1e-6)
        for k in list(sparse_agg.keys()):
            sparse_agg[k] = (sparse_agg[k] - mu) / sigma

    # Сводный скор
    items: List[Tuple[int, float]] = []
    candidate_ids = set(list(dense_agg.keys()) + list(sparse_agg.keys()) + regex_idx + mapped_idx)
    for idx in candidate_ids:
        emb_sc = dense_agg.get(idx, 0.0)
        sp_sc  = sparse_agg.get(idx, 0.0)
        rx_sc  = 1.0 if idx in regex_idx  else 0.0
        mp_sc  = 1.0 if idx in mapped_idx else 0.0

        total = (
            W_EMB * emb_sc
            + W_BM25 * sp_sc
            + W_REGEX * rx_sc
            + W_MAP * mp_sc
        )
        # тематические бусты (иностранное/исключения, конкретные категории)
        txt = PUNKTS[idx].get("text", "")
        total += W_KW  * kw_boost(q, txt)
        total += W_CAT * kw_category_boost(q, txt)

        items.append((idx, total))
    

    # ── Приоритизация обязательных пунктов + Top-K ──
    # 1) сортируем все кандидаты по убыванию суммарного скора
    items.sort(key=lambda x: -x[1])
    ranked = [i for i, _ in items]

    # 2) собираем "обязательные" пункты, которые должны попасть в итог
    must_have: List[int] = []

    # 2.1) явные совпадения из mapped/regex
    must_have.extend(mapped_idx + regex_idx)  # порядок не критичен, дубли уберём ниже

    # 2.2) если в вопросе явно упомянута категория — форсим все пункты, где она названа
    for k in CAT_KEYS:
        if k in ql:
            cat_ids = [
                i for i, p in enumerate(PUNKTS)
                if k in (p.get("text", "").lower().replace("ё", "е"))
            ]
            must_have.extend(cat_ids)
            break  # достаточно первой найденной категории

  
    # 2.3) если вопрос про зарубеж/магистратуру — форсим п.32
    foreign_q = any(k in ql for k in ("магист", "зарубеж", "за границ", "иностран", "болаш", "nazarbayev", "nazarbayev university"))
    if foreign_q:
        forced_32 = [i for i, p in enumerate(PUNKTS) if str(p.get("punkt_num","")).strip() == "32"]
        # ставим п.32 в самое начало must_have
        must_have = forced_32 + must_have

    # 3) формируем итоговый список: сначала must_have, потом — по рейтингу; без дублей и с обрезкой до K
    top_idx: List[int] = []
    for i in must_have + ranked:
        if i not in top_idx:
            top_idx.append(i)
        if len(top_idx) >= final_k:
            break

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
1) Верни СТРОГО корректный JSON с полями:
   "short_answer": одна строка (≤200 символов). Форматируй так:
     - если в контексте есть освобождение/льгота → "Зависит: если <условие>, то <итог>; иначе — по общим правилам".
     - иначе → "По общим правилам: <итог>".
   "reasoned_answer": 1–3 коротких абзаца (главная норма → спец-исключение → практический вывод). Без описания процедур, если их нет в цитатах.
   "citations": СПИСОК объектов {"punkt_num":"N","subpunkt_num":"M" или "","quote":"точная выдержка из контекста"}.
     Минимум 1–2 шт. Сначала ключевая норма по сути вопроса.
   "related": СПИСОК объектов {"punkt_num":"N","subpunkt_num":"M" или ""} (может быть пустым []).

2) Цитаты берём ТОЛЬКО из переданного контекста. Если точную фразу трудно выделить — процитируй короткий фрагмент (≤180 знаков).
3) НЕ упоминай периодичность/оплату/ОЗП/кол-во попыток и т.п., если ЭТО НЕ процитировано.
4) Для вопросов про зарубежную магистратуру/иностр. образование, если в контексте есть освобождение (напр., п.32):
   — отрази это в short_answer в формате "Зависит: …; иначе — по общим правилам" и процитируй норму как первую в "citations".
5) Для вопросов про конкретную категорию (модератор/эксперт/исследователь/мастер) — среди "citations" должна быть цитата,
   где категория названа явно.
6) JSON — без лишних полей, без комментариев, кавычки только двойные.
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

def ask_llm(question: str, punkts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Строит строгий JSON-ответ на основе фрагментов Правил, с пост-коэрсией схемы.
    При слабом ответе — повторный ужесточённый запрос.
    """

    # ───── helpers: схема/валидация ─────

    def _build_user_prompt(template: str, q: str, context: str) -> str:
        return template.replace("{question}", q).replace("{context}", context)

    def _allowed_keys_set(punkts_: List[Dict[str, Any]]) -> set[Tuple[str, str]]:
        return {(str(p.get("punkt_num","")).strip(), str(p.get("subpunkt_num","")).strip()) for p in punkts_}

    def _closest_key(pn: str, sp: str, allowed: set[Tuple[str,str]]) -> Optional[Tuple[str,str]]:
        # точное совпадение
        key = (pn, sp)
        if key in allowed:
            return key
        # если подпункт не нашли — попробуем без подпункта
        key2 = (pn, "")
        if key2 in allowed:
            return key2
        return None

    _ALLOWED = _allowed_keys_set(punkts)

    _NUM_RE = re.compile(r"(\d{1,3})(?:\.(\d{1,3}))?")

    def _extract_pairs_from_text(s: str) -> List[Tuple[str,str]]:
        out = []
        for m in _NUM_RE.finditer(s or ""):
            pn = m.group(1) or ""
            sp = m.group(2) or ""
            if pn:
                key = _closest_key(pn, sp, _ALLOWED)
                if key:
                    out.append(key)
        # дедуп
        seen, uniq = set(), []
        for k in out:
            if k not in seen:
                seen.add(k); uniq.append(k)
        return uniq

    def _coerce_citations(raw: Any) -> List[Dict[str, str]]:
        """
        Приводим citations к списку объектов {punkt_num, subpunkt_num, quote}.
        Допускаем вход: строка "п. 32", список строк, список словарей c любыми полями.
        Цитаты-выдержки подставим позже в рендере (validate_citations), здесь можно оставить пустую строку.
        """
        res: List[Dict[str,str]] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    pn = str(item.get("punkt_num","")).strip()
                    sp = str(item.get("subpunkt_num","")).strip()
                    if not pn and isinstance(item.get("quote"), str):
                        # попробуем выдернуть номер из quote
                        pairs = _extract_pairs_from_text(item.get("quote",""))
                        if pairs:
                            pn, sp = pairs[0]
                    if pn:
                        key = _closest_key(pn, sp, _ALLOWED)
                        if key:
                            res.append({"punkt_num": key[0], "subpunkt_num": key[1], "quote": str(item.get("quote","")).strip()})
                elif isinstance(item, str):
                    for pn, sp in _extract_pairs_from_text(item):
                        res.append({"punkt_num": pn, "subpunkt_num": sp, "quote": ""})
        elif isinstance(raw, str):
            for pn, sp in _extract_pairs_from_text(raw):
                res.append({"punkt_num": pn, "subpunkt_num": sp, "quote": ""})
        # дедуп по (pn,sp,quote)
        seen, uniq = set(), []
        for c in res:
            key = (c["punkt_num"], c["subpunkt_num"], c["quote"])
            if key not in seen:
                seen.add(key); uniq.append(c)
        return uniq

    def _coerce_related(raw: Any) -> List[Dict[str,str]]:
        res: List[Dict[str,str]] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    pn = str(item.get("punkt_num","")).strip()
                    sp = str(item.get("subpunkt_num","")).strip()
                    if pn:
                        key = _closest_key(pn, sp, _ALLOWED)
                        if key:
                            res.append({"punkt_num": key[0], "subpunkt_num": key[1]})
                elif isinstance(item, str):
                    for pn, sp in _extract_pairs_from_text(item):
                        res.append({"punkt_num": pn, "subpunkt_num": sp})
        elif isinstance(raw, str):
            for pn, sp in _extract_pairs_from_text(raw):
                res.append({"punkt_num": pn, "subpunkt_num": sp})
        # дедуп по (pn,sp)
        seen, uniq = set(), []
        for r in res:
            key = (r["punkt_num"], r["subpunkt_num"])
            if key not in seen:
                seen.add(key); uniq.append(r)
        return uniq

    def _normalize_llm_json(d: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "short_answer": str(d.get("short_answer","")).strip(),
            "reasoned_answer": str(d.get("reasoned_answer","")).strip(),
            "citations": _coerce_citations(d.get("citations", [])),
            "related": _coerce_related(d.get("related", [])),
        }
        return out

    # ───── первичный запрос ─────
    context_text = build_context_snippets(punkts)
    user_prompt = _build_user_prompt(GEN_PROMPT_TEMPLATE, question, context_text)

    resp = call_with_retries(
        client.chat.completions.create,
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw_text = (resp.choices[0].message.content or "").strip()

    # ───── парсинг + коэрсия схемы ─────

    # ───── парсинг + коэрсия схемы ─────
    try:
        data_raw = json.loads(raw_text)
    except Exception as e:
        logger.warning("LLM JSON decode error: %s; trying strict reprompt. Raw: %s", e, raw_text[:500])
        # жёсткий повторный запрос сразу, если JSON не распарсился
        extra = (
            "\nВАЖНО: Пересобери ответ строго по схеме. "
            "citations — это СПИСОК ОБЪЕКТОВ {punkt_num, subpunkt_num, quote}; "
            "related — СПИСОК ОБЪЕКТОВ {punkt_num, subpunkt_num}. "
            "Минимум две уникальные цитаты из ПЕРЕДАННОГО контекста."
        )
        strict_prompt = _build_user_prompt(GEN_PROMPT_TEMPLATE + extra, question, context_text)
        resp2 = call_with_retries(
            client.chat.completions.create,
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": strict_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw_text2 = (resp2.choices[0].message.content or "").strip()
        try:
            data_raw = json.loads(raw_text2)
        except Exception as e2:
            logger.warning("LLM JSON decode error (strict) too: %s; raw: %s", e2, raw_text2[:500])
            return {
                "short_answer": "Не удалось корректно сформировать структурированный ответ.",
                "reasoned_answer": "Попробуйте сформулировать вопрос иначе или уточните контекст.",
                "citations": [],
                "related": [],
            }


    data = _normalize_llm_json(data_raw)

    # базовая проверка типов (после коэрсии)
    if not isinstance(data["short_answer"], str) or not isinstance(data["reasoned_answer"], str) \
       or not isinstance(data["citations"], list) or not isinstance(data["related"], list):
        logger.warning("LLM schema still invalid after normalize; raw: %s", str(data_raw)[:500])
        return {
            "short_answer": "Не удалось корректно сформировать структурированный ответ.",
            "reasoned_answer": "Попробуйте сформулировать вопрос иначе или уточните контекст.",
            "citations": [],
            "related": [],
        }

    # ───── эвристики: нужна ли пересборка ─────
    def _mentions_obligation(txt: str) -> bool:
        return bool(re.search(r"\b(обязан|обязательно|должен|необходимо)\b", (txt or "").lower()))
    def _cit_pts(cits: List[Dict[str,str]]) -> set:
        return {c.get("punkt_num","") for c in (cits or [])}

    need_reask = False

    # иностр. образование — хотим видеть спец-норму
    ql = (question or "").lower()
    if any(k in ql for k in ("магист", "за рубеж", "за границ", "зарубеж", "иностран", "болаш", "bolash", "nazarbayev", "nazarbayev university")):
        if "32" not in _cit_pts(data["citations"]):
            need_reask = True

    

    # слишком мало уникальных цитат?
    uniq_cits = {(c.get("punkt_num",""), c.get("subpunkt_num","")) for c in data["citations"]}
    if len([u for u in uniq_cits if u[0]]) < 2:
        need_reask = True

    # «обязан/обязательно…», но среди цитат есть 3/41 — переспросим
    if _mentions_obligation(data["reasoned_answer"]) and (_cit_pts(data["citations"]) & {"3","41"}):
        need_reask = True

    if not need_reask:
        return data

    # ───── строгий повторный запрос ─────
    extra = (
        "\nВАЖНО: Пересобери ответ.\n"
        "1) Для вопросов про зарубежную магистратуру процитируй специальные нормы (например, п.32) из ПЕРЕДАННОГО контекста;\n"
        "2) НЕ используй п.3/п.41 как доказательство обязанности проходить аттестацию (их можно упоминать только как справочную информацию);\n"
        "3) Для вопросов про конкретную категорию (модератор/эксперт/исследователь/мастер) — процитируй фрагменты, где эта категория названа явно,\n"
        "   и кратко перечисли этапы: заявление, документы/портфолио, критерии/баллы, сроки, решение комиссии;\n"
        "4) Сохраняй строгую схему: citations — СПИСОК ОБЪЕКТОВ {punkt_num, subpunkt_num, quote}; related — СПИСОК ОБЪЕКТОВ {punkt_num, subpunkt_num}.\n"
        "5) Минимум две уникальные цитаты."
    )
    strict_prompt = _build_user_prompt(GEN_PROMPT_TEMPLATE + extra, question, context_text)

    resp2 = call_with_retries(
        client.chat.completions.create,
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": strict_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw_text2 = (resp2.choices[0].message.content or "").strip()

    try:
        data_raw2 = json.loads(raw_text2)
        data2 = _normalize_llm_json(data_raw2)

        uniq_cits2 = {(c.get("punkt_num",""), c.get("subpunkt_num","")) for c in data2["citations"]}
        ok = (
            isinstance(data2["short_answer"], str)
            and isinstance(data2["reasoned_answer"], str)
            and isinstance(data2["citations"], list)
            and isinstance(data2["related"], list)
            and len([u for u in uniq_cits2 if u[0]]) >= 2
            and not (_mentions_obligation(data2["reasoned_answer"]) and (_cit_pts(data2["citations"]) & {"3","41"}))
        )
        if ok:
            return data2
    except Exception as e:
        logger.warning("LLM JSON parse (strict) error: %s; raw: %s", e, raw_text2[:500])

    # если пересборка не помогла — возвращаем первичный нормализованный
    return data



def enforce_short_answer(question: str, data: dict, ctx_text: str) -> dict:
    import re
    sa = (data.get("short_answer") or "").strip()
    cites = {str(c.get("punkt_num","")) for c in (data.get("citations") or [])}
    ql = (question or "").lower()
    ctx = (ctx_text or "").lower()

    foreign_trigger = any(t in ql for t in ("магист", "за рубеж", "зарубеж", "иностран", "болаш", "nazarbayev"))
    is_category_q = any(k in ql for k in ("исследовател", "модератор", "эксперт", "мастер"))

    # Если есть льгота (п.32 в цитатах) и вопрос про зарубеж — формируем строго по шаблону
    if foreign_trigger and ("32" in cites or "без прохождения процедуры аттестации" in ctx):
        sa = ("Зависит: если магистратура окончена в зарубежной организации из перечня «Болашақ», "
              "категория «педагог-модератор» присваивается без аттестации; иначе — по общим правилам.")
    else:
        # Категории — приводим к "По общим правилам"
        if is_category_q and not sa.lower().startswith(("по общим правилам:", "зависит:")):
            sa = "По общим правилам: " + sa

    # Нормируем ответы вида "да/нет"
    if re.fullmatch(r"\s*(да|нет)[\.\!]*\s*", sa, flags=re.I):
        sa = "По общим правилам: " + sa.capitalize()

    # Убираем жёсткие формулировки обязательности, если они опираются на п.3/п.41
    bad = {"3","41"}
    if (cites & bad):
        sa = re.sub(r"\b(обязан|обязательно|должен|необходимо)\b", "требуется по общим нормам", sa, flags=re.I)
    # Если это вопрос про категорию и среди цитат есть п.10 (про процедуру) — добавим компактные этапы
    if is_category_q and "10" in cites and "→" not in sa:
        tail = " (этапы: заявление → портфолио → ОЗП → обобщение → решение комиссии)"
        if len(sa) + len(tail) <= 200:
            sa += tail


    data["short_answer"] = sa[:200]
    return data





# ───────────────────── Пост-процесс и рендер ────────────────────

def validate_citations(citations: List[Dict[str, Any]], punkts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Очищает и нормализует цитаты:
    - оставляет только те, что есть в выданном контексте;
    - полностью выкидывает «мусорные» пункты (напр., 3 и 41);
    - если quote не найден дословно в тексте пункта — подставляет безопасную авто-выдержку;
    - дедуп и ограничение до 8 элементов.
    """
    SKIP_AS_EVIDENCE = {"3", "41"}

    allowed_keys: set[Tuple[str, str]] = {
        (str(p.get("punkt_num", "")), str(p.get("subpunkt_num", ""))) for p in punkts
    }
    by_key: Dict[Tuple[str, str], str] = {
        (str(p.get("punkt_num", "")), str(p.get("subpunkt_num", ""))): (p.get("text") or "")
        for p in punkts
    }

    out: List[Dict[str, Any]] = []
    for c in (citations or []):
        pn = str(c.get("punkt_num", "")).strip()
        sp = str(c.get("subpunkt_num", "")).strip()
        if not pn:
            continue

        # выкидываем п.3/п.41 целиком
        if pn in SKIP_AS_EVIDENCE:
            continue

        if (pn, sp) not in allowed_keys:
            continue

        base = (by_key.get((pn, sp), "") or "").strip()
        if not base:
            continue
        qt = (c.get("quote") or "").strip()
        base_clean = re.sub(r"\s+", " ", base).strip()
        if qt and qt.lower() in base.lower():
            qt_clean = re.sub(r"\s+", " ", qt).strip()
            good_quote = qt_clean[:180] + ("…" if len(qt_clean) > 180 else "")
        else:
            good_quote = base_clean[:180] + ("…" if len(base_clean) > 180 else "")



        out.append({"punkt_num": pn, "subpunkt_num": sp, "quote": good_quote})

    # дедуп и лимит
    seen: set[Tuple[str, str, str]] = set()
    uniq: List[Dict[str, Any]] = []
    for c in out:
        key = (c["punkt_num"], c["subpunkt_num"], c["quote"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    return uniq[:8]
def _ensure_category_citation(question: str,
                              citations: List[Dict[str, Any]],
                              punkts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ql = (question or "").lower().replace("ё", "е")
    cats = ("исследовател", "модератор", "эксперт", "мастер")
    target_cat = next((c for c in cats if c in ql), None)
    if not target_cat:
        return citations

    # если уже есть цитата с нужной категорией — ничего не делаем
    by_key_txt = {
        (str(p.get("punkt_num","")).strip(), str(p.get("subpunkt_num","")).strip()):
            (p.get("text") or "").lower().replace("ё", "е")
        for p in punkts
    }
    def _mentions_cat(c: Dict[str, Any]) -> bool:
        key = (str(c.get("punkt_num","")).strip(), str(c.get("subpunkt_num","")).strip())
        return target_cat in by_key_txt.get(key, "")

    if any(_mentions_cat(c) for c in (citations or [])):
        return citations

    # иначе найдём подходящий пункт в контексте и добавим его первой цитатой
    for p in punkts:
        txt = (p.get("text") or "").lower().replace("ё", "е")
        if target_cat in txt:
            pn = str(p.get("punkt_num","")).strip()
            sp = str(p.get("subpunkt_num","")).strip()
            return [{"punkt_num": pn, "subpunkt_num": sp, "quote": ""}] + (citations or [])
    return citations

# ── Фильтрация цитат по ключевым словам из вопроса ──
def filter_citations_by_question(
    question: str,
    citations: List[Dict[str, Any]],
    punkts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Делает цитаты максимально релевантными вопросу + перекраивает quote на нужный фрагмент.
    — «зарубеж/магистратура/Болашак»: п.32 первым, максимум 1–2 цитаты, quote вырезаем вокруг ключевых слов.
    — «конкретная категория»: оставляем пункты, где категория названа явно (до 3), quote вырезаем вокруг категории.
    — Пункты 3 и 41 выкидываем как «доказательство».
    — По умолчанию: до 3 цитат.
    """
    import re
    ql = (question or "").lower().replace("ё", "е")
    if not citations:
        return citations

    # Полные тексты пунктов по ключу
    by_key_full = {
        (str(p.get("punkt_num", "")), str(p.get("subpunkt_num", ""))): (p.get("text") or "")
        for p in punkts
    }
    by_key = {
        k: v.lower().replace("ё", "е") for k, v in by_key_full.items()
    }

    # На всякий случай выкинем 3/41 (validate_citations уже делает это)
    clean = [c for c in citations if str(c.get("punkt_num", "")) not in {"3", "41"}]
    if not clean:
        clean = citations[:]

    # Хелпер: вырезать 180-символьный отрывок вокруг первого попадания любой из ключевых фраз
    def _crop_around(text_full: str, keys: Tuple[str, ...], width: int = 180) -> str:
        tf = re.sub(r"\s+", " ", text_full or "").strip()
        tl = tf.lower().replace("ё", "е")
        pos = -1
        for k in keys:
            i = tl.find(k)
            if i != -1 and (pos == -1 or i < pos):
                pos = i
        if pos == -1:
            return tf[:width] + ("…" if len(tf) > width else "")
        # центрируем окно около попадания
        pad = width // 2
        start = max(0, pos - pad)
        end = min(len(tf), pos + pad)
        # обрезка по словам
        while start > 0 and tf[start] not in " .,;:!?()[]{}«»":
            start -= 1
        while end < len(tf) and tf[end-1] not in " .,;:!?()[]{}«»":
            end += 1
        snippet = tf[start:end].strip()
        return snippet[:width] + ("…" if len(snippet) > width else "")

    # Если вопрос про «зарубеж/магистратуру»
    foreign = any(k in ql for k in ("магист", "зарубеж", "за границ", "иностран", "болаш", "bolash", "nazarbayev"))
    category_keys = ("исследовател", "модератор", "эксперт", "мастер")
    is_category_q = any(k in ql for k in category_keys)

    if foreign:
        p32 = [c for c in clean if str(c.get("punkt_num", "")).strip() == "32"]
        rest = [c for c in clean if str(c.get("punkt_num", "")).strip() != "32"]
        pref_terms = ("зарубеж", "болаш", "nazarbayev", "без прохождения")

        def _rel(c: Dict[str, Any]) -> bool:
            key = (str(c.get("punkt_num","")), str(c.get("subpunkt_num","")))
            return any(t in by_key.get(key, "") for t in pref_terms)

        ordered = p32 + sorted(rest, key=lambda c: (not _rel(c)))
        out = (ordered[:2] or clean[:2])

        # Перекраиваем quote на релевантное место
        for c in out:
            key = (str(c.get("punkt_num","")), str(c.get("subpunkt_num","")))
            base_full = by_key_full.get(key, "")  # исходный текст пункта (без lower)
            if base_full:
                c["quote"] = _crop_around(base_full, pref_terms)
        return out

    if is_category_q:
        def _mentions_category(c: Dict[str, Any]) -> bool:
            key = (str(c.get("punkt_num","")), str(c.get("subpunkt_num","")))
            return any(k in by_key.get(key, "") for k in category_keys)

        cat_cits = [c for c in clean if _mentions_category(c)]
        out = (cat_cits or clean)[:3]

        # Выбираем ключ согласно вопросу (точнее таргетим)
        target = next((k for k in category_keys if k in ql), None)
        target_terms = (target,) if target else category_keys

        # Перекраиваем quote так, чтобы в отрывок попала нужная категория
        for c in out:
            key = (str(c.get("punkt_num","")), str(c.get("subpunkt_num","")))
            base_full = by_key_full.get(key, "")
            if base_full:
                c["quote"] = _crop_around(base_full, target_terms)
        return out

    # По умолчанию — компактно до 3 и безопасный quote, если LLM дал пустой
    out = clean[:3]
    for c in out:
        if not (c.get("quote") or "").strip():
            key = (str(c.get("punkt_num","")), str(c.get("subpunkt_num","")))
            base_full = by_key_full.get(key, "")
            if base_full:
                c["quote"] = _crop_around(base_full, tuple())
    return out


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

    # ДО валидации: гарантируем, что при вопросе про категорию первая цитата — с нужной категорией
    data["citations"] = _ensure_category_citation(question, data.get("citations", []), punkts)

    citations = validate_citations(data.get("citations", []), punkts)
    citations = filter_citations_by_question(question, citations, punkts)
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

        # ДО рендера — подстрахуем short_answer по найденному контексту
        context_text = build_context_snippets(punkts)
        data_struct = enforce_short_answer(text, data_struct, context_text)

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
        if len(LAST_RESPONSES) > 200:
           # удаляем самый старый
           FIRST = next(iter(LAST_RESPONSES))
           if FIRST != key:
               LAST_RESPONSES.pop(FIRST, None)
        log_to_sheet_safe(chat_id, text, data_struct.get("short_answer", ""))
    except Exception:
        logger.exception("Processing failed")
        tg_send_message(chat_id, "Произошла ошибка при обработке запроса. Попробуйте позже.")

    return web.Response(text="ok")

# ─────────────────────────── main() ─────────────────────────────

async def on_startup(app: web.Application):
    full_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
    tg_set_webhook(full_url, TELEGRAM_WEBHOOK_SECRET)
    logger.info("Service started on port %s", PORT)

def main():
    app = web.Application()

    app.router.add_get("/health", handle_health)
    app.router.add_post(WEBHOOK_PATH, handle_webhook)

    async def handle_root(request):
        return web.Response(text="ok")
    app.router.add_get("/", handle_root)

    app.on_startup.append(on_startup)

    loop = asyncio.get_event_loop()
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    loop.run_until_complete(site.start())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(runner.cleanup())



if __name__ == "__main__":
    main()
