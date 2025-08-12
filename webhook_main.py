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
import threading  # NEW

from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import requests
from aiohttp import web
from collections import OrderedDict  # NEW

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
# sanitize WEBHOOK_PATH/WEBHOOK_URL
if not WEBHOOK_PATH.startswith("/"):
    WEBHOOK_PATH = "/" + WEBHOOK_PATH
WEBHOOK_URL = WEBHOOK_URL.rstrip("/")
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
W_BM25 = 1.0
W_MAP = 1.0
W_REGEX = 0.15

# категории (усиливаем вклад)
W_CAT = 2.6
CAT_KEYS = ("исследовател", "модератор", "эксперт", "мастер")

# канонические подпункты для категорий (п.5.x)
CAT_CANON = {
    "исследовател": ("5", "4"),
    "модератор":    ("5", "1"),
    "эксперт":      ("5", "3"),
    "мастер":       ("5", "5"),
}
# синонимы/варианты упоминания категорий в вопросах (ловим опечатки и англ/кз)
CATEGORY_SYNONYMS = {
    "исследовател": (
        "исследовател", "исследователь", "исследоват", "иследовател",  # опечатки/корни
        "research", "issled", "issledovatel", "зерттеуш", "зерттеуші", "pedagog-issled"
    ),
    "модератор":    ("модератор", "moderator", "moderat"),
    "эксперт":      ("эксперт", "expert", "ekspert"),
    "мастер":       ("мастер", "master"),
}

# человекочитаемые подписи для вывода
CATEGORY_LABEL = {
    "исследовател": "исследователь",
    "модератор":    "модератор",
    "эксперт":      "эксперт",
    "мастер":       "мастер",
}

# ─────────────────────── Intent + Policy (универсальный слой) ───────────────────────

INTENT_KEYWORDS = {
    "threshold": ("порог", "порогов", "пороговы", "балл", "баллы", "сколько баллов", "озп", "оценка знаний", "тест", "тестирован", "процент", "80%", "минимальный процент", "минимальный балл", "сколько процентов", "сколько баллов", "надо набрать"),
    "procedure": ("как сдать", "как проходит", "этап", "этапы", "заявлен", "подать", "портфолио", "комисси", "обобщен"),
    "exemption_foreign": (
    "болаш", "bolash",
    "nazarbayev", "nazarbayev university",
    "перечень рекомендованных", "перечень организаций"
),

    "exemption_retirement": ("пенсионер", "работающий пенсионер", "пенсионного возраста", "до пенсии", "осталось до пенсии"),

    
}

def _detect_category_key(q: str) -> Optional[str]:
    ql = (q or "").lower().replace("ё","е")
    for key, syns in CATEGORY_SYNONYMS.items():
        if any(s in ql for s in syns):
            return key
    return None

def classify_question(q: str) -> Dict[str, Any]:
    ql = (q or "").lower().replace("ё","е")
    cat = _detect_category_key(ql)
    # приоритет: пенсионеры → порог → льготы зарубеж → категория → процедура → general
    if any(k in ql for k in INTENT_KEYWORDS["exemption_retirement"]):
        return {"intent": "exemption_retirement", "category": None, "confidence": 0.9}
    if any(k in ql for k in INTENT_KEYWORDS["threshold"]):
        return {"intent": "threshold", "category": cat, "confidence": 0.9}
    if any(k in ql for k in INTENT_KEYWORDS["exemption_foreign"]):
        return {"intent": "exemption_foreign", "category": None, "confidence": 0.9}
    if cat:
        return {"intent": "category_requirements", "category": cat, "confidence": 0.85}
    if any(k in ql for k in INTENT_KEYWORDS["procedure"]):
        return {"intent": "procedure", "category": None, "confidence": 0.75}
    return {"intent": "general", "category": None, "confidence": 0.5}


POLICIES = {
    "threshold": {
        "primary": [("39","")],                # всегда цитируем п.39
        "secondary": [("10","")],              # процедура вторым номером
        "max_citations": 3,
        "short_template": "По общим правилам: порог ОЗП — {threshold_percent} (п. 39).{procedure_tail}"
    },
    "category_requirements": {
        "primary": [("5","<cat>")],
        "secondary": [("10",""), ("39","")],
        "max_citations": 3,
        "short_template": "По общим правилам: требования для «педагога-{cat_human}» см. п. 5.{cat_sp}."
    },

    "procedure": {
        "primary": [("10","")],
        "secondary": [("39","")],
        "max_citations": 3,
        "short_template": "По общим правилам: этапы — заявление → портфолио → ОЗП → обобщение → решение комиссии (п.10)."
    },
    "general": {
        "primary": [],
        "secondary": [("10","")],  # либо вообще []
        "max_citations": 3,
        "short_template": "{fallback_short}"
    },
    "exemption_foreign": {
    # Льгота по п.32: присвоение «модератора» без аттестации при наличии степени
    "primary":   [("32","")],
    # Можно дать процедурный п.10 вторично, но БЕЗ п.39
    "secondary": [("10","")],
    "max_citations": 2,
    "short_template": (
        "Если у вас есть учёная степень (канд./д-р наук/PhD) — присваивается «педагог-модератор» без аттестации (п.32); "
        "если степени нет — проходите аттестацию по общим правилам."
    )
    },
    "exemption_retirement": {
        # Освобождение за ≤4 года до пенсии
        "primary":   [("30","")],
        # Порядок для работающих пенсионеров (обычно без ОЗП, комплексное обобщение)
        "secondary": [("57","")],
        "max_citations": 2,
        "short_template": (
            "Зависит: если до пенсии ≤ 4 лет — освобождение от процедуры (п.30); "
            "если уже пенсионер и продолжаете работать — действует особый порядок (п.57)."
    ),
    # Если у тебя поддерживается long_template — будет аккуратный «Подробный вывод»
    "long_template": (
        "Если до достижения пенсионного возраста осталось ≤ 4 лет, вы освобождаетесь от прохождения процедуры аттестации; "
        "имеющаяся категория сохраняется до наступления пенсии (п.30). "
        "Если вы уже достигли пенсионного возраста и продолжаете работать, применяется порядок п.57 "
        "(как правило, освобождение от ОЗП и проведение комплексного обобщения результатов деятельности)."
    )

    },
}
def build_short_answer(policy: dict | None, ctx: dict, fallback_short: str) -> str:
    """
    Если у политики есть свой short_template — используем только его.
    fallback_short применяем только при отсутствии политики или шаблона.
    """
    if not policy:
        return fallback_short

    tmpl = policy.get("short_template")
    if tmpl:
        # Безопасное форматирование, даже если каких-то ключей нет в ctx
        try:
            return tmpl.format(**(ctx or {}))
        except Exception:
            return tmpl  # используем как есть, чтобы не упасть

    return fallback_short

def _policy_primary_pairs(intent: str, category_key: Optional[str]) -> List[Tuple[str,str]]:
    pairs = []
    if intent not in POLICIES: return pairs
    for pn, sp in POLICIES[intent]["primary"]:
        if sp == "<cat>":
            if not category_key: continue
            pn_c, sp_c = CAT_CANON.get(category_key, ("",""))
            if pn_c: pairs.append((pn_c, sp_c))
        else:
            pairs.append((pn, sp))
    return pairs

def _human_cat(category_key: Optional[str]) -> Tuple[str, str]:
    if not category_key: return ("", "")
    human = CATEGORY_LABEL.get(category_key, category_key)
    _, sp = CAT_CANON.get(category_key, ("",""))
    return (human, sp)

# ─────────── Slot-extractors (универсальные извлекатели фактов) ───────────
def extract_threshold_percent_from_p39(punkts: List[Dict[str,Any]]) -> Optional[str]:
    import re
    for p in punkts:
        if str(p.get("punkt_num","")).strip() == "39":
            t = (p.get("text") or "")
            # ищем числа %; если нет — словесное "восемьдесят"
            perc = re.findall(r'(\d{1,3})\s*%', t)
            if perc:
                # берём максимальный процент, чтобы не промахнуться по формулировкам
                try:
                    m = max(int(x) for x in perc)
                    return f"{m}%"
                except Exception:
                    return perc[-1] + "%"
            if "восемьдесят процент" in t.lower(): return "80%"
    return None

def build_procedure_tail_if_p10(punkts: List[Dict[str,Any]]) -> str:
    for p in punkts:
        if str(p.get("punkt_num","")).strip() == "10":
            return " (этапы: заявление → портфолио → ОЗП → обобщение → решение комиссии)"
    return ""
def extract_threshold_percent_from_p39_for_category(punkts: List[Dict[str,Any]], category_key: Optional[str]) -> Optional[str]:
    """
    Пытаемся вытащить % из п.39 именно для указанной категории (мастер/эксперт/…).
    Логика:
      1) Находим п.39.
      2) Если задана категория, ищем в пределах предложений/фрагментов, где встречается корень категории.
      3) Берём ближайший % к таким упоминаниям.
      4) Фоллбек — None (пусть сработает общий экстрактор/дефолт).
    """
    if not category_key:
        return None

    target_syns = CATEGORY_SYNONYMS.get(category_key, ())
    if not target_syns:
        return None

    text39 = None
    for p in punkts:
        if str(p.get("punkt_num","")).strip() == "39":
            text39 = (p.get("text") or "")
            break
    if not text39:
        return None

    tl = text39.lower().replace("ё","е")

    # Режем на предложения и крупные фрагменты
    parts = re.split(r"(?:\n+|[.;]\s+)", tl)
    # собираем кандидаты, где есть синонимы категории
    cand = [s for s in parts if any(syn in s for syn in target_syns)]
    if not cand:
        # иногда категория упомянута только как «педагог-мастер»
        base = tl
        cand = [base] if any(syn in base for syn in target_syns) else []

    # достаём проценты из кандидатов
    get_perc = re.compile(r"(\d{1,3})\s*%")
    found: List[int] = []
    for s in cand:
        found += [int(x) for x in get_perc.findall(s)]
    if found:
        # берём максимальный из привязанных к категории — безопаснее, если в одном фрагменте несколько чисел
        return f"{max(found)}%"

    return None

# ─────────── Policy-aware helpers для цитат и краткого ответа ───────────
def ensure_min_citations_policy(question: str,
                                data: Dict[str,Any],
                                punkts: List[Dict[str,Any]],
                                intent_info: Dict[str,Any]) -> Dict[str,Any]:
    intent = intent_info.get("intent","general")
    category_key = intent_info.get("category")
    primary = _policy_primary_pairs(intent, category_key)
    # формируем must-have список: primary из политики + если есть — вторичные из политики
    want = list(primary)
    for pn, sp in POLICIES.get(intent, {}).get("secondary", []):
        if sp == "<cat>":
            if category_key:
                pn_c, sp_c = CAT_CANON.get(category_key, ("",""))
                if pn_c: want.append((pn_c, sp_c))
        else:
            want.append((pn, sp))

    # оставим существующие цитаты, но перед ними вставим нужные пункты (если они в контексте)
    have = {(str(c.get("punkt_num","")).strip(), str(c.get("subpunkt_num","")).strip())
            for c in (data.get("citations") or [])}
    out: List[Dict[str,str]] = []

    def _exists(pn: str, sp: str="") -> bool:
        for p in punkts:
            if str(p.get("punkt_num","")).strip()==pn and (sp=="" or str(p.get("subpunkt_num","")).strip()==sp):
                return True
        return False

    for pn, sp in want:
        if _exists(pn, sp) and (pn, sp) not in have:
            out.append({"punkt_num": pn, "subpunkt_num": sp, "quote": ""})

    # добавим назад старые (без дублей)
    for c in (data.get("citations") or []):
        pn = str(c.get("punkt_num","")).strip()
        sp = str(c.get("subpunkt_num","")).strip()
        if (pn, sp) not in {(x["punkt_num"], x["subpunkt_num"]) for x in out}:
            out.append({"punkt_num": pn, "subpunkt_num": sp, "quote": c.get("quote","")})

    # ограничим количеством
    maxc = POLICIES.get(intent, {}).get("max_citations", 3)
    data["citations"] = out[:maxc]
    return data

def enforce_short_answer_policy(question: str,
                                data: Dict[str,Any],
                                punkts: List[Dict[str,Any]],
                                intent_info: Dict[str,Any]) -> Dict[str,Any]:
    intent = intent_info.get("intent","general")
    category_key = intent_info.get("category")
    policy = POLICIES.get(intent, POLICIES["general"])
    templ = policy.get("short_template","{fallback_short}")

    human, sp = _human_cat(category_key)
    facts = {
        "cat_human": human,
        "cat_sp": sp,
        "threshold_percent": (
            extract_threshold_percent_from_p39_for_category(punkts, category_key)
            or extract_threshold_percent_from_p39(punkts)
            or "80%"
        ),

        "procedure_tail": build_procedure_tail_if_p10(punkts)
    }

    fallback = (data.get("short_answer") or "По общим правилам.").strip()
    sa = templ.format(fallback_short=fallback, **facts).strip()
    data["short_answer"] = sa[:200]
    return data


def policy_get_must_have_pairs(intent_info: Dict[str,Any]) -> List[Tuple[str,str]]:
    intent = intent_info.get("intent","general")
    category_key = intent_info.get("category")
    pairs = _policy_primary_pairs(intent, category_key)

    # добавляем secondary для ЛЮБОЙ политики
    for pn, sp in POLICIES.get(intent, {}).get("secondary", []):
        if sp == "<cat>":
            if category_key:
                pn_c, sp_c = CAT_CANON.get(category_key, ("",""))
                if pn_c:
                    pairs.append((pn_c, sp_c))
        else:
            pairs.append((pn, sp))
    return pairs





# ключи, по которым считаем, что речь про ОЗП/порог (для п.39)
KW_OZP_TERMS = ("озп", "оценка знаний педагогов", "порог", "тестирован", "80 %", "80%")

# лимиты длины цитат
QUOTE_WIDTH_DEFAULT = int(os.environ.get("QUOTE_WIDTH_DEFAULT", "180"))
QUOTE_WIDTH_LONG    = int(os.environ.get("QUOTE_WIDTH_LONG", "600"))


def kw_category_boost(question: str, doc_text: str) -> float:
    ql = (question or "").lower().replace("ё", "е")
    dl = (doc_text or "").lower().replace("ё", "е")
    # есть ли в вопросе хоть один синоним категории
    trig = None
    for key, syns in CATEGORY_SYNONYMS.items():
        if any(s in ql for s in syns):
            trig = key
            break
    if not trig:
        return 0.0
    variants = ("педагог-", "педагог —", "педагог –", "педагог ")
    # и в документе явно упоминается нужная категория
    if trig in dl or any((v + trig) in dl for v in variants):
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
    is_category_q = any(any(s in ql for s in syns) for syns in CATEGORY_SYNONYMS.values())


    # зарубеж/магистратура — вытаскиваем исключения и «Болашак»
    foreign_q = any(k in ql for k in ("магист", "за рубеж", "за границ", "зарубеж", "иностран"))
    if foreign_q and not is_category_q:
        if any(k in dl for k in KW_EXCEPTION_TERMS):
            boost += 0.6
        if any(k in dl for k in KW_FOREIGN_TERMS):
            boost += 0.6

    # общая льгота «без прохождения…» немного бустим,
    # но если вопрос про категорию — наоборот, приглушаем, чтобы не путать
    has_exception_phrase = any(k in dl for k in KW_EXCEPTION_TERMS)
    if has_exception_phrase:
        boost += 0.3
        if is_category_q:
            boost -= 0.6

    # антишум: когда спрашивают про категорию/зарубеж, а текст про оплату/периодичность
    if (is_category_q or foreign_q) and any(k in dl for k in KW_PERIOD_TERMS):
        boost -= 0.4

    # !!! новенькое: при вопросах про категории — бустим документы, где есть ОЗП/порог (п.39 и близко)
    if is_category_q and any(t in dl for t in KW_OZP_TERMS):
        boost += 0.5

    return boost


   


TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# ──────────────────────────── Логгер ────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag-bot")
LAST_RESPONSES: Dict[Tuple[int, int], Dict[str, Any]] = {}
# --- неблокирующие обёртки для sync-функций ---
async def run_blocking(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)

# --- пер-чатовая блокировка, чтобы не было параллельной обработки одного чата ---
LOCKS: Dict[int, asyncio.Lock] = {}

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

def _topk_desc(arr: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, arr.size))
    idx = np.argpartition(-arr, k-1)[:k]
    return idx[np.argsort(-arr[idx])]


def vector_search(q: str, top_k: int = 100) -> List[Tuple[int, float]]:
    vec = embed_query(q)
    b = PUNKT_EMBS
    a_norm = vec / (np.linalg.norm(vec) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sims = (b_norm @ a_norm)
    idxs = _topk_desc(sims, top_k)
    return [(int(i), float(sims[i])) for i in idxs]

def bm25_search(q: str, top_k: int = 100) -> List[Tuple[int, float]]:
    toks = normalize_tokens(tokenize(q))
    if BM25:
        scores = np.array(BM25.get_scores(toks), dtype=np.float64)
    else:
        # простой TF fallback
        scores_list = []
        for doc in DOCS_TOKENS:
            if not doc:
                scores_list.append(0.0); continue
            s = 0
            for t in toks:
                s += doc.count(t)
            scores_list.append(float(s))
        scores = np.array(scores_list, dtype=np.float64)
    idxs = _topk_desc(scores, top_k)
    return [(int(i), float(scores[i])) for i in idxs]


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


# ───────────────────── Маппинги и регэкспы ──────────────────────

KEY_REGEXES = [
    r"\bп\.?\s*(\d{1,3})(?:\.(\d{1,3}))?\b",
    r"\bпункт[а-я]*\s*(\d{1,3})(?:\.(\d{1,3}))?\b",
    r"\bподпункт[а-я]*\s*(\d{1,3})\.(\d{1,3})\b",
    r"\bпп\.\s*(\d{1,3})\.(\d{1,3})\b",
]


def regex_hits(q: str) -> List[int]:
    hits: List[int] = []
    ql = (q or "").lower()
    for rgx in KEY_REGEXES:
        for m in re.finditer(rgx, ql):
            pn = (m.group(1) or "").strip()
            sp = (m.group(2) or "").strip() if (m.lastindex or 0) >= 2 else ""
            for i, p in enumerate(PUNKTS):
                if str(p.get("punkt_num","")).strip() == pn and (not sp or str(p.get("subpunkt_num","")).strip() == sp):
                    hits.append(i)
    # убрать дубликаты, сохранив порядок
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


# ---- небольшой LRU-кэш для эмбеддингов запроса ----
_EMB_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_EMB_CACHE_CAP = 512
_EMB_LOCK = threading.Lock()  # NEW

def _emb_cache_get(key: str) -> Optional[np.ndarray]:
    with _EMB_LOCK:
        if key in _EMB_CACHE:
            vec = _EMB_CACHE.pop(key)
            _EMB_CACHE[key] = vec
            return vec
    return None

def _emb_cache_put(key: str, vec: np.ndarray) -> None:
    with _EMB_LOCK:
        if key in _EMB_CACHE:
            _EMB_CACHE.pop(key)
        _EMB_CACHE[key] = vec
        if len(_EMB_CACHE) > _EMB_CACHE_CAP:
            _EMB_CACHE.popitem(last=False)


def embed_query(text: str) -> np.ndarray:  # override
    key = (text or "").strip()
    got = _emb_cache_get(key)
    if got is not None:
        return got
    resp = call_with_retries(
        client.embeddings.create,
        model=EMBEDDING_MODEL,
        input=key,
    )
    vec = np.array(resp.data[0].embedding, dtype=np.float64)
    _emb_cache_put(key, vec)
    return vec

# ---- rag_search с условным HyDE и более узким final_k для категорий ----
def rag_search(q: str, top_k_stage1: int = 120, final_k: int = 45,
               must_have_pairs: Optional[List[Tuple[str,str]]] = None) -> List[Dict[str, Any]]:
    q = normalize_query(q)
    ql = q.lower().replace("ё", "е")

    def _is_cat_q(ql_: str) -> Optional[str]:
        for key, syns in CATEGORY_SYNONYMS.items():
            if any(s in ql_ for s in syns):
                return key
        return None

    cat_key = _is_cat_q(ql)
    is_category_q = cat_key is not None
    if is_category_q:
        final_k = min(final_k, 24)  # было 30, сузим ещё сильнее

    variants = multi_query_rewrites(q)
    dense_agg: Dict[int, float] = {}
    sparse_agg: Dict[int, float] = {}

    # Dense pass 1
    for v in variants:
        for idx, sc in vector_search(v, top_k=top_k_stage1):
            dense_agg[idx] = max(dense_agg.get(idx, 0.0), sc)

    # Условный HyDE: включаем только если сигнал слабый
    best_dense = (max(dense_agg.values()) if dense_agg else 0.0)
    if HYDE and best_dense < 0.30:
        hyde = hyde_passage(q)
        if hyde:
            for idx, sc in vector_search(hyde, top_k=top_k_stage1 // 2):
                dense_agg[idx] = max(dense_agg.get(idx, 0.0), sc)

    # Sparse
    for v in [q] + variants:
        for idx, sc in bm25_search(v, top_k=top_k_stage1):
            sparse_agg[idx] = max(sparse_agg.get(idx, 0.0), sc)

    # Regex/Mapping
    regex_idx = regex_hits(q)
    mapped_idx = mapped_hits(q)

    # z-score нормализация sparse
    if sparse_agg:
        vals = np.array(list(sparse_agg.values()), dtype=np.float64)
        mu = float(vals.mean()); sigma = float(vals.std() + 1e-6)
        for k in list(sparse_agg.keys()):
            sparse_agg[k] = (sparse_agg[k] - mu) / sigma

    # сводный скор
    items: List[Tuple[int, float]] = []
    candidate_ids = set(list(dense_agg.keys()) + list(sparse_agg.keys()) + regex_idx + mapped_idx)
    for idx in candidate_ids:
        txt = PUNKTS[idx].get("text", "")
        total = (
            W_EMB * dense_agg.get(idx, 0.0)
            + W_BM25 * sparse_agg.get(idx, 0.0)
            + W_REGEX * (1.0 if idx in regex_idx else 0.0)
            + W_MAP * (1.0 if idx in mapped_idx else 0.0)
            + W_KW  * kw_boost(q, txt)
            + W_CAT * kw_category_boost(q, txt)
        )
        items.append((idx, total))

    items.sort(key=lambda x: -x[1])
    ranked = [i for i, _ in items]

    # must-have
    must_have: List[int] = []
    must_have.extend(mapped_idx + regex_idx)

    if is_category_q:
        for i, p in enumerate(PUNKTS):
            txt = (p.get("text") or "").lower().replace("ё", "е")
            if cat_key and cat_key in txt:
                must_have.append(i)
        pn, sp = CAT_CANON.get(cat_key, ("", ""))
        if pn:
            canon_ids = [i for i, p in enumerate(PUNKTS)
                         if str(p.get("punkt_num","")).strip()==pn
                         and str(p.get("subpunkt_num","")).strip()==sp]
            must_have = canon_ids + must_have
        for wanted in ("10", "39"):
            add = [i for i, p in enumerate(PUNKTS) if str(p.get("punkt_num","")).strip()==wanted]
            must_have.extend(add)

    
    foreign_generic = any(k in ql for k in ("магист", "зарубеж", "за границ", "иностран"))
    bolashak_explicit = any(k in ql for k in ("болаш", "nazarbayev", "перечень рекомендованных"))
    if bolashak_explicit:
        p32 = [i for i, p in enumerate(PUNKTS) if str(p.get("punkt_num","")).strip()=="32"]
        must_have = p32 + must_have

  # ── ДОБАВИТЬ: внешние must-have по политике ──
    if must_have_pairs:
        for pn, sp in must_have_pairs:
            for i, p in enumerate(PUNKTS):
                if str(p.get("punkt_num","")).strip() == pn and (not sp or str(p.get("subpunkt_num","")).strip() == sp):
                    # вставим в начало must_have, если такого индекса ещё нет
                    if i not in must_have:
                        must_have.insert(0, i)
    top_idx: List[int] = []
    for i in must_have + ranked:
        if i not in top_idx:
            top_idx.append(i)
        if len(top_idx) >= final_k:
            break

    return [PUNKTS[i] for i in top_idx]
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

2) Цитаты берём ТОЛЬКО из переданного контекста. Если точную фразу трудно выделить — процитируй короткий фрагмент (≤180/600 знаков по ситуации).
3) НЕ упоминай периодичность/оплату/ОЗП/кол-во попыток и т.п., если ЭТО НЕ процитировано.
4) Для вопросов про зарубежную магистратуру/иностр. образование, если в контексте есть освобождение (напр., п.32):
   — отрази это в short_answer в формате "Зависит: …; иначе — по общим правилам" и процитируй норму как первую в "citations".
5) Для вопросов про конкретную категорию (модератор/эксперт/исследователь/мастер) — среди "citations" должна быть цитата,
   где категория названа явно, с конкретным подпунктом (например, п. 5.4 для «педагога-исследователя»).

5a) Если вопрос про конкретную категорию — в "reasoned_answer" сделай КОРОТКИЙ маркированный список (2–6 пунктов)
     ключевых компетенций ИСКЛЮЧИТЕЛЬНО из процитированного п.5.x (без домыслов), затем один абзац с процедурой (если процитированы соответствующие пункты).
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

    # детектор категории по синонимам
    def _is_category_q(q: str) -> bool:
        qn = q.replace("ё","е")
        for _, syns in CATEGORY_SYNONYMS.items():
            if any(s in qn for s in syns):
                return True
        return False

    foreign_trigger = any(t in ql for t in ("магист", "за рубеж", "зарубеж", "иностран", "болаш", "nazarbayev"))
    is_category_q = _is_category_q(ql)

    # Если есть льгота (п.32 в цитатах) и вопрос про зарубеж — строгий шаблон
    if foreign_trigger and ("32" in cites or "без прохождения процедуры аттестации" in ctx):
        sa = ("Зависит: если магистратура окончена в зарубежной организации из перечня «Болашақ», "
              "категория «педагог-модератор» присваивается без аттестации; иначе — по общим правилам.")
    else:
        if is_category_q and not sa.lower().startswith(("по общим правилам:", "зависит:")):
            sa = "По общим правилам: " + sa

    # Нормализуем односложные ответы
    if re.fullmatch(r"\s*(да|нет)[\.\!]*\s*", sa, flags=re.I):
        sa = "По общим правилам: " + sa.capitalize()

    # Убираем жёсткие «обязан/обязательно», если в цитатах есть 3/41
    bad = {"3","41"}
    if (cites & bad):
        sa = re.sub(r"\b(обязан|обязательно|должен|необходимо)\b", "требуется по общим нормам", sa, flags=re.I)

    # Если вопрос про категорию и среди цитат есть п.10 — добавим этапы (если влезают в лимит)
    if is_category_q and "10" in cites and "→" not in sa:
        tail = " (этапы: заявление → портфолио → ОЗП → обобщение → решение комиссии)"
        if len(sa) + len(tail) <= 200:
            sa += tail
    # Если категория — добавим ссылку на канонический подпункт, если влезает
    if is_category_q:
        target = None
        for key, syns in CATEGORY_SYNONYMS.items():
            if any(s in ql for s in syns):
                target = key
                break
        if target:
            pn, sp = CAT_CANON.get(target, ("",""))
            tag = f" (см. п. {pn}.{sp})" if pn and sp else ""
            if tag and "см. п." not in sa and len(sa) + len(tag) <= 200:
                sa += tag


    data["short_answer"] = sa[:200]
    return data


def ensure_min_citations(question: str, data: Dict[str, Any], punkts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Гарантирует минимум 2 релевантные цитаты.
    Для категорий — п.5.<канон> → п.10 → п.39.
    Для зарубеж/магистратуры — п.32 → п.10 (если есть в контексте).
    Иначе — первые 2 из выданного контекста (без 3 и 41).
    """
    ql = (question or "").lower().replace("ё", "е")
    cits = [dict(c) for c in (data.get("citations") or [])]

    def _exists_in_context(pn: str, sp: str = "") -> Optional[Dict[str, str]]:
        for p in punkts:
            if str(p.get("punkt_num","")).strip() == pn and (not sp or str(p.get("subpunkt_num","")).strip() == sp):
                return {"punkt_num": pn, "subpunkt_num": sp, "quote": ""}
        return None

    # выкинем 3/41
    cits = [c for c in cits if str(c.get("punkt_num","")).strip() not in {"3","41"}]

    # если уже >=2 цитат — оставим как есть
    if len(cits) >= 2:
        data["citations"] = cits[:3]
        return data

    # определяем, это категория или "зарубеж"
    target = None
    for key, syns in CATEGORY_SYNONYMS.items():
        if any(s in ql for s in syns):
            target = key
            break

    need: List[Dict[str, str]] = []

    if target:
        pn, sp = CAT_CANON.get(target, ("",""))
        if pn:
            hit = _exists_in_context(pn, sp) or _exists_in_context(pn, "")
            if hit: need.append(hit)
        for pn in ("10","39"):
            hit = _exists_in_context(pn, "")
            if hit: need.append(hit)
    elif any(k in ql for k in ("магист","зарубеж","за границ","иностран","болаш","bolash","nazarbayev")):
        hit32 = _exists_in_context("32","")
        if hit32: need.append(hit32)
        hit10 = _exists_in_context("10","")
        if hit10: need.append(hit10)

    # если всё ещё пусто — возьмём первые 2 пункта из контекста (кроме 3/41)
    if not need:
        seen = set()
        for p in punkts:
            pn = str(p.get("punkt_num","")).strip()
            sp = str(p.get("subpunkt_num","")).strip()
            if pn in {"3","41"}: 
                continue
            key = (pn, sp)
            if key in seen:
                continue
            need.append({"punkt_num": pn, "subpunkt_num": sp, "quote": ""})
            seen.add(key)
            if len(need) >= 2:
                break

    # склеим need + уже существующие (сохраняя порядок и убирая дубли)
    out: List[Dict[str, str]] = []
    seen_keys = set()
    for c in need + cits:
        pn = str(c.get("punkt_num","")).strip()
        sp = str(c.get("subpunkt_num","")).strip()
        key = (pn, sp)
        if pn and key not in seen_keys:
            out.append({"punkt_num": pn, "subpunkt_num": sp, "quote": c.get("quote","")})
            seen_keys.add(key)

    data["citations"] = out[:3]
    return data




# ───────────────────── Пост-процесс и рендер ────────────────────

def _collapse_repeats(text: str) -> str:
    """
    Удаляет подряд идущие одинаковые строки, сжимает повторяющиеся фрагменты
    и вычищает «мусорные» строки вида '1.2.' / '2.3.4.' / пустые маркеры.
    """
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    out = []
    prev = None
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        # выкидываем строки, состоящие только из нумерации (1.2., 3.4.5. и т.п.)
        if re.fullmatch(r"\d+(?:\.\d+){0,4}\.?", s):
            continue
        if s == prev:
            continue
        out.append(s)
        prev = s
    s = "\n".join(out)
    # защита от многократных повторов одинаковой фразы в одной строке
    s = re.sub(r"(Оценивание методов[^\.]*\.)\s*(\1\s*)+", r"\1 ", s, flags=re.I)
    # мягко убираем одиночные «вкрапления» нумерации внутри строки
    s = re.sub(r"(?<=\s)\d+(?:\.\d+){1,4}\.?(?=\s|$)", "", s)
    return re.sub(r"[ \t]+", " ", s).strip()


def validate_citations(citations: List[Dict[str, Any]], punkts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Нормализует цитаты:
    - оставляет только те, что реально есть в выданном контексте;
    - выкидывает п.3 и п.41 как «доказательство»;
    - сохраняет переносы строк;
    - обрезает текст по лимиту (длиннее для п.5.x);
    - удаляет повторы и «мусорные» дубликаты;
    - лимит до 8.
    """
    SKIP_AS_EVIDENCE = {"3", "41"}

    allowed_keys = {(str(p.get("punkt_num","")).strip(), str(p.get("subpunkt_num","")).strip()) for p in punkts}
    by_key = {
        (str(p.get("punkt_num","")).strip(), str(p.get("subpunkt_num","")).strip()): (p.get("text") or "")
        for p in punkts
    }

    out: List[Dict[str, Any]] = []
    for c in (citations or []):
        pn = str(c.get("punkt_num","")).strip()
        sp = str(c.get("subpunkt_num","")).strip()
        if not pn or pn in SKIP_AS_EVIDENCE or (pn, sp) not in allowed_keys:
            continue

        base = by_key.get((pn, sp), "")
        if not base.strip():
            continue

        limit = QUOTE_WIDTH_LONG if pn == "5" else QUOTE_WIDTH_DEFAULT
        qt = (c.get("quote") or "")
        base_clean = _collapse_repeats(base)
        if qt and qt.lower().replace("ё","е") in base.lower().replace("ё","е"):
            qt_clean = _collapse_repeats(qt)
            good = qt_clean if len(qt_clean) <= limit else (qt_clean[:limit] + "…")
        else:
            good = base_clean if len(base_clean) <= limit else (base_clean[:limit] + "…")

        out.append({"punkt_num": pn, "subpunkt_num": sp, "quote": good})

    # дедуп по (pn, sp, quote)
    seen: set[Tuple[str,str,str]] = set(); uniq: List[Dict[str, Any]] = []
    for c in out:
        key = (c["punkt_num"], c["subpunkt_num"], c["quote"])
        if key not in seen:
            seen.add(key); uniq.append(c)
    return uniq[:8]

async def handle_debug_webhook(request: web.Request) -> web.Response:
    try:
        r = requests.get(f"{TELEGRAM_API}/getWebhookInfo", timeout=15)
        return web.json_response(r.json(), status=r.status_code)
    except Exception as e:
        logger.exception("getWebhookInfo failed")
        return web.json_response({"error": str(e)}, status=500)


def _ensure_category_citation(question: str,
                              citations: List[Dict[str, Any]],
                              punkts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ql = (question or "").lower().replace("ё", "е")

    # ищем целевую категорию по синонимам
    target = None
    for key, syns in CATEGORY_SYNONYMS.items():
        if any(s in ql for s in syns):
            target = key
            break
    if not target:
        return citations

    # уже есть цитата с нужной категорией?
    by_key_txt = {
        (str(p.get("punkt_num","")).strip(), str(p.get("subpunkt_num","")).strip()):
            (p.get("text") or "").lower().replace("ё", "е")
        for p in punkts
    }
    def _mentions_cat(c: Dict[str, Any]) -> bool:
        key = (str(c.get("punkt_num","")).strip(), str(c.get("subpunkt_num","")).strip())
        return target in by_key_txt.get(key, "")

    if any(_mentions_cat(c) for c in (citations or [])):
        return citations

    # если нет — добавляем канонический п.5.x первым, если он в контексте
    pn, sp = CAT_CANON.get(target, ("", ""))
    for p in punkts:
        if str(p.get("punkt_num","")).strip()==pn and str(p.get("subpunkt_num","")).strip()==sp:
            return [{"punkt_num": pn, "subpunkt_num": sp, "quote": ""}] + (citations or [])
    # иначе — добавим первый пункт из контекста, где встречается корень категории
    for p in punkts:
        txt = (p.get("text") or "").lower().replace("ё", "е")
        if target in txt:
            return [{"punkt_num": str(p.get("punkt_num","")).strip(),
                     "subpunkt_num": str(p.get("subpunkt_num","")).strip(),
                     "quote": ""}] + (citations or [])
    return citations

# ── Фильтрация цитат по ключевым словам из вопроса ──
def filter_citations_by_question(
    question: str,
    citations: List[Dict[str, Any]],
    punkts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Релевантность и порядок:
    — Зарубеж/магистратура: п.32 первым, максимум 1–2 цитаты.
    — Категория: порядок — п.5.целевой → п.10 → п.39, до 3 цитат.
      Если п.10/п.39 есть в контексте, но отсутствуют в цитатах — ДОБАВЛЯЕМ сюда.
    — 3 и 41 выкидываем.
    — Иначе: до 3 цитат.
    """
    import re

    ql = (question or "").lower().replace("ё", "е")
    if not citations:
        return citations

    by_key_full = {
        (str(p.get("punkt_num","")).strip(), str(p.get("subpunkt_num","")).strip()): (p.get("text") or "")
        for p in punkts
    }
    by_key = {k: v.lower().replace("ё","е") for k, v in by_key_full.items()}

    # helper — обрезка вокруг ключевых слов с аккуратными границами
    def _crop_around(text_full: str, keys: Tuple[str, ...], width: int = QUOTE_WIDTH_DEFAULT) -> str:
        tf = re.sub(r"[ \t]+", " ", text_full or "").strip()
        tf = tf.replace("\u00A0", " ")
        tl = tf.lower().replace("ё", "е")
        pos = -1
        for k in keys:
            i = tl.find(k)
            if i != -1 and (pos == -1 or i < pos):
                pos = i
        if pos == -1:
            return tf[:width] + ("…" if len(tf) > width else "")
        pad = width // 2
        start = max(0, pos - pad); end = min(len(tf), pos + pad)
        while start > 0 and tf[start] not in " .,;:!?()[]{}«»": start -= 1
        while end < len(tf) and tf[end - 1] not in " .,;:!?()[]{}«»": end += 1
        snippet = tf[start:end].strip()
        # вот эти две строки — новое:
        snippet = snippet.lstrip(" ;,.:—-–•")
        snippet = snippet.rstrip(" ,;:")
        return snippet[:width] + ("…" if len(snippet) > width else "")
  

    # remove 3/41
    clean = [c for c in citations if str(c.get("punkt_num","")).strip() not in {"3","41"}]
    if not clean:
        clean = citations[:]

    # foreign?
    if any(k in ql for k in ("магист","зарубеж","за границ","иностран","болаш","bolash","nazarbayev")):
        p32 = [c for c in clean if str(c.get("punkt_num","")).strip() == "32"]
        rest = [c for c in clean if str(c.get("punkt_num","")).strip() != "32"]
        pref_terms = ("зарубеж","болаш","nazarbayev","без прохождения")
        def _rel(c):
            key = (str(c.get("punkt_num","")).strip(), str(c.get("subpunkt_num","")).strip())
            return any(t in by_key.get(key, "") for t in pref_terms)
        ordered = p32 + sorted(rest, key=lambda c: (not _rel(c)))
        out = (ordered[:2] or clean[:2])
        for c in out:
            key = (str(c.get("punkt_num","")).strip(), str(c.get("subpunkt_num","")).strip())
            base_full = by_key_full.get(key, "")
            if base_full:
                c["quote"] = _crop_around(base_full, pref_terms, width=QUOTE_WIDTH_DEFAULT)
        return out
        # retirement?
    if any(k in ql for k in ("пенсион", "работающий пенсионер", "до пенсии")):
        p30 = [c for c in clean if str(c.get("punkt_num","")).strip() == "30"]
        p57 = [c for c in clean if str(c.get("punkt_num","")).strip() == "57"]
        rest = [c for c in clean if c not in (p30 + p57)]
        out = (p30 + p57 + rest)[:2]  # обычно 1–2 цитаты достаточно
        for c in out:
            key = (str(c.get("punkt_num","")).strip(), str(c.get("subpunkt_num","")).strip())
            base_full = by_key_full.get(key, "")
            if base_full:
                # короткий фрагмент вокруг ключевых слов
                c["quote"] = _crop_around(base_full, ("пенсион", "освобожда", "обобщен", "озп"), width=QUOTE_WIDTH_DEFAULT)
        return out

    # category?
    target = None
    for key, syns in CATEGORY_SYNONYMS.items():
        if any(s in ql for s in syns):
            target = key
            break

    if target:
        # ensure p.10 and p.39 exist in context
        def _exists_p(pn: str) -> Optional[Tuple[str, str, str]]:
            for (kpn, ksp), t in by_key_full.items():
                if kpn == pn:
                    return (pn, ksp, t)
            return None

        # сортировка: п.5.канон -> прочие п.5 -> 10 -> 39 -> остальное
        canon_sp = CAT_CANON.get(target, ("",""))[1]
        def _ord(c):
            pn = str(c.get("punkt_num","")).strip()
            sp = str(c.get("subpunkt_num","")).strip()
            if pn == "5" and sp == canon_sp: return (0, 0)
            if pn == "5": return (1, 0)
            if pn == "10": return (2, 0)
            if pn == "39": return (3, 0)
            return (4, 0)

        clean.sort(key=_ord)
        out = clean[:3]

        # если 10/39 отсутствуют, но есть в контексте — добавим
        have = {(str(c.get("punkt_num","")).strip(), str(c.get("subpunkt_num","")).strip()) for c in out}
        for pn in ("10","39"):
            if not any(h[0] == pn for h in have):
                hit = _exists_p(pn)
                if hit:
                    pn_, sp_, txt_ = hit
                    out.append({"punkt_num": pn_, "subpunkt_num": sp_, "quote": ""})

        # финальная нарезка по типам пунктов
        for c in out:
            key = (str(c.get("punkt_num","")).strip(), str(c.get("subpunkt_num","")).strip())
            base_full = by_key_full.get(key, "")
            if not base_full:
                continue

            pn = key[0]
            if pn == "5":
                width = QUOTE_WIDTH_LONG
                c["quote"] = _collapse_repeats(_crop_around(base_full, (target,), width=width))
            elif pn == "10":
                keys10 = ("заявлен", "портфолио", "озп", "обобщен", "комисси")
                c["quote"] = _collapse_repeats(_crop_around(base_full, keys10, width=QUOTE_WIDTH_DEFAULT))
            elif pn == "39":
                perc = (extract_threshold_percent_from_p39_for_category(punkts, target)
                        or extract_threshold_percent_from_p39(punkts))
                if perc:
                    # perc вида '90%'; добавим варианты '90 %' и '90\u00A0%'
                    num = re.search(r"\d{1,3}", perc).group(0)
                    keys39 = (perc, f"{num} %", f"{num}\u00A0%")
                else:
                    keys39 = ()
                c["quote"] = _collapse_repeats(_crop_around(base_full, keys39, width=QUOTE_WIDTH_DEFAULT))
            else:
                c["quote"] = _collapse_repeats(_crop_around(base_full, tuple(), width=QUOTE_WIDTH_DEFAULT))

        return out[:3]

    # default: просто до 3 цитат и аккуратная нарезка
    out = clean[:3]
    for c in out:
        key = (str(c.get("punkt_num","")).strip(), str(c.get("subpunkt_num","")).strip())
        base_full = by_key_full.get(key, "")
        if base_full:
            c["quote"] = _collapse_repeats(_crop_around(base_full, tuple(), width=QUOTE_WIDTH_DEFAULT))
    return out
def enforce_reasoned_answer(question: str, data: Dict[str, Any], punkts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Для вопросов про категорию:
    — короткий маркированный список (2–6 пунктов) компетенций из канонического п.5.x;
    — затем одна строка про процедуру (п.10), если он есть в контексте/цитатах.
    """
    ql = (question or "").lower().replace("ё", "е")

    # определить категорию
    target = None
    for key, syns in CATEGORY_SYNONYMS.items():
        if any(s in ql for s in syns):
            target = key; break
    if not target:
        return data

    # человекочитаемая метка
    human = CATEGORY_LABEL.get(target, target)

    # найти канонический п.5.x
    pn, sp = CAT_CANON.get(target, ("",""))
    base_txt = ""
    for p in punkts:
        if str(p.get("punkt_num","")).strip()==pn and str(p.get("subpunkt_num","")).strip()==sp:
            base_txt = p.get("text","") or ""
            break
    if not base_txt:
        return data

    def _bullets_from(text: str, max_items: int = 6) -> List[str]:
        t = _collapse_repeats(text)
        parts = re.split(r"(?:\n+|•|—|\u2014|;|\.\s+|\d+\)|\d+\.)", t)
        parts = [re.sub(r"[ \t]+"," ", s).strip(" -—•.;") for s in parts]
        parts = [s for s in parts if 20 <= len(s) <= 240]
        uniq, seen = [], set()
        for s in parts:
            key = s.lower()[:80]
            if key not in seen:
                seen.add(key); uniq.append(s)
            if len(uniq) >= max_items:
                break
        return uniq[:max_items]

    bullets = _bullets_from(base_txt)
    if not bullets:
        return data

    cites = {str(c.get("punkt_num","")).strip() for c in (data.get("citations") or [])}
    have_p10 = ("10" in cites) or any(str(p.get("punkt_num","")).strip()=="10" for p in punkts)

    lines = [f"Ключевые компетенции («педагог-{human}», п.{pn}.{sp}):"]
    lines += [f"— {b}" for b in bullets[:6]]
    if have_p10:
        lines.append("Процедура: заявление → портфолио → ОЗП → обобщение → решение комиссии (п.10).")

    current = (data.get("reasoned_answer") or "").strip()
    if len(current) < 60 or "— " not in current:
        data["reasoned_answer"] = "\n".join(lines)
    else:
        data["reasoned_answer"] = current + "\n\n" + "\n".join(lines)
    return data


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
        payload["reply_markup"] = reply_markup

    r = requests.post(url, json=payload, timeout=15)
    try:
        if r.ok:
            return r.json().get("result", {}).get("message_id")
        # fallback: убрать HTML при parse error
        if "can't parse entities" in r.text.lower():
            payload.pop("parse_mode", None)
            r2 = requests.post(url, json=payload, timeout=15)
            if r2.ok:
                return r2.json().get("result", {}).get("message_id")
        logger.error("sendMessage failed: %s %s", r.status_code, r.text)
    except Exception:
        logger.exception("sendMessage decode error")
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
        payload["reply_markup"] = reply_markup

    r = requests.post(url, json=payload, timeout=15)
    if r.ok:
        return
    if "can't parse entities" in r.text.lower():
        payload.pop("parse_mode", None)
        r2 = requests.post(url, json=payload, timeout=15)
        if r2.ok:
            return
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
    # не шумим «связанными пунктами» для порога/льгот/пенсионеров
    intent = classify_question(question).get("intent", "general")


    # Авто-related: если в наборе пунктов контекста присутствуют 39 или 63.*, а в citations их нет — добавим в related
    have_cit = {(str(c.get("punkt_num","")).strip(), str(c.get("subpunkt_num","")).strip()) for c in citations}
    related = data.get("related", []) or []
    if intent not in INTENTS_HIDE_RELATED:
        # Авто-related только если это уместно
        have_cit = {(str(c.get("punkt_num","")).strip(), str(c.get("subpunkt_num","")).strip()) for c in citations}
        def _exists(pn: str, sp: str = "") -> bool:
            for p in punkts:
                if str(p.get("punkt_num","")).strip()==pn and (sp=="" or str(p.get("subpunkt_num","")).strip()==sp):
                    return True
            return False
        def _push(pn: str, sp: str = "") -> None:
            if (pn, sp) not in have_cit:
                related.append({"punkt_num": pn, "subpunkt_num": sp})
        if _exists("39"): _push("39","")
        for p in punkts:
            if str(p.get("punkt_num","")).strip()=="63":
                _push("63", str(p.get("subpunkt_num","")).strip())
                break
    data["related"] = related


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
            # ВАЖНО: НИКАКИХ <br>. Telegram HTML их не понимает.
            qt = html.escape(c.get("quote", ""))  # переносы строк оставляем как \n
            lines.append(f"— <i>{head}</i>:\n{qt}")
    if related:
        lines.append("<b>Связанные пункты:</b>")
        for r in related[:12]:
            pn = html.escape(str(r.get("punkt_num", "")))
            sp = html.escape(str(r.get("subpunkt_num", "")))
            head = f"п. {pn}{('.' + sp) if sp else ''}".strip()
            lines.append(f"• {head}")
    return "\n".join(lines).strip()
# Интенты, где «Связанные пункты» лучше скрыть, чтобы не шуметь
INTENTS_HIDE_RELATED = {"threshold", "exemption_foreign", "exemption_retirement"}

def render_related(intent: str, related_items: list[str]) -> str:
    if intent in INTENTS_HIDE_RELATED:
        return ""
    if not related_items:
        return ""
    return "Связанные пункты:\n" + "\n".join(f"• {x}" for x in related_items)

def tg_set_webhook(full_url: str, secret: Optional[str]) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook"
    payload = {"url": full_url, "allowed_updates": ["message", "callback_query"], "drop_pending_updates": True}
    if secret:
        payload["secret_token"] = secret
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
    ok = (PUNKT_EMBS.ndim == 2 and PUNKT_EMBS.shape[0] == len(PUNKTS))
    payload = {
        "ok": ok,
        "punkts": len(PUNKTS),
        "emb_shape": list(PUNKT_EMBS.shape),
        "bm25": bool(BM25 is not None),
        "cache_size": len(_EMB_CACHE) if "_EMB_CACHE" in globals() else 0,
    }
    return web.json_response(payload)


async def handle_webhook(request: web.Request) -> web.Response:
    if TELEGRAM_WEBHOOK_SECRET:
        recv_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if recv_secret != TELEGRAM_WEBHOOK_SECRET:
            logger.warning("Webhook rejected: secret mismatch or missing header. "
                           "Most likely your reverse proxy drops 'X-Telegram-Bot-Api-Secret-Token'. "
                           "Either pass the header through or unset TELEGRAM_WEBHOOK_SECRET.")
            return web.Response(status=403, text="forbidden")

    data = await request.json()
    logger.info("Update keys: %s", list(data.keys()))

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

        await run_blocking(tg_answer_callback_query, cq.get("id"))

        if not stash:
            await run_blocking(tg_edit_message_text, chat_id, message_id, "Данные недоступны. Отправьте вопрос заново.")
            return web.Response(text="ok")

        if action == "show_detailed":
            detailed = stash["detailed_html"]
            if len(detailed) <= 4000:
                await run_blocking(tg_edit_message_text, chat_id, message_id, detailed, reply_markup=kb_show_short())
            else:
                notice = stash["short_html"] + "\n\n<i>Подробный ответ отправлен отдельными сообщениями ниже.</i>"
                await run_blocking(tg_edit_message_text, chat_id, message_id, notice, reply_markup=kb_show_short())
                for chunk in split_for_telegram(detailed, 4000):
                    await run_blocking(tg_send_message, chat_id, chunk)
        elif action == "show_short":
            await run_blocking(tg_edit_message_text, chat_id, message_id, stash["short_html"], reply_markup=kb_show_detailed())

        return web.Response(text="ok")

    # 2) Обычное сообщение
    message = data.get("message", {}) if isinstance(data, dict) else {}
    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = message.get("text", "") or ""

    if not chat_id:
        return web.Response(text="ok")

    if text.strip().startswith("/start"):
        await run_blocking(tg_send_message, chat_id, "Здравствуйте! Задайте вопрос по Правилам аттестации педагогов — я отвечу с цитатами.")
        return web.Response(text="ok")

    if not text.strip():
        await run_blocking(tg_send_message, chat_id, "Пожалуйста, пришлите текстовый вопрос.")
        return web.Response(text="ok")

    # Пер-чатовая блокировка
    lock = LOCKS.setdefault(int(chat_id), asyncio.Lock())
    if lock.locked():
        await run_blocking(tg_send_message, chat_id, "Уже обрабатываю ваш предыдущий вопрос, одну секунду 🙌")
        return web.Response(text="ok")
    async with lock:
        try:
            # 0) Интент
            intent_info = classify_question(text)

            # 1) must-have по политике в retrieve
            policy_pairs = policy_get_must_have_pairs(intent_info)
            punkts = await run_blocking(rag_search, text, must_have_pairs=policy_pairs)

            # 2) LLM
            data_struct = await run_blocking(ask_llm, text, punkts)

            # 3) Политика: минимум цитат и краткий ответ
            data_struct = ensure_min_citations_policy(text, data_struct, punkts, intent_info)
            data_struct = enforce_short_answer_policy(text, data_struct, punkts, intent_info)

            # 4) Буллеты только для вопросов про категорию
            if intent_info.get("intent") == "category_requirements":
                data_struct = enforce_reasoned_answer(text, data_struct, punkts)

            # HTML
            short_html = render_short_html(text, data_struct)
            detailed_html = render_detailed_html(text, data_struct, punkts)

            # отправляем short; если длиннее — разбиваем
            if len(short_html) <= 4000:
                msg_id = await run_blocking(tg_send_message, chat_id, short_html, reply_markup=kb_show_detailed())
            else:
                parts = split_for_telegram(short_html, 4000)
                msg_id = await run_blocking(tg_send_message, chat_id, parts[0], reply_markup=kb_show_detailed())
                for extra in parts[1:]:
                    await run_blocking(tg_send_message, chat_id, extra)

            if msg_id:
                key = (int(chat_id), int(msg_id))
                LAST_RESPONSES[key] = {
                    "message_id": int(msg_id),
                    "short_html": short_html,
                    "detailed_html": detailed_html,
                }

            if len(LAST_RESPONSES) > 200:
                FIRST = next(iter(LAST_RESPONSES))
                if FIRST != key:
                    LAST_RESPONSES.pop(FIRST, None)

            await run_blocking(log_to_sheet_safe, chat_id, text, data_struct.get("short_answer", ""))

        except Exception:
            logger.exception("Processing failed")
            await run_blocking(tg_send_message, chat_id, "Произошла ошибка при обработке запроса. Попробуйте позже.")

    return web.Response(text="ok")

   

# ─────────────────────────── main() ─────────────────────────────
# в main():


async def on_startup(app: web.Application):
    full_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"

    tg_set_webhook(full_url, TELEGRAM_WEBHOOK_SECRET)
    logger.info("Service started on port %s", PORT)


def main():
    app = web.Application()

    app.router.add_get("/health", handle_health)
    app.router.add_post(WEBHOOK_PATH, handle_webhook)
    app.router.add_get("/debug/webhook", handle_debug_webhook)
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
