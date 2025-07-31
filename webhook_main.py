import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters,
    CallbackQueryHandler
)
from aiohttp import web
import asyncio

# Твой весь импорт + твои функции (log_to_sheet, build_human_friendly, и т.д.)
# Просто скопируй из bot.py (или как у тебя сейчас называется) — ничего менять не надо.
# ────────────────  bot.py  ────────────────

# --- Создаём файл credentials, если его нет ---
if not os.path.exists("service_account.json"):
    creds = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if creds:
        with open("service_account.json", "w") as f:
            f.write(creds)
    else:
        raise RuntimeError("Переменная GOOGLE_CREDENTIALS_JSON не задана!")

import inspect, json, re, time
from functools import lru_cache
from itertools import groupby
import gspread
from google.oauth2.service_account import Credentials
import datetime
def log_to_sheet(user_id, username, message, bot_answer, timestamp):
    try:
        creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        creds_dict = json.loads(creds_json)
        creds = Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        gc = gspread.authorize(creds)
        sh = gc.open_by_key('11dG_R527d6toFdcgxtyyykxYjVZzodg05TLtQidDCHo')
        ws = sh.sheet1
        ws.append_row([str(user_id), username, message, bot_answer, timestamp])
    except Exception as e:
        print("Ошибка логирования в Google Sheets:", e)

try:
    from unidecode import unidecode
except ModuleNotFoundError:
    def unidecode(s: str) -> str:
        return s

import openai
from bs4 import BeautifulSoup
import numpy as np
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters,
)

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = lambda f: inspect.getfullargspec(f)[:4]

from config_maps import UNIVERSAL_MAP
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("rag-telegram-bot")
# user_id : (question, punkts, human_friendly, official_answer)
LAST_QA = {}

PUNKT_EMBS_PATH = "embeddings.npy"
PUNKT_JSON_PATH = "pravila_detailed_tagged_autofix.json"

PUNKT_EMBS = np.load(PUNKT_EMBS_PATH)
logger.info("Загружено эмбеддингов: %s", PUNKT_EMBS.shape)

def _sanitize(punkts):
    for p in punkts:
        for k in ("chapter","paragraph","punkt_num","subpunkt_num","list_item"):
            p[k] = str(p.get(k,"") or "")
        p["text"] = p["text"].replace("\\n","\n")
    return punkts

with open(PUNKT_JSON_PATH, encoding="utf-8") as f:
    PUNKTS = _sanitize(json.load(f))
logger.info("Загружено пунктов: %d", len(PUNKTS))

TYPE_INDEX = {}
for p in PUNKTS:
    st = p.get("soft_type")
    if st:
        TYPE_INDEX.setdefault(st, []).append(p)

def clean_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup.find_all(True):
        if tag.name not in {"b","i","u","s","code","pre","small"}:
            tag.unwrap()
        else:
            tag.attrs = {}
    out = str(soup).replace("<br/>","\n").replace("<br>","\n")
    out = re.sub(r"<(\w+)>\s*</\1>", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()

def fix_unclosed_tags(html: str) -> str:
    for tag in ("b","i","u","s","code","pre","small"):
        diff = html.count(f"<{tag}>") - html.count(f"</{tag}>")
        if diff>0:
            html += f"</{tag}>" * diff
    return html

try:
    from spellchecker import SpellChecker
    SPELL = SpellChecker(language="ru")
    def autocorrect(txt: str) -> str:
        return " ".join(SPELL.correction(w) or w for w in txt.split())
except:
    autocorrect = lambda txt: txt

def blocks_commission(q: str):
    return [p for p in PUNKTS if "Пункт 20" in p["text"]][:3]

def blocks_appeal(q: str):
    if re.search(r"(апелляц|жалоб)", q, re.I):
        return [p for p in PUNKTS if p["punkt_num"] in {"17","18"}][:5]
    return []

SOFT_RULES = {
    "violation": re.compile(r"\b(наруш|акт наруш|аннулир|аннулировать|аннулируется|аннулирован|аннулировано|ответственность|понижен|недостоверн|фальсификац|отклон|жалоб|апелляц|комисси)\b", re.I),
    "notice": re.compile(r"\b(уведомлени|отказ|решение|комисси|протокол|приказ)\b", re.I),
    "form": re.compile(r"\b(форма|заявлени|перечень|приложен|портал|egov|электронное правительство|веб-портал|платформ)\b", re.I),
}

UNIVERSAL_VIOLATION_PATTERN = re.compile(
    r"(наруш|акт|ответственн|аннулир|аннулировать|аннулируется|аннулирован|аннулировано|отказ|отстран|дисквалификац|дисциплинарн|фальсификац|недостоверн|лишен|приостанов|запрет|взыскан|несоответств|неправомерн|санкц|недопуск|отклон|жалоб|апелляц|претенз|обжал|рассмотрен|комисси|пересмотр|спорн|несоглас|отзыв|повторн|протокол|приказ|уведомлени|заключен|лист|решени|форма|приложен|заявлени|выписка|портал|egov|электронное правительство|веб-портал|платформ)",
    re.I
)

SOFT_TYPES_KEYWORDS = {
    "portal": r"(портал|egov|электронное правительство)",
    "defer": r"(отсрочк|приостанов|продлен|перенос|сохранени|продление|освобожд|стажировк|беременн|уход за ребен|воинская служб|болезн|нетрудоспособн|пенси)"
}

def soft_blocks(question: str, limit_per_type: int = 8):
    rows = []

    # Основные soft‑type группы (по вашей логике)
    for stype, rx in SOFT_RULES.items():
        if rx.search(question):
            rows.extend(TYPE_INDEX.get(stype, [])[:limit_per_type])

    # Универсальный паттерн нарушений и спорных ситуаций
    if UNIVERSAL_VIOLATION_PATTERN.search(question):
        for p in PUNKTS:
            if UNIVERSAL_VIOLATION_PATTERN.search(p["text"]):
                if p not in rows:
                    rows.append(p)

    # Справочник критических тем: ключевые слова —> номера пунктов
    CRITICAL_BLOCKS = [
    # Зарубежное образование (магистратура, дипломы, сертификаты)
    (r"(магистратур[а-я]* за рубежом|зарубеж[а-я]* магистратур[а-я]*|иностран[а-я]* диплом|зарубежн[а-я]* вуз|иностран[а-я]* сертификат)", ["32", "5"]),

    # Освобождение от аттестации (пенсия, декрет, болезнь, воинская служба)
    (r"(пенси[а-я]*|декрет[а-я]*|беремен[а-я]*|отпуск по уходу|нетрудоспособн[а-я]*|болезн[а-я]*|воинск[а-я]* служб[а-я]*|освобожден[а-я]*|не проходить аттестаци[а-я]*)", ["29", "30"]),

    # Досрочное присвоение категории
    (r"(досрочн[а-я]*|внеочередн[а-я]*|срочн[а-я]*|ускорен[а-я]*|немедлен[а-я]*|раньше срока)", ["31", "32", "63", "64"]),

    # Апелляции, жалобы, споры
    (r"(апелляц[а-я]*|жалоб[а-я]*|обжал[а-я]*|претенз[а-я]*|оспариван[а-я]*|пересмотр[а-я]*|несоглас[а-я]*)", ["17", "18", "20"]),

    # Нарушения, аннулирование, ответственность
    (r"(наруш[а-я]*|ответственн[а-я]*|аннулир[а-я]*|фальсификац[а-я]*|недостоверн[а-я]*|отказ[а-я]*|отстран[а-я]*|дисциплинар[а-я]*|санкц[а-я]*)", ["10", "11", "12", "20"]),

    # Документы, формы, портал egov
    (r"(форм[а-я]*|заявлени[а-я]*|портал|egov|электронн[а-я]* правительств[а-я]*|веб-портал|приложен[а-я]*|документ[а-я]*|регистрац[а-я]*)", ["13", "15", "16"]),

    # Протоколы, акты, приказы, заседания
    (r"(протокол[а-я]*|акт[а-я]*|приказ[а-я]*|выписк[а-я]*|заседани[а-я]*|решени[а-я]* комисси[а-я]*)", ["10", "11", "12", "13", "20", "24", "25", "26"]),

    # Категории стажеров и первичная аттестация
    (r"(стажер[а-я]*|стажёр[а-я]*|первичн[а-я]* аттестаци[а-я]*|впервые|начинающ[а-я]*)", ["5", "6", "23", "24"]),

    # Присвоение и подтверждение категории
    (r"(присвоен[а-я]*|подтвержден[а-я]*|исключени[а-я]*|отозван[а-я]*|замен[а-я]*|утвержден[а-я]*)", ["6", "7", "10", "24", "25", "26"]),
]


    for pattern, punkt_nums in CRITICAL_BLOCKS:
        if re.search(pattern, question, re.I):
            for num in punkt_nums:
                for p in PUNKTS:
                    if p["punkt_num"] == num and p not in rows:
                        rows.append(p)

    # SPECIAL_PUNKTS для точечных универсальных случаев (осталось из вашей логики)
    SPECIAL_PUNKTS = {
        "15": r"(исследоват|заявк|заявлен|подать|портал|egov|электронное правительство)",
        "29": r"(отсрочк|приостанов|продлен|перенос|сохранени|продление|освобожд|стажировк|беременн|уход за ребен|воинская служб|болезн|нетрудоспособн|пенси)",
        "30": r"(отсрочк|освобожд|пенси|возраст|освобождение|не проходить|не обязан)",
    }
    for punkt_num, regex in SPECIAL_PUNKTS.items():
        if re.search(regex, question, re.I):
            for p in PUNKTS:
                if p["punkt_num"] == punkt_num and p not in rows:
                    rows.insert(0, p)

    return rows


def unique(seq):
    seen, out = set(), []
    for p in seq:
        key = (p['chapter'],p['paragraph'],p['punkt_num'],p['subpunkt_num'])
        if key not in seen:
            seen.add(key); out.append(p)
    return out

def merge_bullets(rows):
    key = lambda x:(x['chapter'],x['paragraph'],x['punkt_num'],x['subpunkt_num'])
    merged = []
    for _, grp in groupby(sorted(rows, key=key), key):
        grp = list(grp)
        bullets = [g for g in grp
                   if g["text"].lstrip().startswith(("–","—","-"))
                   or g.get("list_item") in {"True","1","true"}]
        if len(bullets)>=2:
            merged.append({
                **{k:grp[0][k] for k in ('chapter','paragraph','punkt_num','subpunkt_num')},
                "text": "\n".join(g["text"] for g in grp)
            })
        else:
            merged.extend(grp)
    return merged

def _drop_headers(rows):
    return [p for p in rows if not p["text"].strip().endswith(":")]

def _boost_by_category(rows, question):
    if "эксперт" not in question.lower():
        return rows
    boost, other = [], []
    for p in rows:
        cat = p.get("category") or []
        if isinstance(cat, str): cat = [cat]
        (boost if any("эксперт" in c.lower() for c in cat) else other).append(p)
    return boost + other

def vector_search(question: str, top_k: int = 40):
    logger.info("➡️ vector_search: %s", question)
    emb = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=question
    ).data[0].embedding
    qv = np.asarray(emb, dtype=np.float32)
    sims = (PUNKT_EMBS @ qv) / (
        np.linalg.norm(PUNKT_EMBS, axis=1) * np.linalg.norm(qv) + 1e-8
    )
    idx = np.argsort(-sims)[:top_k]
    return _boost_by_category([PUNKTS[i] for i in idx], question)

def bm25_fallback(question: str, k: int = 10):
    base = autocorrect(question.lower())
    toks = [t for t in re.findall(r"\w+", base) if len(t) > 3]
    scored = []
    for p in PUNKTS:
        scr = sum(tok in p["text"].lower() for tok in toks)
        if scr:
            scored.append((scr, p))
    return [p for _, p in sorted(scored, key=lambda x:-x[0])[:k]]
def add_all_violation_punkts(rows, question):
    """
    Добавляет в rows все пункты, где явно упоминаются нарушения/санкции/ответственность,
    если в вопросе есть соответствующие слова.
    """
    VIOLATION_RX = re.compile(
        r"(наруш|акт|аннулир|ответственн|санкц|отказ|отстран|дисквалификац|фальсификац|дисциплинарн|наказан|обнаружен|отклон|недостоверн|лишен|понижен|неправомерн)",
        re.I
    )
    if VIOLATION_RX.search(question):
        for p in PUNKTS:
            if VIOLATION_RX.search(p["text"]):
                if p not in rows:
                    rows.append(p)
    return rows

@lru_cache(maxsize=256)
def rag_search(question: str, k: int = 60):
    mapped = []
    q_lower = question.lower()
    for kword, v in UNIVERSAL_MAP.items():
        if kword in q_lower:
            mapped += [p for p in PUNKTS if p["punkt_num"] == v["punkt_num"]
                       and (v["subpunkt_num"] == "" or p["subpunkt_num"] == v["subpunkt_num"])]

    # Эмбеддинг-поиск (через OpenAI, если вопрос релевантен)
    try:
        emb = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=question
        ).data[0].embedding
        qv = np.asarray(emb, dtype=np.float32)
        sims = (PUNKT_EMBS @ qv) / (np.linalg.norm(PUNKT_EMBS, axis=1) * np.linalg.norm(qv) + 1e-8)
        idx = np.argsort(-sims)[:40]
        semant = [PUNKTS[i] for i in idx]
    except Exception:
        semant = []

    # Regex/keyword-поиск
    keywords = set(re.findall(r'\w+', q_lower))
    regexed = []
    for word in keywords:
        if len(word) >= 4:
            regexed += [p for p in PUNKTS if word in p["text"].lower()]
    # Trigger-based (универсальные важные пункты)
    trigger_nums = []
    if re.search(r"наруш|ответственн|аннулир|акт|фальсификац|недостоверн", q_lower):
        trigger_nums += ["45", "46", "47", "62"]
    if re.search(r"пенси|возраст|пенсион|освобожд|уход за ребен|беременн|декрет", q_lower):
        trigger_nums += ["29", "30"]
    if re.search(r"болаша|nazarbayev|phd|степень|исследоват|магистр", q_lower):
        trigger_nums += ["32"]
    if re.search(r"апелляц|жалоб|обжал|спорн|пересмотр|комисси", q_lower):
        trigger_nums += ["17", "18", "20"]
    triggers = [p for p in PUNKTS if p["punkt_num"] in trigger_nums]

    # Собираем всё в rows
    rows = []
    seen = set()
    for l in (mapped, semant, regexed, triggers):
        for p in l:
            key = (p["punkt_num"], p["subpunkt_num"])
            if key not in seen:
                rows.append(p)
                seen.add(key)

    # --- ДОБАВЛЯЕМ soft_blocks для максимального покрытия ---
    rows += soft_blocks(question)

    # Fallback если всё пусто
    if not rows:
        rows = bm25_fallback(question, 10)

    # Оставляем ваши постпроцессоры!
    rows = unique(rows)
    rows = merge_bullets(rows)
    rows = _drop_headers(rows)
    rows = add_all_violation_punkts(rows, question)
    rows = unique(rows)
    print("RAG SELECTED:", [(p["punkt_num"], p["text"][:70]) for p in rows[:k]])
    return rows[:k]


def build_prompt(q, punkts):
    context = "\n\n".join(
        f"{i+1}. [{p['punkt_num']}{('/'+p['subpunkt_num']) if p['subpunkt_num'] else ''}]\n{p['text']}"
        for i, p in enumerate(punkts)
    )

    addendum = ""

    # 1. Подача заявлений/портал/категории
    if re.search(r"(исследоват|заявк|заявлен|подать|категор|портал|egov|электронное правительство)", q, re.I):
        addendum += (
            "\n\nЕсли среди приведённых фрагментов есть пункт про подачу заявки на категорию или через веб-портал электронного правительства, "
            "в разделе <b>1. Резюме:</b> процитируй дословно про веб-портал (например, «egov.kz») с указанием номера пункта."
        )

    # 2. Возраст/пенсия/до скольки лет
    if re.search(r"(до скольк|возраст|пенси|лет|years|старше|пенсионн|возрастн|пенсионер)", q, re.I):
        addendum += (
            "\n\nЕсли среди приведённых фрагментов есть пункты про освобождение от аттестации по возрасту или пенсии, "
            "или про порядок аттестации после выхода на пенсию, процитируй их с номерами пунктов в <b>1. Резюме:</b>. "
            "Ясно укажи, до какого возраста обязательна аттестация и как она проводится после пенсии."
        )

    # 3. Отсрочка/приостановка/освобождение/перенос/болезнь/декрет и т.д.
    if re.search(r"(отсрочк|приостанов|продлен|перенос|сохранени|продление|освобожд|исключен|перерыв|прекращен|пауза|не проход|не обяз|выход на пенси|беременн|уход за ребен|стажировк|воинская служб|болезн|нетрудоспособн)", q, re.I):
        addendum += (
            "\n\nЕсли среди найденных есть пункты об отсрочке, освобождении, приостановке или сохранении квалификационной категории — процитируй их дословно в <b>1. Резюме:</b> с указанием пунктов. "
            "Детально опиши условия, порядок и документы для каждого такого случая."
        )

    # 4. Первичная аттестация/стажёр/первая категория/впервые
    if re.search(r"(стажер|стажёр|первичн|первый раз|впервые|начал|первая|только начал|по истечении года|после года|стаж работы)", q, re.I):
        addendum += (
            "\n\nЕсли среди фрагментов есть порядок присвоения категории «педагог» после статуса «педагог-стажер», процитируй, что по истечении года подается заявление на категорию «педагог» без процедуры аттестации (с номером пункта). "
            "Если есть сроки первой формальной аттестации — процитируй их в <b>1. Резюме:</b> и приведи пошагово процедуру."
        )

    # 5. Апелляция/жалоба/обжалование/спорные ситуации
    if re.search(r"(апелляц|жалоб|обжал|претенз|спорн|несоглас|оспор|пересмотр|обратиться|комисси|рассмотрен)", q, re.I):
        addendum += (
            "\n\nЕсли среди фрагментов есть пункты о жалобе, апелляции или спорных ситуациях — процитируй дословно куда, когда и как подается жалоба или апелляция (с точными формулировками и пунктами). "
            "В разделах «Порядок» и «Документы» пошагово опиши процедуру и сроки, если такие предусмотрены."
        )

    # 6. Нарушения/дисциплина/санкции/ответственность/отказ
    if re.search(r"(наруш|акт наруш|аннулир|ответственн|отклон|дисквалификац|фальсификац|недостоверн|отказ|отстран|санкц|дисциплинарн|лишен|понижен|наказан|неправомерн|обнаружен)", q, re.I):
        addendum += (
            "\n\nЕсли среди фрагментов есть информация о нарушениях, актах нарушения, последствиях или санкциях, процитируй эти положения с указанием пункта. "
            "В разделах «Особые случаи» и «Порядок» опиши последствия и официальные шаги."
        )

    # 7. Документы/формы/уведомления/протоколы
    if re.search(r"(документ|форма|приложен|заявлени|перечень|уведомлени|решени|протокол|приказ|лист|выписка)", q, re.I):
        addendum += (
            "\n\nЕсли среди найденных фрагментов есть формы заявлений, уведомлений, протоколов или приказов, перечисли их все с точными названиями, приложениями и номерами пунктов в разделе <b>3. Документы, которые заполняются:</b>."
        )

    # 8. Внеочередная/досрочная/ускоренная аттестация
    if re.search(r"(досрочн|внеочередн|срочн|ускорен|немедленн|неотложн)", q, re.I):
        addendum += (
            "\n\nЕсли среди фрагментов есть информация о внеочередной или досрочной аттестации — в разделе <b>1. Резюме:</b> процитируй официальную формулировку с номером пункта. "
            "В разделах «Порядок» и «Документы» опиши всю процедуру и основания."
        )

    # 9. Прочие смешанные случаи (универсальная защита)
    if re.search(r"(оценк|стаж|категор|комисси|обобщ|результат|процедур|решени|приказ|рассмотрен|повышен|прохожд|присвоен|отстранен|назначен|назначени|освобожден|отозван|заменен|поручен|утвержден)", q, re.I):
        addendum += (
            "\n\nЕсли среди приведённых фрагментов есть любые положения о процедуре, назначении, результатах рассмотрения, комиссий, итоговых решениях или присвоении категорий — обязательно процитируй эти положения в соответствующих разделах с указанием пунктов."
        )
        # Новый блок: Если вопрос сложный, с несколькими основаниями (пенсия+декрет+внеочередная и пр.)
    if re.search(r"(внеочередн|досрочн|срочн|повторн|пенси|декрет|уход за ребен|беременн|комисси)", q, re.I):
        addendum += (
            "\n\nЕсли в вопросе или найденных фрагментах затрагиваются несколько разных оснований (например, пенсия и декрет, декрет и внеочередная аттестация, несколько повторных процедур), подробно опиши требования и процедуру для каждого случая отдельно. " 
            "Если часть информации отсутствует — явно напиши об этом в соответствующем разделе."
        )
    # 10. Универсальный фолбэк — если вопрос затрагивает несколько аспектов (например, пенсия и стаж, жалоба и дисциплина и т.д.)
    addendum += (
        "\n\nЕсли в вопросе или найденных фрагментах затрагиваются сразу несколько разных тем (например, пенсия и стаж, стажёр и досрочная аттестация, нарушение и жалоба), обязательно перечисли и процитируй каждый аспект и все связанные основания с точными формулировками и пунктами — по структуре ответа."
    )

    system = (
        "Ты — эксперт-консультант по Правилам аттестации педагогов Республики Казахстан.\n"
        "Отвечай ТОЛЬКО на основе предоставленных фрагментов официального текста.\n\n"
        "Стиль и требования к ответу:\n"
        "– Ответ строго официальный, без разговорных выражений или упрощений.\n"
        "– Не используй собственные трактовки, пересказы и пояснения.\n"
        "– Категорически запрещено использовать выражения вроде «проще говоря», «это значит», «иначе говоря».\n"
        "– Если ответа на вопрос в предоставленных фрагментах нет, явно напиши: «В документе отсутствует информация, напрямую отвечающая на данный вопрос.»\n\n"
        "Структура ответа:\n"
        "<b>1. Резюме:</b> одна фраза — строго официальная цитата или итоговая формулировка с указанием пункта. "
        "Если есть случаи отсрочки, освобождения, приостановки, особые правила, первичная аттестация, жалобы или нарушения — процитируй их с номерами пунктов.\n\n"
        "<b>2. Основные условия/критерии:</b>\n"
        "– Используй дословные цитаты официальных пунктов и подпунктов (ссылка обязательна: п. X подп. Y).\n"
        "– Строго без сокращений и обобщений.\n\n"
        "<b>3. Документы, которые заполняются:</b>\n"
        "– Дословно перечисли документы с точными названиями приложений (например, «Акт нарушения правил и условий проведения оценки знаний педагога (Приложение 10 к Правилам)»). "
        "ВСЕГДА указывай номер приложения и пункт, если он есть. Если информация отсутствует, напиши это явно.\n\n"
        "<b>4. Порядок процедуры / этапы:</b>\n"
        "– Только официальные шаги с указанием соответствующих пунктов.\n"
        "– Не пересказывай и не добавляй шаги самостоятельно.\n\n"
        "<b>5. Особые случаи / последствия (если есть):</b>\n"
        "– Дословные цитаты о последствиях нарушений, отказах, ответственности с обязательным указанием пункта. "
        "Включи все предусмотренные документом особые случаи (например, болезнь, беременность, пенсия, стажировка, апелляция, дисциплинарные нарушения, первичная аттестация и др.), с дословной цитатой и номером пункта.\n\n"
        "<b>6. Связанные пункты:</b>\n"
        "– Включай пункты, которые НЕ были процитированы в основном ответе, но имеют отношение к вопросу.\n"
        "– Перечисляй кратко и чётко, указывая номер пункта и краткое официальное содержание.\n"
        "– Не дублируй уже процитированные пункты.\n\n"
        "Правила оформления:\n"
        "– Используй HTML-теги <b>...</b> для заголовков разделов.\n"
        "– Чётко разделяй абзацы.\n"
        "– Каждый элемент списков начинай с «– ».\n\n"
        "Дополнительно:\n"
        "– Если пользователь задаёт уточняющий вопрос, повторно приводи точные цитаты.\n"
        "– Категорически запрещено использовать внешние источники или информацию вне предложенных фрагментов."
        + addendum
    )

    user = f"Вопрос: «{q}»\n\nФрагменты:\n{context}\n\nОтвет:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def ask_llm(q, punkts):
    for attempt in range(3):
        try:
            r = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=build_prompt(q, punkts),
                temperature=0.2,
                max_tokens=2000,
            )
            return clean_html(r.choices[0].message.content.strip())
        except openai.RateLimitError:
            time.sleep(2**attempt)
        except openai.OpenAIError as e:
            logger.error("OpenAIError: %s", e)
            break
    sample = "\n".join(f"• {p['text']}" for p in punkts[:5])
    return f"Не удалось получить ответ от LLM.\n\nВозможные фрагменты:\n{sample}"

def postprocess_answer(q, punkts, answer):
    # Универсальный постпроцессор для ответа:
    # – Удаляет дубли пунктов/приложений между секциями.
    # – Добавляет все пропущенные приложения в "Документы".
    # – Если раздел отсутствует/пустой — вставляет 'Нет информации...'
    # – Сортирует внутри секций по возрастанию номера.

    all_punkt_nums = set(p["punkt_num"] for p in punkts if p["punkt_num"])
    all_apps = set(re.findall(r"Приложение\s?\d+", "\n".join([p["text"] for p in punkts])))

    cited_nums = set(re.findall(r"п\. ?(\d+[\w\.]*)", answer))
    cited_apps = set(re.findall(r"Приложение\s?\d+", answer))

    missed_apps = all_apps - cited_apps
    if missed_apps:
        docs_block = "\n" + "\n".join([f"– {app} (см. текст Правил)" for app in sorted(missed_apps)]) + "\n"
        answer = re.sub(
            r"(<b>3\. Документы, которые заполняются:</b>[\s\S]*?)(?=<b>|$)",
            lambda m: m.group(1) + docs_block,
            answer,
        )

    for sec in [
        "1. Резюме",
        "2. Основные условия/критерии",
        "3. Документы, которые заполняются",
        "4. Порядок процедуры / этапы",
        "5. Особые случаи / последствия (если есть)",
        "6. Связанные пункты",
    ]:
        if f"<b>{sec}:</b>" not in answer:
            answer += f"\n\n<b>{sec}:</b>\nНет информации по данному разделу."

    # НОВЫЙ БЛОК: Вставляем только релевантные "Связанные пункты"
    related = filter_related_punkts(q, punkts, cited_nums)
    if related:
        related_block = "<b>6. Связанные пункты:</b>\n" + "\n".join(
            f"– Пункт {p['punkt_num']}: {p['text'].split('.')[0][:80]}..." for p in related
        )
    else:
        related_block = "<b>6. Связанные пункты:</b>\nНет информации по данному разделу."

    # Заменяем в ответе соответствующую секцию (или добавляем, если нет)
    if re.search(r"<b>6\. Связанные пункты:</b>[\s\S]*?(?=<b>|$)", answer):
        answer = re.sub(
            r"<b>6\. Связанные пункты:</b>[\s\S]*?(?=<b>|$)",
            related_block + "\n",
            answer,
        )
    else:
        answer += "\n\n" + related_block

    # Сортировка документов
    def sort_docs(m):
        docs = m.group(1).strip().splitlines()
        docs = sorted(set(docs), key=lambda x: re.sub(r'\D', '', x))
        return "<b>3. Документы, которые заполняются:</b>\n" + "\n".join(docs) + "\n"

    answer = re.sub(
        r"<b>3\. Документы, которые заполняются:</b>\n([\s\S]*?)(?=<b>|$)",
        sort_docs,
        answer,
    )

    bad_phrases = [
        "по закону", "по документу", "по правилам",
        "обычно", "можно", "как правило", "чаще всего",
        "означает", "в случае", "если"
    ]
    for phrase in bad_phrases:
        if phrase in answer.lower():
            logger.warning(f"❗ Найдена недопустимая фраза в ответе: {phrase}")

    return answer.strip()
   

def check_completeness(q, punkts, answer):
    cited_nums = set(re.findall(r"п\. ?(\d+)", answer))
    missed = []
    for p in punkts:
        if p["punkt_num"] and p["punkt_num"] not in cited_nums:
            missed.append(p["punkt_num"])
    if missed:
        logger.warning(f"❗ Не отражены пункты: {missed} в ответе на: {q}")

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Здравствуйте! Задайте вопрос по Правилам аттестации педагогов.",
        parse_mode="HTML"
    )

def build_human_friendly(q, punkts, official_answer):
    # Универсальный поиск всех оснований для освобождения/льгот и без дублей
    exception_phrases = {}
    for p in punkts:
        txt = p["text"].lower()
        num = f"(п. {p['punkt_num']})" if p.get('punkt_num') else ""
        # Выпускники Болашак и зарубежных вузов
        if ("болашақ" in txt or "bolashak" in txt or "зарубеж" in txt or "иностранн" in txt):
            if ("освобожд" in txt or "без прохождения процедуры" in txt or "без аттестации" in txt):
                key = "Выпускники зарубежных организаций, входящих в список рекомендованных по программе «Болашақ», освобождаются от аттестации"
                exception_phrases[key] = num
        # Ученые степени
        if (("кандидат наук" in txt or "доктор наук" in txt or "phd" in txt)
            and ("без прохождения процедуры" in txt or "без аттестации" in txt or "освобожд" in txt)):
            key = "Педагогам, имеющим степень кандидата/доктора наук или доктора PhD, квалификационная категория присваивается без процедуры аттестации"
            exception_phrases[key] = num
        # Пенсионный возраст
        if (("пенси" in txt or "пенсион" in txt or "возраст" in txt)
            and "освобожд" in txt):
            key = "Педагоги, которым до пенсии по возрасту остается не более четырех лет, освобождаются от аттестации"
            exception_phrases[key] = num
        # Инвалидность
        if (("инвалид" in txt or "ограниченн" in txt)
            and "освобожд" in txt):
            key = "Педагоги с инвалидностью освобождаются от прохождения аттестации"
            exception_phrases[key] = num
        # Декрет, беременность, уход за ребенком
        if (("декрет" in txt or "беремен" in txt or "уход за ребен" in txt or "по уходу за ребен" in txt)
            and ("освобожд" in txt or "отсроч" in txt or "приостан" in txt)):
            key = "Педагоги, находящиеся в декретном отпуске или в отпуске по уходу за ребенком, могут быть освобождены или получить отсрочку от аттестации"
            exception_phrases[key] = num
        # Воинская служба, стажировка, болезнь
        if (("воинск" in txt or "военная служб" in txt or "стажировк" in txt or "болезн" in txt or "нетрудоспособн" in txt)
            and ("освобожд" in txt or "отсроч" in txt or "приостан" in txt)):
            key = "Педагоги, находящиеся на воинской службе, в стажировке или по болезни, могут быть освобождены или получить отсрочку от аттестации"
            exception_phrases[key] = num
        # Повторная/досрочная/внеочередная категория
        if (("досрочн" in txt or "внеочередн" in txt or "ускорен" in txt or "повторн" in txt)
            and ("присваива" in txt or "процедур" in txt or "категор" in txt)):
            key = "Досрочное или внеочередное присвоение квалификационной категории допускается по установленной процедуре"
            exception_phrases[key] = num

    # Только уникальные по смыслу
    if exception_phrases:
        return " ".join(sorted({f"{k} {v}".strip() for k, v in exception_phrases.items()}))

    # Платность аттестации (спецкейс)
    if re.search(r"платн|оплат|стоимост|платно|стоимость", q, re.I):
        if "п. 41" in official_answer or "41" in official_answer:
            return (
                "Аттестация педагогических работников проводится бесплатно один раз в год, "
                "повторное прохождение — на платной основе согласно утвержденной сумме (п. 41). "
                "Досрочная аттестация — бесплатно. Пробное тестирование — на платной основе (п. 41)."
            )
    if re.search(r"(магистратур[а-я]* за рубежом|зарубежн[а-я]* вуз|иностранн[а-я]* диплом)", q, re.I):
    return ("Выпускники зарубежных организаций, входящих в список рекомендованных "
            "по программе «Болашақ», освобождаются от аттестации (п. 32). "
            "В остальных случаях аттестация обязательна (п. 5).")
elif re.search(r"(пенси[а-я]*|декрет[а-я]*|беремен[а-я]*|болезн[а-я]*|воинск[а-я]* служб[а-я]*)", q, re.I):
    return ("Педагоги в указанных случаях освобождаются от аттестации с сохранением "
            "действующей категории (пп. 29-30).")
elif re.search(r"(досрочн[а-я]*|внеочередн[а-я]* присвоен[а-я]*)", q, re.I):
    return ("Досрочное или внеочередное присвоение категории возможно "
            "по установленной процедуре (пп. 63-64).")

    # Если ничего не нашли — лаконично на основе официального
    prompt = (
        f"Вопрос: {q}\n"
        "Сформулируй краткий ответ исключительно на основании текста официального ответа ниже. "
        "В ответе используй только точные формулировки из официального ответа и обязательно ссылайся на номер пункта в формате (п. X). "
        "Запрещено делать любые пересказы, упрощения, рассуждения, пояснения или интерпретации. "
        "Не используй разговорные слова, бытовые обороты, такие как «можно», «обычно», «есть возможность» и др. "
        "Только юридически точные, краткие и однозначные фразы. "
        "Если официальный ответ не содержит нужной информации, ответь: «В документе отсутствует информация, напрямую отвечающая на данный вопрос.»\n\n"
        f"Официальный ответ:\n{official_answer}\n\n"
        "Краткий формальный ответ:"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
            max_tokens=180,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Ошибка генерации human-friendly ответа:", e)
        return "(Краткое объяснение не сгенерировано)"



def filter_related_punkts(q, all_punkts, cited_nums):
    """
    Возвращает только релевантные связанные пункты по тематике вопроса, 
    исключая уже процитированные и стоп-пункты (об утверждении, отмене и пр.).
    Максимум 7 уникальных по теме.
    """
    q_lower = q.lower()
    STOP_PUNKTS = {"1", "2"}  # можно дополнять
    themes = {
        "пенси": r"пенси|пенсион|возраст|освобожд[её]н|выход на пенси|старше|возрастн|пенсионер|до скольки лет|лет",
        "декрет": r"декрет|беремен|уход за ребен|отпуск по уход|материнств|родов|нетрудоспособн|болезн",
        "болашак": r"болаша|nazarbayev|phd|магистр|зарубежн|иностранн|вуз|университет|master|зарубежом|заграниц|иностр",
        "стажер": r"стажер|стажёр|первичн|первый раз|впервые|начал|только начал|стаж работы|после года|по истечении года|испытательн|начинающ",
        "наруш": r"наруш|санкц|акт|ответственн|аннулир|дисциплинар|отказ|дисквалификац|наказан|отстран|фальсификац|недостоверн|лишен|понижен|обнаружен|неправомерн|претенз",
        "жалоб": r"жалоб|апелляц|обжал|претенз|спорн|оспор|несоглас|пересмотр|рассмотрен|обратиться|претензия|спор",
        "портал": r"портал|egov|электронное правительство|веб-портал|подача|заявк|заявлени|регистрац|личный кабинет|онлайн|платформа|форма",
        "категор": r"категор|присвоен|подтвержден|присвоени|подтверждени|квалификац|присвоить|подтвердить|повышени[ея] квалификации|повышение",
        "аттестаци": r"аттестаци|процедур|порядок|этап|срок|прохождени|повторн|внеочередн|досрочн|ускорен|немедленн|неотложн|экзамен",
        "документ": r"документ|форма|приложен|уведомлени|решени|протокол|приказ|лист|выписка|заключени|акт|заявление|справка",
        "отсрочка": r"отсрочк|приостанов|продлен|перенос|сохранени|продление|освобожд|исключен|перерыв|прекращен|пауза|стажировк|воинская служб|болезн",
        "комисси": r"комисси|рассмотрен|назначен|назначени|решени|заседание|рассмотрение комиссии|члены комиссии|эксперт",
        "стажировка": r"стажировк|испытательн|обучени|практика|практикант|интернатур|повышени[ея] квалификации|переподготовк|переподготовка|обучение|повышение квалификации",
        "язык": r"язык|иностранн|английск|немецк|франц|китайск|турецк|арабск",
    }
    matched_themes = [rx for rx in themes.values() if re.search(rx, q_lower, re.I)]
    if not matched_themes:
        return []
    related = []
    for p in all_punkts:
        pn = p.get("punkt_num", "")
        if pn in cited_nums or pn in STOP_PUNKTS or not pn or not pn.isdigit():
            continue
        if any(re.search(rx, p["text"], re.I) for rx in matched_themes):
            related.append(p)
        if len(related) >= 7:  # максимум 7 связанных пунктов
            break
    return related




async def handle(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.message.text or ""
    user = update.effective_user

    logger.info("⇢ %s", q)
    punkts = merge_bullets(rag_search(q))[:45]
    logger.info("Нашли пунктов: %d", len(punkts))

    # Получаем оба ответа
    official_answer = ask_llm(q, punkts) if punkts else "Прямого ответа не найдено."
    official_answer = postprocess_answer(q, punkts, official_answer)
    human_friendly = build_human_friendly(q, punkts, official_answer)

    # Сохраняем последний вопрос пользователя (user.id)
    LAST_QA[user.id] = (q, punkts, human_friendly, official_answer)

    # Формируем human-friendly сообщение с кнопкой
    reply_text = (
        f"💡 <b>Краткий ответ:</b>\n{human_friendly}\n\n"
        f"<i>Для детального ответа — нажмите кнопку ниже.</i>"
    )
    keyboard = [
        [InlineKeyboardButton("Показать детальный ответ", callback_data="show_official")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        fix_unclosed_tags(reply_text),
        parse_mode="HTML",
        reply_markup=reply_markup
    )

    # Логируем human-friendly ответ (официальный залогируется после кнопки)
    timestamp = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
    log_to_sheet(
        user_id=user.id if user else "",
        username=user.username if user and user.username else "",
        message=q,
        bot_answer=human_friendly,
        timestamp=timestamp
    )
async def button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user = query.from_user
    await query.answer()

    if query.data == "show_official":
        qa = LAST_QA.get(user.id)
        if qa:
            q, punkts, human_friendly, official_answer = qa
            reply_text = f"<b>Детальный ответ по Правилам:</b>\n{official_answer}"

            await query.message.reply_text(
                fix_unclosed_tags(reply_text),
                parse_mode="HTML"
            )

            # Логируем официальный ответ
            import datetime
            timestamp = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
            log_to_sheet(
                user_id=user.id if user else "",
                username=user.username if user and user.username else "",
                message=f"[Официальный ответ на] {q}",
                bot_answer=official_answer,
                timestamp=timestamp
            )
        else:
            await query.message.reply_text("Официальный ответ не найден.")


# ---- Ниже — основной код для запуска через webhook ----

TOKEN = os.environ.get("TELEGRAM_TOKEN")
PORT = int(os.environ.get("PORT", "8080"))  # Render по умолчанию выставляет PORT=10000 или 8080
WEBHOOK_PATH = f"/{TOKEN}"
WEBHOOK_URL = os.environ.get("WEBHOOK_URL") 


# ----- твой глобальный LAST_QA, а также все функции, которые были раньше -----

# (Оставь свои handle, button, start — без изменений!)

# ---- Запуск бота и aiohttp сервера ----

async def on_startup(app):
    await app['bot'].bot.set_webhook(url=WEBHOOK_URL + WEBHOOK_PATH)
    logger.info("Webhook set to %s", WEBHOOK_URL + WEBHOOK_PATH)

async def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    app.add_handler(CallbackQueryHandler(button))
    logger.info("Бот запущен через webhook.")
    await app.initialize()
    # Создаем aiohttp веб-сервер для Telegram
    web_app = web.Application()
    web_app['bot'] = app

    # endpoint для Telegram webhook
    async def handle_update(request):
        data = await request.json()
        update = Update.de_json(data, app.bot)
        await app.process_update(update)
        return web.Response(text="ok")
    web_app.router.add_post(WEBHOOK_PATH, handle_update)
    web_app.on_startup.append(on_startup)

    # Запускаем сервер на нужном порту
    runner = web.AppRunner(web_app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()

    # Не даем процессу умереть
    while True:
        await asyncio.sleep(3600)
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
