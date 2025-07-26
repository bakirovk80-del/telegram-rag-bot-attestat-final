# ────────────────  config_maps.py  ────────────────
"""
Карта «ключевое слово → координаты пункта Правил».
Поле subpunkt_num == ""  → берётся ВЕСЬ пункт.
"""

# 1. Компетенции 5.1‑5.5
COMPETENCY_MAP = {
    "стажер":        {"punkt_num": "5", "subpunkt_num": "1"},
    "стажёр":        {"punkt_num": "5", "subpunkt_num": "1"},
    "модератор":     {"punkt_num": "5", "subpunkt_num": "2"},
    "эксперт":       {"punkt_num": "5", "subpunkt_num": "3"},
    "исследователь": {"punkt_num": "5", "subpunkt_num": "4"},
    "мастер":        {"punkt_num": "5", "subpunkt_num": "5"},
}

# 2. Досрочное присвоение (63/1‑3)
FASTTRACK_MAP = {
    "досрочн": {"punkt_num": "63", "subpunkt_num": "1"},
    "63/1":    {"punkt_num": "63", "subpunkt_num": "1"},
    "63/2":    {"punkt_num": "63", "subpunkt_num": "2"},
    "63/3":    {"punkt_num": "63", "subpunkt_num": "3"},
}

# 3. Документы (64/2‑3)
DOCS_MAP = {
    "документ":  {"punkt_num": "64", "subpunkt_num": "2"},
    "портфолио": {"punkt_num": "64", "subpunkt_num": "2"},
    "64/2":      {"punkt_num": "64", "subpunkt_num": "2"},
    "64/3":      {"punkt_num": "64", "subpunkt_num": "3"},
}

# 4. Порядок (64/1)
PROCEDURE_MAP = {
    "порядок": {"punkt_num": "64", "subpunkt_num": "1"},
    "этап":    {"punkt_num": "64", "subpunkt_num": "1"},
}

# 5. ОЗП / испытания
EXAM_MAP = {
    "озп":     {"punkt_num": "45", "subpunkt_num": ""},
    "испытан": {"punkt_num": "45", "subpunkt_num": ""},
}

# 6. Апелляция и ответственность
APPEAL_RESP_MAP = {
    "апелляц":    {"punkt_num": "48", "subpunkt_num": ""},
    "жалоб":      {"punkt_num": "48", "subpunkt_num": ""},
    "ответствен": {"punkt_num": "46", "subpunkt_num": ""},
}

# 7. Льготы / освобождения (п. 29)
EXEMPTION_MAP = {
    "пенси":         {"punkt_num": "29", "subpunkt_num": ""},
    "пенсион":       {"punkt_num": "29", "subpunkt_num": ""},
    "беремен":       {"punkt_num": "29", "subpunkt_num": ""},
    "нетрудоспособ": {"punkt_num": "29", "subpunkt_num": ""},
    "воинск":        {"punkt_num": "29", "subpunkt_num": ""},
    "отпуск":        {"punkt_num": "29", "subpunkt_num": ""},
}

# 8. Льготы «Болашақ» (п. 32) и иностранные сертификаты
BOLASHAQ_MAP = {
    "болаш":      {"punkt_num": "32", "subpunkt_num": ""},
    "bolash":     {"punkt_num": "32", "subpunkt_num": ""},
    "nazarbayev": {"punkt_num": "32", "subpunkt_num": ""},
    "celtа":      {"punkt_num": "32", "subpunkt_num": ""},
    "delta":      {"punkt_num": "32", "subpunkt_num": ""},
    "phd":        {"punkt_num": "32", "subpunkt_num": ""},
}

# 9. Единая карта
UNIVERSAL_MAP: dict[str, dict[str, str]] = {}
for _m in (
        COMPETENCY_MAP,
        FASTTRACK_MAP,
        DOCS_MAP,
        PROCEDURE_MAP,
        EXAM_MAP,
        APPEAL_RESP_MAP,
        EXEMPTION_MAP,
        BOLASHAQ_MAP,
):
    UNIVERSAL_MAP.update(_m)
# ───────────────────────────────────────────────
