"""Normalização de texto pré-TTS para leitura natural em português brasileiro."""

import re
from datetime import date

_UNITS_M = (
    "", "um", "dois", "três", "quatro", "cinco",
    "seis", "sete", "oito", "nove", "dez",
    "onze", "doze", "treze", "quatorze", "quinze",
    "dezesseis", "dezessete", "dezoito", "dezenove",
)
_UNITS_F = (
    "", "uma", "duas", "três", "quatro", "cinco",
    "seis", "sete", "oito", "nove", "dez",
    "onze", "doze", "treze", "quatorze", "quinze",
    "dezesseis", "dezessete", "dezoito", "dezenove",
)
_TENS = (
    "", "", "vinte", "trinta", "quarenta", "cinquenta",
    "sessenta", "setenta", "oitenta", "noventa",
)
_HUNDREDS_M = (
    "", "cento", "duzentos", "trezentos", "quatrocentos",
    "quinhentos", "seiscentos", "setecentos", "oitocentos", "novecentos",
)
_HUNDREDS_F = (
    "", "cento", "duzentas", "trezentas", "quatrocentas",
    "quinhentas", "seiscentas", "setecentas", "oitocentas", "novecentas",
)
_ORD_UNITS = (
    "", "primeiro", "segundo", "terceiro", "quarto",
    "quinto", "sexto", "sétimo", "oitavo", "nono",
)
_ORD_TENS = (
    "", "décimo", "vigésimo", "trigésimo", "quadragésimo",
    "quinquagésimo", "sexagésimo", "septuagésimo", "octogésimo", "nonagésimo",
)
_ORD_HUNDREDS = (
    "", "centésimo", "ducentésimo", "trecentésimo", "quadringentésimo",
    "quingentésimo", "sexcentésimo", "septingentésimo", "octingentésimo", "nongentésimo",
)
_MONTHS = (
    "", "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
)
_MONTH_DAYS = (0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _under_thousand(n: int, feminine: bool = False) -> str:
    if n == 0:
        return ""
    if n == 100:
        return "cem"
    units = _UNITS_F if feminine else _UNITS_M
    hundreds = _HUNDREDS_F if feminine else _HUNDREDS_M
    parts = []
    h, rem = divmod(n, 100)
    if h:
        parts.append(hundreds[h])
    if rem < 20:
        if rem:
            parts.append(units[rem])
    else:
        t, u = divmod(rem, 10)
        parts.append(_TENS[t])
        if u:
            parts.append(units[u])
    return " e ".join(parts)


def int_to_pt(n: int, feminine: bool = False) -> str:
    """Converte um inteiro não negativo para português por extenso."""
    if n < 0:
        return f"menos {int_to_pt(-n, feminine=feminine)}"
    if n == 0:
        return "zero"
    if n >= 10 ** 12:
        return " ".join(int_to_pt(int(d)) for d in str(n))

    billions, rem = divmod(n, 1_000_000_000)
    millions, rem = divmod(rem, 1_000_000)
    thousands, rest = divmod(rem, 1_000)

    groups: list[tuple[str, int]] = []
    if billions:
        label = "bilhão" if billions == 1 else "bilhões"
        groups.append((f"{int_to_pt(billions)} {label}", billions * 1_000_000_000))
    if millions:
        label = "milhão" if millions == 1 else "milhões"
        groups.append((f"{int_to_pt(millions)} {label}", millions * 1_000_000))
    if thousands:
        prefix = "mil" if thousands == 1 else f"{_under_thousand(thousands, feminine=feminine)} mil"
        groups.append((prefix, thousands * 1_000))
    if rest:
        groups.append((_under_thousand(rest, feminine=feminine), rest))

    if len(groups) == 1:
        return groups[0][0]

    result = groups[0][0]
    for index in range(1, len(groups)):
        text_part, value = groups[index]
        is_last = index == len(groups) - 1
        needs_e = is_last and (value < 100 or value % 100 == 0)
        result += f" e {text_part}" if needs_e else f" {text_part}"
    return result


def _ordinal_to_pt(n: int, feminine: bool = False) -> str:
    if n <= 0 or n >= 1000:
        return int_to_pt(n, feminine=feminine)
    h, rem = divmod(n, 100)
    t, u = divmod(rem, 10)
    words = [w for w in (_ORD_HUNDREDS[h], _ORD_TENS[t], _ORD_UNITS[u]) if w]
    if feminine:
        words = [w[:-1] + "a" for w in words]
    return " ".join(words)


def _decimal_to_pt(int_part: str, frac_part: str) -> str:
    left = int_to_pt(int(int_part.replace(".", "")))
    frac_clean = frac_part.rstrip("0") or "0"
    if frac_clean.startswith("0") and len(frac_clean) > 1:
        zeros = len(frac_clean) - len(frac_clean.lstrip("0"))
        rest = frac_clean.lstrip("0")
        right = " ".join(["zero"] * zeros + ([int_to_pt(int(rest))] if rest else []))
    else:
        right = int_to_pt(int(frac_clean))
    return f"{left} vírgula {right}"


def _replace_currency(match: re.Match) -> str:
    reais = int(match.group(1).replace(".", ""))
    frac = match.group(2)
    centavos = int(frac.ljust(2, "0")[:2]) if frac else 0
    parts = []
    if reais > 0 or centavos == 0:
        unit = "real" if reais == 1 else "reais"
        connector = " de " if reais and reais % 1_000_000 == 0 else " "
        parts.append(f"{int_to_pt(reais)}{connector}{unit}")
    if centavos > 0:
        unit = "centavo" if centavos == 1 else "centavos"
        parts.append(f"{int_to_pt(centavos)} {unit}")
    return " e ".join(parts)


def _replace_percent(match: re.Match) -> str:
    int_part, frac_part = match.group(1), match.group(2)
    if frac_part is not None:
        number = _decimal_to_pt(int_part, frac_part)
    else:
        number = int_to_pt(int(int_part.replace(".", "")))
    return f"{number} por cento"


def _replace_date(match: re.Match) -> str:
    day, month, year = int(match.group(1)), int(match.group(2)), match.group(3)
    try:
        date(int(year) if year is not None else 2000, month, day)
    except ValueError:
        return match.group(0)
    day_text = "primeiro" if day == 1 else int_to_pt(day)
    month_text = _MONTHS[month]
    if year is not None:
        return f"{day_text} de {month_text} de {int_to_pt(int(year))}"
    return f"{day_text} de {month_text}"


def _replace_ordinal(match: re.Match) -> str:
    value = int(match.group(1))
    feminine = match.group(2) == "ª"
    return _ordinal_to_pt(value, feminine=feminine)


def normalize_text_for_tts(text: str) -> str:
    """Normaliza expressões numéricas; preserva identificadores e datas inválidas.

    Um ponto com três dígitos à direita é tratado como milhar. Versões com
    múltiplos pontos e códigos alfanuméricos são preservados por inteiro.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    normalized = re.sub(r"(?m)^[ \t]*(?:---+|[•▪◦])[ \t]*", "", text)
    normalized = re.sub(r"§§[ \t]*", "parágrafos ", normalized)
    normalized = re.sub(r"§[ \t]*", "parágrafo ", normalized)
    normalized = re.sub(r"[ \t]*&[ \t]*", " e ", normalized)
    # Uma única passagem evita reconverter trechos deliberadamente preservados.
    number = r"(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)"
    pattern = (
        r"(?P<code>(?=[\w./-]*\d)[A-Za-z_][\w]*(?:[./-][\w]+)*|\d+(?:\.\d+){2,}|\d+(?:/\d+)+)"
        r"|(?P<money>R\$[ \t]*" + number + r")"
        r"|(?P<ordinal>\d+[ \t]*[ºª])"
        r"|(?P<numeric>-?" + number + r"(?:[ \t]*%)?)"
    )

    def replace(match: re.Match) -> str:
        token = match.group()
        if match.lastgroup == "code":
            if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", token):
                return int_to_pt(int(token.replace(".", "")))
            candidate = re.fullmatch(r"(\d{1,2})/(\d{1,2})(?:/(\d{4}))?", token)
            return _replace_date(candidate) if candidate else token
        if match.lastgroup == "ordinal":
            ordinal = re.fullmatch(r"(\d+)[ \t]*([ºª])", token)
            return _replace_ordinal(ordinal) if 0 < int(ordinal[1]) < 1000 else token
        if match.lastgroup == "money":
            currency = re.fullmatch(r"R\$[ \t]*(\d[\d.]*)(?:,(\d{1,2}))?", token)
            return _replace_currency(currency) if currency else token
        suffix = " por cento" if token.endswith("%") else ""
        value = token.rstrip("% \t")
        sign = "menos " if value.startswith("-") else ""
        value = value.lstrip("-")
        if "," in value:
            integer, fraction = value.split(",")
            result = _decimal_to_pt(integer, fraction)
        elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", value):
            result = int_to_pt(int(value.replace(".", "")))
        elif "." in value:
            result = _decimal_to_pt(*value.split("."))
        else:
            # Valores fora da faixa suportada e identificadores com zeros à
            # esquerda permanecem legíveis sem perder informação.
            if len(value) > 12 or (len(value) > 1 and value.startswith("0")):
                return token
            result = int_to_pt(int(value))
        return sign + result + suffix

    return re.sub(r"(?<!\w)(?:" + pattern + r")(?!\w)", replace, normalized)
