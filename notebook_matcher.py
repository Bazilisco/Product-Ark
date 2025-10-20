# notebook_matcher.py
import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional

# ------------------ Scoring de CPU ------------------
CPU_TIERS = {
    "I3": 1, "I5": 2, "I7": 3, "I9": 4,
    "ULTRA 5": 3, "ULTRA 7": 4, "ULTRA 9": 5,
    "RYZEN 3": 1, "RYZEN 5": 2, "RYZEN 7": 3, "RYZEN 9": 4,
}
APPLE_BASE = {"M1": 2, "M2": 3, "M3": 4, "M4": 5}

def _norm(s):
    if pd.isna(s): return ""
    return str(s).upper()

def parse_cpu(s: str) -> float:
    s = _norm(s)
    if not s:
        return 0.0

    # Apple Silicon Mx + sufixos
    m = re.search(r"\bM([1-5])\b(?:\s*(PRO|MAX|ULTRA))?", s)
    if m:
        base = APPLE_BASE.get(f"M{m.group(1)}", 0)
        suf = (m.group(2) or "").upper()
        bump = {"PRO": 0.5, "MAX": 1.0, "ULTRA": 1.5}.get(suf, 0)
        return base + bump

    # Intel Core Ultra
    m = re.search(r"CORE\s+ULTRA\s+(5|7|9)", s)
    if m:
        tier = f"ULTRA {m.group(1)}"
        base = CPU_TIERS.get(tier, 0)
        gen_bonus = 0.2 if re.search(r"\b18\d", s) else 0.1
        return base + gen_bonus

    # Intel Core Ix
    m = re.search(r"CORE\s*I\s*([3579])", s)
    if m:
        ix = f"I{m.group(1)}"
        base = CPU_TIERS.get(ix, 0)
        g = re.search(r"(GER|GEN|G[EÊ]RA[CÇ][AÃ]O|\b)\s*(\d{1,2})", s)
        gen_bonus = 0.0
        if g:
            try:
                gg = int(g.group(2))
                gen_bonus = min(max((gg - 5), 0), 9) * 0.05
            except:
                pass
        return base + gen_bonus

    # AMD Ryzen
    m = re.search(r"RYZEN\s+([3579])", s)
    if m:
        rx = f"RYZEN {m.group(1)}"
        base = CPU_TIERS.get(rx, 0)
        g = re.search(r"GER\s*(\d{1,2})", s)
        gen_bonus = 0.0
        if g:
            try:
                gg = int(g.group(1))
                gen_bonus = min(max((gg - 2), 0), 9) * 0.05
            except:
                pass
        return base + gen_bonus

    # ARM / mobile (heurística para ordenação relativa)
    if "SNAPDRAGON" in s:
        m = re.search(r"SNAPDRAGON\s*(\d+)", s)
        if m:
            val = int(m.group(1))
            return 1 + (val / 1000)
        return 2.5
    if "DIMENSITY" in s: return 2.8
    if "EXYNOS"   in s: return 2.6
    if re.search(r"\bA1[0-9]|A\d{2}\b", s): return 2.5  # Apple A15, A17, etc.

    return 0.0

# ------------------ Storage (tamanho + tipo) ------------------
STO_PATTERNS = [
    r"\b(SSD|NVME|NV\.?ME|M\.?2|HDD|HD)\s*([\d\.,]+)\s*(TB|GB)\b",
    r"\b([\d\.,]+)\s*(TB|GB)\s*(SSD|NVME|NV\.?ME|M\.?2|HDD|HD)\b",
    r"\b([\d\.,]+)\s*(TB|GB)\s*(STORAGE|ROM)\b",
]

def _parse_num(num_str: str) -> float:
    s = str(num_str).strip().replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(",", "")       # vírgula de milhar
    else:
        s = s.replace(",", ".")      # vírgula decimal
    return float(s)

def _to_gb(num: str, unit: str) -> float:
    val = _parse_num(num)
    return val * 1024.0 if unit == "TB" else val

def parse_storage_with_type(text: str) -> Tuple[float, str, str]:
    """
    Retorna (storage_gb, storage_type, texto_sem_storage).
    Blindagem de 'M2' (CPU Apple) e validação numérica segura.
    """
    s_orig = _norm(text)
    # Blindagem: Apple M1/M2/M3/M4 -> APL1/APL2/... apenas para regex de storage
    s = re.sub(r"\bM([1-5])\b", r"APL\1", s_orig)

    for pat in STO_PATTERNS:
        m = re.search(pat, s)
        if not m:
            continue

        groups = [g for g in m.groups() if g]

        # Normaliza ordem
        if groups[0] in ["SSD", "NVME", "NV.ME", "M.2", "M2", "HDD", "HD"]:
            stype, num, unit = groups[0], groups[1], groups[2]
        elif groups[-1] in ["SSD", "NVME", "NV.ME", "M.2", "M2", "HDD", "HD", "STORAGE", "ROM"]:
            num, unit, stype = groups[0], groups[1], groups[2]
        else:
            continue

        unit = unit.replace(" ", "")
        stype = stype.replace("NV.ME", "NVME").replace("M2", "M.2")
        if stype in ["SSD", "NVME", "M.2"]:
            stype = "SSD"
        elif stype in ["HDD", "HD"]:
            stype = "HDD"
        else:
            stype = "STORAGE"

        # valida que 'num' é realmente numérico
        if not re.fullmatch(r"[\d\.,]+", str(num).strip()):
            continue

        try:
            gb = _to_gb(num, unit)
        except Exception:
            continue

        # remove do texto original
        matched_txt = m.group(0)
        matched_txt_for_original = matched_txt.replace("APL", "M")
        clean = s_orig.replace(matched_txt_for_original, "", 1)
        return gb, stype, clean

    return 0.0, "", s_orig

# ------------------ RAM (evitando confusão com storage) ------------------
RAM_TIPICOS = [2, 3, 4, 6, 8, 12, 16, 18, 24, 32, 48, 64, 96, 128]

def parse_ram_from_text(text_without_storage: str) -> float:
    s = _norm(text_without_storage)
    tuples = re.findall(r"\b(\d{1,3})\s*(GB|G)\b", s)
    candidates = [int(t[0]) for t in tuples] if tuples else []
    if not candidates:
        return 0.0
    tipicos = [x for x in candidates if x in RAM_TIPICOS]
    if tipicos:
        return float(max(tipicos))
    return float(max([x for x in candidates if x <= 128] or [candidates[0]]))

# ------------------ GPU / Tela ------------------
def parse_gpu_dedicated(s1: str, s2: str) -> int:
    text = f"{_norm(s1)} {_norm(s2)}"
    if re.search(r"\b(NVIDIA|GEFORCE|RTX|GTX|RADEON|DEDICADA)\b", text):
        return 1
    return 0

def parse_screen_inches(s: str) -> float:
    """
    Extrai polegadas evitando confundir '18 GB' com '18 pol'.
    Regras:
      - primeiro tenta <número> + (POL|")
      - depois 'TELA' seguido de número
      - por fim, número de 10–19.9 que NÃO esteja antes de 'GB'
    """
    s = _norm(s).replace(",", ".")  # 15,6 -> 15.6

    # 1) 14 POL, 15.6", etc.
    m = re.search(r'\b(\d{1,2}(?:\.\d)?)\s*(?:POL|")\b', s)
    if m:
        val = float(m.group(1))
        if 7.0 <= val <= 19.9:
            return val

    # 2) TELA 14 (opcional POL)
    m = re.search(r'\bTELA\b[^0-9]{0,6}(\d{1,2}(?:\.\d)?)', s)
    if m:
        val = float(m.group(1))
        if 7.0 <= val <= 19.9:
            return val

    # 3) fallback: um número 10–19.9 que NÃO esteja seguido de GB
    m = re.search(r'\b(1\d(?:\.\d)?)\b(?!\s*GB)', s)
    if m:
        val = float(m.group(1))
        if 10.0 <= val <= 19.9:
            return val

    return 0.0

# ------------------ Família ------------------
def family_from_row(row: pd.Series) -> str:
    code = _norm(row.get("codigo"))
    desc = _norm(row.get("descricao"))
    if code.startswith("NB"): return "NOTEBOOK"
    if code.startswith("DT"): return "DESKTOP"
    if code.startswith("TB"): return "TABLET"
    if code.startswith("SM"): return "SMARTPHONE"
    if "TABLET" in desc or "IPAD" in desc: return "TABLET"
    if "SMARTPHONE" in desc or "CELULAR" in desc or "IPHONE" in desc: return "SMARTPHONE"
    if "NOTEBOOK" in desc or "LAPTOP" in desc or "MACBOOK" in desc or "NOTE " in desc: return "NOTEBOOK"
    if "DESKTOP" in desc or "PC" in desc: return "DESKTOP"
    return "OUTROS"

# ------------------ Enriquecimento ------------------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    text = (
        d.get("descricao", "").astype(str) + " " +
        d.get("descricao_subtitulo", "").astype(str) + " " +
        d.get("processador", "").astype(str) + " " +
        d.get("memoria_ram", "").astype(str) + " " +
        d.get("armazenamento", "").astype(str) + " " +
        d.get("placa_de_video", "").astype(str) + " " +
        d.get("tamanho_tela", "").astype(str)
    )

    # CPU
    d["cpu_score"] = text.apply(parse_cpu)

    # Storage (tamanho + tipo) e texto sem storage
    stor = text.apply(parse_storage_with_type)
    stor_df = pd.DataFrame(stor.tolist(), index=d.index)
    d["storage_gb"] = stor_df[0]
    d["storage_type"] = stor_df[1]
    text_wo_storage = stor_df[2]

    # RAM a partir da coluna e fallback no texto sem storage
    ram_from_col = d.get("memoria_ram", "").astype(str).apply(parse_ram_from_text)
    ram_from_text = text_wo_storage.apply(parse_ram_from_text)
    d["ram_gb"] = ram_from_col.mask(ram_from_col == 0, ram_from_text)

    # GPU / Tela (preferir coluna; fallback no texto com parser robusto)
    d["gpu_dedicated"] = [parse_gpu_dedicated(a, b) for a, b in zip(d.get("placa_de_video", ""), text)]
    screen_from_col = d.get("tamanho_tela", "").astype(str).apply(parse_screen_inches)
    screen_from_text = text.apply(parse_screen_inches)
    d["screen_in"] = screen_from_col.mask(screen_from_col == 0, screen_from_text)

    # Estoque
    for c in ["saldo_novo", "saldo_seminovo", "saldo_sustentacao"]:
        if c not in d: d[c] = 0
    d["total_saldo"] = d[["saldo_novo", "saldo_seminovo", "saldo_sustentacao"]].sum(axis=1, min_count=1).fillna(0)

    # Família
    d["familia"] = d.apply(family_from_row, axis=1)

    return d

# ------------------ Spec / Comparação ------------------
def row_specs(r: pd.Series) -> Dict:
    return {
        "codigo": r.get("codigo"),
        "descricao": r.get("descricao"),
        "familia": r.get("familia", ""),
        "cpu_score": float(r.get("cpu_score", 0)),
        "ram_gb": float(r.get("ram_gb", 0)),
        "storage_gb": float(r.get("storage_gb", 0)),
        "storage_type": r.get("storage_type", ""),
        "gpu_dedicated": int(r.get("gpu_dedicated", 0)),
        "screen_in": float(r.get("screen_in", 0)),
        "total_saldo": float(r.get("total_saldo", 0)),
        "status": r.get("status"),
        "valor": r.get("valor"),
    }

def is_equal_or_better(
    candidate: pd.Series, base: pd.Series,
    cpu_cap: Optional[float] = 1.5,
    ram_factor_cap: Optional[float] = 2.0,
    storage_factor_cap: Optional[float] = 2.0
) -> Tuple[bool, int]:
    c = row_specs(candidate); b = row_specs(base)

    def ge(a, b):
        return (np.isnan(b) or a >= b or abs(a - b) < 1e-6)

    ok = True; wins = 0

    # CPU
    if ok and not ge(c["cpu_score"], b["cpu_score"]): ok = False
    if ok and cpu_cap is not None and b["cpu_score"] > 0 and c["cpu_score"] > b["cpu_score"] + cpu_cap: ok = False
    if ok and c["cpu_score"] > b["cpu_score"] + 1e-6: wins += 1

    # RAM
    if ok and not ge(c["ram_gb"], b["ram_gb"]): ok = False
    if ok and ram_factor_cap is not None and b["ram_gb"] > 0 and c["ram_gb"] > b["ram_gb"] * ram_factor_cap: ok = False
    if ok and c["ram_gb"] > b["ram_gb"] + 1e-6: wins += 1

    # Storage
    if ok and not ge(c["storage_gb"], b["storage_gb"]): ok = False
    if ok and storage_factor_cap is not None and b["storage_gb"] > 0 and c["storage_gb"] > b["storage_gb"] * storage_factor_cap: ok = False
    if ok and c["storage_gb"] > b["storage_gb"] + 1e-6: wins += 1

    # GPU
    if ok and c["gpu_dedicated"] < b["gpu_dedicated"]: ok = False
    if ok and c["gpu_dedicated"] > b["gpu_dedicated"]: wins += 1

    # Tela (bônus leve)
    if ok and c["screen_in"] >= b["screen_in"]: wins += 1

    return ok, wins

# ------------------ Recomendação ------------------
def recommend(
    df: pd.DataFrame, sku: str, topn: int = 30,
    only_status: tuple | None = None,
    min_screen: float | None = None,
    allow_family_override: bool = False,
    cpu_cap: Optional[float] = 1.5,
    ram_factor_cap: Optional[float] = 2.0,
    storage_factor_cap: Optional[float] = 2.0
) -> pd.DataFrame:

    d = enrich(df)

    base = d[d["codigo"].astype(str).str.upper() == sku.upper()]
    if base.empty:
        raise ValueError(f"SKU {sku} não encontrado.")
    base = base.iloc[0]
    base_family = base["familia"]

    pool = d.copy()
    if not allow_family_override:
        pool = pool[pool["familia"] == base_family]

    pool = pool[pool["total_saldo"] > 0]

    if only_status:
        pool = pool[pool["status"].astype(str).str.upper().isin([s.upper() for s in only_status])]
    if min_screen is not None:
        pool = pool[pool["screen_in"] >= float(min_screen)]

    rows = []
    for _, r in pool.iterrows():
        ok, wins = is_equal_or_better(
            r, base,
            cpu_cap=cpu_cap,
            ram_factor_cap=ram_factor_cap,
            storage_factor_cap=storage_factor_cap
        )
        if ok:
            score = wins * 2 + (r["cpu_score"] - base["cpu_score"])
            rows.append((score, r))

    if not rows:
        return pd.DataFrame()

    rows.sort(key=lambda x: x[0], reverse=True)
    return pd.DataFrame([row_specs(r) for _, r in rows[:topn]])
