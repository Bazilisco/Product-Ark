# notebook_matcher.py
from __future__ import annotations
import re
from typing import Tuple, Dict, Optional
import pandas as pd

# ------------------ util ------------------
def _norm(s) -> str:
    if pd.isna(s):
        return ""
    return str(s).upper().strip()

# ------------------ CPU: score + CLASSE ------------------
CPU_TIERS = {
    "I3": 1, "I5": 2, "I7": 3, "I9": 4,
    "ULTRA 5": 3, "ULTRA 7": 4, "ULTRA 9": 5,  # mantido p/ referência
    "RYZEN 3": 1, "RYZEN 5": 2, "RYZEN 7": 3, "RYZEN 9": 4,
}
APPLE_BASE = {"M1": 2, "M2": 3, "M3": 4, "M4": 5}

def parse_cpu(s: str) -> float:
    """Score numérico (usado só para desempate/limite de 'muito superior')."""
    s = _norm(s)
    if not s:
        return 0.0
    m = re.search(r"\bM([1-5])\b(?:\s*(PRO|MAX|ULTRA))?", s)
    if m:
        base = APPLE_BASE.get(f"M{m.group(1)}", 0)
        bump = {"PRO": .5, "MAX": 1.0, "ULTRA": 1.5}.get((m.group(2) or "").upper(), 0)
        return base + bump
    m = re.search(r"\bCORE\s+ULTRA\s+(5|7|9)\b", s)
    if m:
        return {"5":3.0, "7":4.0, "9":5.0}[m.group(1)]
    m = re.search(r"\bCORE\s*I\s*([3579])\b", s)
    if m:
        return CPU_TIERS.get(f"I{m.group(1)}", 0)
    m = re.search(r"\bRYZEN\s+([3579])\b", s)
    if m:
        return CPU_TIERS.get(f"RYZEN {m.group(1)}", 0)
    if any(k in s for k in ["SNAPDRAGON", "DIMENSITY", "EXYNOS"]):
        return 2.5
    return 0.0

def cpu_class_from_text(s: str) -> int:
    """
    Classe de equivalência:
      1: i3 / ryzen3
      2: i5 / ryzen5 / core ultra 5 / M1
      3: i7 / ryzen7 / core ultra 7 / M2 / M3
      4: i9 / ryzen9 / core ultra 9 / M4
    """
    s = _norm(s)
    if not s:
        return 0
    # Apple
    m = re.search(r"\bM([1-5])\b", s)
    if m:
        x = int(m.group(1))
        if x == 1: return 2
        if x in (2,3): return 3
        if x >= 4: return 4
    # Core Ultra
    m = re.search(r"\bCORE\s+ULTRA\s+(5|7|9)\b", s)
    if m:
        return {"5":2, "7":3, "9":4}[m.group(1)]
    # Core iX
    m = re.search(r"\bCORE\s*I\s*([3579])\b", s)
    if m:
        return {"3":1, "5":2, "7":3, "9":4}[m.group(1)]
    # Ryzen
    m = re.search(r"\bRYZEN\s+([3579])\b", s)
    if m:
        return {"3":1, "5":2, "7":3, "9":4}[m.group(1)]
    # Mobile genérica
    if any(k in s for k in ["SNAPDRAGON", "DIMENSITY", "EXYNOS"]):
        return 2
    return 0

# ------------------ Storage (tamanho + tipo) ------------------
def _parse_num(x: str) -> float:
    s = str(x).strip().replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    return float(s)

def _to_gb(num: str, unit: str) -> float:
    val = _parse_num(num)
    return val * 1024.0 if unit == "TB" else val

def parse_storage_with_type(text: str) -> Tuple[float, str, str]:
    s0 = _norm(text)
    s = re.sub(r"\bM([1-5])\b", r"APL\1", s0)  # protege M1/M2... (CPU) de virar "M.2"
    pats = [
        r"\b(SSD|NVME|NV\.?ME|M\.?2|HDD|HD)\s*([\d\.,]+)\s*(TB|GB)\b",
        r"\b([\d\.,]+)\s*(TB|GB)\s*(SSD|NVME|NV\.?ME|M\.?2|HDD|HD)\b",
        r"\b([\d\.,]+)\s*(TB|GB)\b",
    ]
    for pat in pats:
        m = re.search(pat, s)
        if not m:
            continue
        g = [gg for gg in m.groups() if gg]
        if len(g) == 3:
            if g[0] in ["SSD","NVME","NV.ME","M.2","HDD","HD"]:
                stype, num, unit = g[0], g[1], g[2]
            else:
                num, unit, stype = g[0], g[1], g[2]
        elif len(g) == 2:
            num, unit = g; stype = ""
        else:
            continue
        if not re.fullmatch(r"[\d\.,]+", str(num).strip()):
            continue
        unit = unit.replace(" ", "")
        try:
            gb = _to_gb(num, unit)
        except Exception:
            continue
        stype = stype.replace("NV.ME","NVME").replace("M2","M.2")
        stype = "SSD" if stype in ["SSD","NVME","M.2"] else ("HDD" if stype in ["HDD","HD"] else "")
        matched = m.group(0).replace("APL","M")
        clean = s0.replace(matched, "", 1).strip()
        return gb, stype, clean
    return 0.0, "", s0

# ------------------ RAM ------------------
RAM_TIPICOS = [2,3,4,6,8,12,16,18,24,32,48,64,96,128]
def parse_ram_from_text(t: str) -> float:
    s = _norm(t)
    pairs = re.findall(r"\b(\d{1,3})\s*(GB|G)\b", s)
    if not pairs:
        return 0.0
    vals = [int(p[0]) for p in pairs]
    tip = [v for v in vals if v in RAM_TIPICOS]
    return float(max(tip) if tip else max(vals))

# ------------------ GPU / Tela ------------------
def parse_gpu_dedicated(a: str, b: str) -> int:
    t = f"{_norm(a)} {_norm(b)}"
    return 1 if re.search(r"\b(NVIDIA|GEFORCE|RTX|GTX|RADEON|DEDICAD[AO])\b", t) else 0

def parse_screen_inches(s: str) -> float:
    s = _norm(s).replace(",", ".")
    m = re.search(r'(\d{1,2}(?:\.\d)?)\s*(POL|")', s)
    if m:
        v = float(m.group(1))
        if 7 <= v <= 19.9:
            return v
    m = re.search(r'\bTELA\b[^0-9]{0,6}(\d{1,2}(?:\.\d)?)', s)
    if m:
        v = float(m.group(1))
        if 7 <= v <= 19.9:
            return v
    m = re.search(r'\b(1\d(?:\.\d)?)\b(?!\s*GB)', s)
    if m:
        v = float(m.group(1))
        if 10 <= v <= 19.9:
            return v
    return 0.0

# ------------------ Família ------------------
NOTEBOOK_KEYS = ["NOTEBOOK","LAPTOP","MACBOOK","LATITUDE","THINKPAD","ELITEBOOK","PROBOOK","VOSTRO","IDEAPAD","LEGION","YOGA","CHROMEBOOK","XPS","VIVOBOOK","ZENBOOK"]
DESKTOP_KEYS  = ["DESKTOP","TORRE","TOWER","WORKSTATION","OPTIPLEX","THINKCENTRE","ELITEDESK","PRODESK","MICRO TOWER","USFF","SFF"]
TABLET_KEYS   = ["TABLET","IPAD","GALAXY TAB","SURFACE","LENOVO TAB","MI PAD"]
PHONE_KEYS    = ["SMARTPHONE","CELULAR","IPHONE","GALAXY S","MOTO","XIAOMI","REDMI","POCO"]

def _collect_text_for_family(r: pd.Series) -> str:
    parts = [str(r.get(c,"")) for c in ["codigo","descricao","descricao_subtitulo","modelo","ger_equipamento","tipo","tela"]]
    return _norm(" ".join(parts))

def _kw_score(t: str, keys: list[str]) -> int:
    return sum(1 for k in keys if k in t)

def family_from_row(r: pd.Series) -> tuple[str,int]:
    t = _collect_text_for_family(r)
    s_note = _kw_score(t, NOTEBOOK_KEYS)
    s_desk = _kw_score(t, DESKTOP_KEYS)
    s_tab  = _kw_score(t, TABLET_KEYS)
    s_ph   = _kw_score(t, PHONE_KEYS)
    code = _norm(r.get("codigo"))
    if code.startswith("NB"): s_note += 1
    if code.startswith("DT") or code.startswith("PC"): s_desk += 1
    if code.startswith("TB"): s_tab += 1
    if code.startswith("SM"): s_ph += 1
    scores = {"NOTEBOOK":s_note,"DESKTOP":s_desk,"TABLET":s_tab,"SMARTPHONE":s_ph}
    fam = max(scores, key=scores.get); conf = scores[fam]
    if conf <= 1:
        return "OUTROS", 0
    return fam, conf

# ------------------ Enriquecimento ------------------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    text = (
        d.get("descricao","").astype(str) + " " +
        d.get("descricao_subtitulo","").astype(str) + " " +
        d.get("processador","").astype(str) + " " +
        d.get("memoria_ram","").astype(str) + " " +
        d.get("armazenamento","").astype(str) + " " +
        d.get("placa_de_video","").astype(str) + " " +
        d.get("tamanho_tela","").astype(str)
    )
    d["cpu_score"] = text.apply(parse_cpu)          # para desempate/limites
    d["cpu_class"] = text.apply(cpu_class_from_text)  # regra de equivalência

    stor = text.apply(parse_storage_with_type)
    s_df = pd.DataFrame(stor.tolist(), index=d.index)
    d["storage_gb"] = s_df[0].astype(float)
    d["storage_type"] = s_df[1].astype(str)
    text_wo_storage = s_df[2].astype(str)

    ram_col  = d.get("memoria_ram","").astype(str).apply(parse_ram_from_text)
    ram_text = text_wo_storage.apply(parse_ram_from_text)
    d["ram_gb"] = ram_col.mask(ram_col == 0, ram_text)

    d["gpu_dedicated"] = [parse_gpu_dedicated(a,b) for a,b in zip(d.get("placa_de_video",""), text)]

    scr_col  = d.get("tamanho_tela","").astype(str).apply(parse_screen_inches)
    scr_text = text.apply(parse_screen_inches)
    d["screen_in"] = scr_col.mask(scr_col == 0, scr_text)

    for c in ["saldo_novo","saldo_seminovo","saldo_sustentacao"]:
        if c not in d: d[c] = 0
    d["total_saldo"] = d[["saldo_novo","saldo_seminovo","saldo_sustentacao"]].sum(axis=1, min_count=1).fillna(0)

    fam_conf = d.apply(family_from_row, axis=1)
    fam_df = pd.DataFrame(fam_conf.tolist(), columns=["familia","familia_conf"], index=d.index)
    d["familia"] = fam_df["familia"]; d["familia_conf"] = fam_df["familia_conf"]
    if "fabricante" not in d.columns: d["fabricante"] = ""

    return d

# ------------------ Matching ------------------
def row_specs(r: pd.Series) -> Dict:
    return {
        "codigo": r.get("codigo"),
        "descricao": r.get("descricao"),
        "familia": r.get("familia",""),
        "familia_conf": int(r.get("familia_conf",0)),
        "fabricante": r.get("fabricante",""),
        "cpu_score": float(r.get("cpu_score",0)),
        "cpu_class": int(r.get("cpu_class",0)),
        "ram_gb": float(r.get("ram_gb",0)),
        "storage_gb": float(r.get("storage_gb",0)),
        "storage_type": r.get("storage_type",""),
        "gpu_dedicated": int(r.get("gpu_dedicated",0)),
        "screen_in": float(r.get("screen_in",0)),
        "total_saldo": float(r.get("total_saldo",0)),
        "status": r.get("status"),
        "valor": r.get("valor"),
    }

def is_equal_or_better(c: pd.Series, b: pd.Series,
                       cpu_cap: float|None=1.5,
                       ram_factor_cap: float|None=2.0,
                       storage_factor_cap: float|None=2.0) -> tuple[bool,int]:
    C, B = row_specs(c), row_specs(b)
    ok, wins = True, 0

    # CPU: usa classe (equivalência) + limita 'muito superior' pelo score
    if C["cpu_class"] < B["cpu_class"]:
        return False, 0
    if cpu_cap is not None and B["cpu_score"] > 0 and (C["cpu_score"] - B["cpu_score"]) > cpu_cap:
        return False, 0
    if C["cpu_score"] > B["cpu_score"] + 1e-6:
        wins += 1

    # RAM
    if C["ram_gb"] < B["ram_gb"] - 1e-6:
        return False, 0
    if ram_factor_cap is not None and B["ram_gb"] > 0 and C["ram_gb"] > B["ram_gb"] * ram_factor_cap:
        return False, 0
    if C["ram_gb"] > B["ram_gb"] + 1e-6:
        wins += 1

    # Storage
    if C["storage_gb"] < B["storage_gb"] - 1e-6:
        return False, 0
    if storage_factor_cap is not None and B["storage_gb"] > 0 and C["storage_gb"] > B["storage_gb"] * storage_factor_cap:
        return False, 0
    if C["storage_gb"] > B["storage_gb"] + 1e-6:
        wins += 1

    # GPU
    if C["gpu_dedicated"] < B["gpu_dedicated"]:
        return False, 0
    if C["gpu_dedicated"] > B["gpu_dedicated"]:
        wins += 1

    # Tela (bônus leve)
    if C["screen_in"] >= B["screen_in"] - 1e-6:
        wins += 1

    return ok, wins

def recommend(df: pd.DataFrame, sku: str, topn: int = 30,
              only_status: tuple|None=None, min_screen: float|None=None,
              allow_family_override: bool=False,
              cpu_cap: float|None=1.5, ram_factor_cap: float|None=2.0, storage_factor_cap: float|None=2.0) -> pd.DataFrame:
    d = enrich(df)
    base_df = d[d["codigo"].astype(str).str.upper() == sku.upper()]
    if base_df.empty:
        raise ValueError(f"SKU {sku} não encontrado.")
    base = base_df.iloc[0]

    base_family = base["familia"]; base_fconf = int(base.get("familia_conf",0))
    base_brand  = _norm(base.get("fabricante",""))

    pool = d.copy()
    if not allow_family_override and base_family != "OUTROS" and base_fconf >= 2:
        pool = pool[pool["familia"] == base_family]

    pool = pool[pool["total_saldo"] > 0]
    if only_status:
        opts = [s.upper() for s in only_status]
        pool = pool[pool["status"].astype(str).str.upper().isin(opts)]
    if min_screen is not None:
        pool = pool[pool["screen_in"] >= float(min_screen)]

    scored = []
    for _, r in pool.iterrows():
        ok, wins = is_equal_or_better(r, base, cpu_cap, ram_factor_cap, storage_factor_cap)
        if not ok:
            continue
        score = wins * 2 + (r["cpu_score"] - base["cpu_score"])
        same_brand = (_norm(r.get("fabricante","")) == base_brand) and (base_brand!="")
        saldo = float(r.get("total_saldo",0))
        scored.append((same_brand, score, saldo, r))

    if not scored:
        return pd.DataFrame()

    # separa p/ priorizar mesma marca mas garantir diversidade
    sb = [t for t in scored if t[0]]
    ob = [t for t in scored if not t[0]]

    sb.sort(key=lambda x: (x[1], x[2]), reverse=True)
    ob.sort(key=lambda x: (x[1], x[2]), reverse=True)

    reserve_other = min(10, max(3, topn // 3))
    reserve_other = min(reserve_other, len(ob))
    take_same = min(len(sb), topn - reserve_other)

    out_rows = sb[:take_same] + ob[:reserve_other]
    if len(out_rows) < topn:
        rem = topn - len(out_rows)
        out_rows += sb[take_same:take_same+rem]
    if len(out_rows) < topn:
        rem = topn - len(out_rows)
        out_rows += ob[reserve_other:reserve_other+rem]

    df_out = pd.DataFrame([row_specs(r) for _,_,_,r in out_rows[:topn]])
    return df_out
