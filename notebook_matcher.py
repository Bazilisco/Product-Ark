# notebook_matcher.py
from __future__ import annotations
import re
from typing import Tuple, Dict, Optional, List
import pandas as pd

# =========================================================
# Utils básicos
# =========================================================
def _norm(s) -> str:
    if pd.isna(s):
        return ""
    return str(s).upper().strip()

def _coalesce(*vals):
    for v in vals:
        if pd.isna(v):
            continue
        ss = str(v).strip()
        if ss:
            return ss
    return ""

# =========================================================
# CPU: score (desempate) + CLASSE (equivalência)
# =========================================================
APPLE_BASE = {"M1": 2, "M2": 3, "M3": 4, "M4": 5}

def parse_cpu_score(s: str) -> float:
    """Score numérico – serve só para desempate e para limitar 'muito superior'."""
    s = _norm(s)
    if not s:
        return 0.0
    # Apple
    m = re.search(r"\bM([1-5])\b(?:\s*(PRO|MAX|ULTRA))?", s)
    if m:
        base = APPLE_BASE.get(f"M{m.group(1)}", 0)
        bump = {"PRO": .5, "MAX": 1.0, "ULTRA": 1.5}.get((m.group(2) or "").upper(), 0)
        return base + bump
    # Core Ultra
    m = re.search(r"\bCORE\s+ULTRA\s+(5|7|9)\b", s)
    if m:
        return {"5": 3.0, "7": 4.0, "9": 5.0}[m.group(1)]
    # Core iX
    m = re.search(r"\bCORE\s*I\s*([3579])\b", s)
    if m:
        return {"3": 1.0, "5": 2.0, "7": 3.0, "9": 4.0}[m.group(1)]
    # Ryzen
    m = re.search(r"\bRYZEN\s+([3579])\b", s)
    if m:
        return {"3": 1.0, "5": 2.0, "7": 3.0, "9": 4.0}[m.group(1)]
    # Mobile
    if any(k in s for k in ["SNAPDRAGON", "DIMENSITY", "EXYNOS", "A1", "A12", "A13", "A14", "A15", "A16"]):
        return 2.5
    return 0.0

def cpu_class_from_text(s: str) -> int:
    """
    Classe de equivalência (usada para aceitar 'igual ou superior'):
      1: i3 / ryzen3
      2: i5 / ryzen5 / core ultra 5 / M1 / mobile médio
      3: i7 / ryzen7 / core ultra 7 / M2 / M3 / mobile alto
      4: i9 / ryzen9 / core ultra 9 / M4 / mobile topo
    """
    s = _norm(s)
    if not s:
        return 0
    # Apple
    m = re.search(r"\bM([1-5])\b", s)
    if m:
        x = int(m.group(1))
        if x == 1: return 2
        if x in (2, 3): return 3
        if x >= 4: return 4
    # Core Ultra
    m = re.search(r"\bCORE\s+ULTRA\s+(5|7|9)\b", s)
    if m:
        return {"5": 2, "7": 3, "9": 4}[m.group(1)]
    # Core iX
    m = re.search(r"\bCORE\s*I\s*([3579])\b", s)
    if m:
        return {"3": 1, "5": 2, "7": 3, "9": 4}[m.group(1)]
    # Ryzen
    m = re.search(r"\bRYZEN\s+([3579])\b", s)
    if m:
        return {"3": 1, "5": 2, "7": 3, "9": 4}[m.group(1)]
    # Mobile – heurística simples
    if any(k in s for k in ["SNAPDRAGON 8", "DIMENSITY 9", "A1", "A15", "A16"]):
        return 4
    if any(k in s for k in ["SNAPDRAGON 7", "DIMENSITY 8"]):
        return 3
    if any(k in s for k in ["SNAPDRAGON", "DIMENSITY", "EXYNOS"]):
        return 2
    return 0

# =========================================================
# Storage (tamanho + tipo)
# =========================================================
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
    """Retorna (GB, tipo, texto_sem_storage) – protege 'M1/M2' de virar M.2."""
    s0 = _norm(text)
    s = re.sub(r"\bM([1-5])\b", r"APL\1", s0)
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
            if g[0] in ["SSD", "NVME", "NV.ME", "M.2", "HDD", "HD"]:
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
        stype = stype.replace("NV.ME", "NVME").replace("M2", "M.2")
        stype = "SSD" if stype in ["SSD", "NVME", "M.2"] else ("HDD" if stype in ["HDD", "HD"] else "")
        matched = m.group(0).replace("APL", "M")
        clean = s0.replace(matched, "", 1).strip()
        return gb, stype, clean
    return 0.0, "", s0

# =========================================================
# RAM / GPU / TELA
# =========================================================
RAM_TIPICOS = [2,3,4,6,8,12,16,18,24,32,48,64,96,128]
def parse_ram_from_text(t: str) -> float:
    s = _norm(t)
    pairs = re.findall(r"\b(\d{1,3})\s*(GB|G)\b", s)
    if not pairs:
        return 0.0
    vals = [int(p[0]) for p in pairs]
    tip = [v for v in vals if v in RAM_TIPICOS]
    return float(max(tip) if tip else max(vals))

def parse_gpu_dedicated(a: str, b: str) -> int:
    t = f"{_norm(a)} {_norm(b)}"
    return 1 if re.search(r"\b(NVIDIA|GEFORCE|RTX|GTX|RADEON|DEDICAD[AO])\b", t) else 0

def parse_screen_inches(s: str) -> float:
    s = _norm(s).replace(",", ".")
    m = re.search(r'(\d{1,2}(?:\.\d)?)\s*(POL|")', s)
    if m:
        v = float(m.group(1))
        if 5 <= v <= 19.9:
            return v
    m = re.search(r'\bTELA\b[^0-9]{0,6}(\d{1,2}(?:\.\d)?)', s)
    if m:
        v = float(m.group(1))
        if 5 <= v <= 19.9:
            return v
    m = re.search(r'\b(1?\d(?:\.\d)?)\b(?!\s*GB)', s)
    if m:
        v = float(m.group(1))
        if 5 <= v <= 19.9:
            return v
    return 0.0

# =========================================================
# Família (NOTEBOOK/DESKTOP/TABLET/SMARTPHONE) – robusta
# =========================================================
NOTEBOOK_KEYS = ["NOTEBOOK","LAPTOP","MACBOOK","LATITUDE","THINKPAD","ELITEBOOK","PROBOOK","VOSTRO","IDEAPAD","LEGION","YOGA","CHROMEBOOK","XPS","VIVOBOOK","ZENBOOK"]
DESKTOP_KEYS  = ["DESKTOP","TORRE","TOWER","WORKSTATION","OPTIPLEX","THINKCENTRE","ELITEDESK","PRODESK","MICRO TOWER","USFF","SFF"]
TABLET_KEYS   = ["TABLET","IPAD","GALAXY TAB","SURFACE","LENOVO TAB","MI PAD","IPAD PRO","IPAD AIR"]
# Smartphone: removeu regex genérico A\d+ (podia confundir 'A14 POL')
PHONE_KEYS    = ["SMARTPHONE","CELULAR","IPHONE","GALAXY S","MOTO","XIAOMI","REDMI","POCO","EDGE",
                 r"GALAXY\s*A\d{1,2}\b", r"MOTO\s*G", r"S\d{1,2}\b"]

def _collect_text_for_family(r: pd.Series) -> str:
    # NÃO usamos 'codigo' aqui para não enviesar pela sigla NB/PC/TB/SM
    parts = [str(r.get(c,"")) for c in ["descricao","descricao_subtitulo","modelo","ger_equipamento","tipo","tela"]]
    return _norm(" ".join(parts))

def _kw_score(t: str, keys: List[str]) -> int:
    sc = 0
    for k in keys:
        if re.search(rf"\b{k}\b", t):
            sc += 1
    return sc

def family_from_row(r: pd.Series) -> tuple[str,int]:
    t = _collect_text_for_family(r)
    s_note = _kw_score(t, NOTEBOOK_KEYS)
    s_desk = _kw_score(t, DESKTOP_KEYS)
    s_tab  = _kw_score(t, TABLET_KEYS)
    s_ph   = _kw_score(t, PHONE_KEYS)
    scores = {"NOTEBOOK":s_note,"DESKTOP":s_desk,"TABLET":s_tab,"SMARTPHONE":s_ph}
    fam = max(scores, key=scores.get); conf = scores[fam]
    if conf <= 1:
        return "OUTROS", 0
    return fam, conf

# =========================================================
# Enriquecimento
# =========================================================
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Texto consolidado
    text = (
        d.get("descricao","").astype(str) + " " +
        d.get("descricao_subtitulo","").astype(str) + " " +
        d.get("processador","").astype(str) + " " +
        d.get("memoria_ram","").astype(str) + " " +
        d.get("armazenamento","").astype(str) + " " +
        d.get("placa_de_video","").astype(str) + " " +
        d.get("tamanho_tela","").astype(str)
    )

    d["cpu_score"] = text.apply(parse_cpu_score)
    d["cpu_class"] = text.apply(cpu_class_from_text)

    stor = text.apply(parse_storage_with_type)
    s_df = pd.DataFrame(stor.tolist(), index=d.index)
    d["storage_gb"]   = s_df[0].astype(float)
    d["storage_type"] = s_df[1].astype(str)
    text_wo_storage   = s_df[2].astype(str)

    ram_col  = d.get("memoria_ram","").astype(str).apply(parse_ram_from_text)
    ram_text = text_wo_storage.apply(parse_ram_from_text)
    d["ram_gb"] = ram_col.mask(ram_col == 0, ram_text)

    d["gpu_dedicated"] = [parse_gpu_dedicated(a,b) for a,b in zip(d.get("placa_de_video",""), text)]

    scr_col  = d.get("tamanho_tela","").astype(str).apply(parse_screen_inches)
    scr_text = text.apply(parse_screen_inches)
    d["screen_in"] = scr_col.mask(scr_col == 0, scr_text)

    # Saldo total
    for c in ["saldo_novo","saldo_seminovo","saldo_sustentacao"]:
        if c not in d: d[c] = 0
    d["total_saldo"] = d[["saldo_novo","saldo_seminovo","saldo_sustentacao"]].sum(axis=1, min_count=1).fillna(0)

    # Família
    fam_conf = d.apply(family_from_row, axis=1)
    fam_df = pd.DataFrame(fam_conf.tolist(), columns=["familia","familia_conf"], index=d.index)
    d["familia"] = fam_df["familia"]; d["familia_conf"] = fam_df["familia_conf"]

    # Fabricante normalizado
    if "fabricante" not in d.columns: d["fabricante"] = ""
    d["fabricante"] = d["fabricante"].astype(str).str.upper()

    # Campo “eh_bateria”
    t_all = (
        d.get("tipo","").astype(str) + " " +
        d.get("ger_equipamento","").astype(str) + " " +
        d.get("descricao","").astype(str)
    ).str.upper()
    d["eh_bateria"] = t_all.str.contains(r"\bBATERIA(S)?\b", regex=True)

    # Tenta extrair “modelo_base” (ex.: LATITUDE 3420, THINKPAD T14, E14, etc.)
    d["modelo_base"] = d.apply(extract_model_base, axis=1)

    return d

# =========================================================
# Extração de “modelo base” de notebook (p/ baterias)
# =========================================================
NOTEBOOK_MODEL_PATTERNS = [
    r"\b(LATITUDE\s+\d{3,4})\b",
    r"\b(THINKPAD\s+[A-Z]\d{1,2})\b",
    r"\b(THINKPAD\s+T\d{1,2})\b",
    r"\b(ELITEBOOK\s+\d{3,4})\b",
    r"\b(PROBOOK\s+\d{3,4})\b",
    r"\b(IDEAPAD\s+[A-Z]?\d{3,4})\b",
    r"\b(VOSTRO\s+\d{3,4})\b",
    r"\b(MACBOOK\s+AIR|MACBOOK\s+PRO)\b",
    r"\b(E\d{1,2}\s*GEN\s*\d)\b",
    r"\b(E\d{1,2})\b",
    r"\b(T14|T15|E14|E15|L14|L15)\b",
]

def extract_model_base(r: pd.Series) -> str:
    """
    Retorna uma 'assinatura' de modelo de notebook a partir de descrição/modelo,
    ex.: 'LATITUDE 3420', 'THINKPAD T14', 'E14', 'MACBOOK PRO'.
    """
    if str(r.get("familia","")).upper() != "NOTEBOOK":
        return ""
    tx = _norm(_coalesce(r.get("modelo",""), r.get("descricao",""), r.get("descricao_subtitulo","")))
    for pat in NOTEBOOK_MODEL_PATTERNS:
        m = re.search(pat, tx)
        if m:
            return m.group(1).strip()
    # fallback: fabricante + número de 3/4 dígitos (ex.: DELL 3420)
    fab = _norm(r.get("fabricante",""))
    m2 = re.search(r"\b(\d{4})\b", tx)
    if fab and m2:
        return f"{fab} {m2.group(1)}"
    return ""

# =========================================================
# Matching principal
# =========================================================
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
        "eh_bateria": bool(r.get("eh_bateria", False)),
        "modelo_base": r.get("modelo_base","")
    }

def is_equal_or_better(c: pd.Series, b: pd.Series,
                       cpu_cap: float|None=1.5,
                       ram_factor_cap: float|None=2.0,
                       storage_factor_cap: float|None=2.0) -> tuple[bool,int]:
    C, B = row_specs(c), row_specs(b)
    ok, wins = True, 0

    # CPU – classe (equivalência) + trava de "muito superior" pelo score
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

    # Tela – bônus leve (para notebooks/tablets/smartphones)
    if C["screen_in"] >= B["screen_in"] - 1e-6:
        wins += 1

    return ok, wins

def recommend(df: pd.DataFrame, sku: str, topn: int = 30,
              only_status: tuple|None=None, min_screen: float|None=None,
              allow_family_override: bool=False,
              cpu_cap: float|None=1.5, ram_factor_cap: float|None=2.0, storage_factor_cap: float|None=2.0) -> pd.DataFrame:
    """
    Retorna recomendações 'iguais ou superiores' PRIORIZANDO a mesma marca,
    mas incluindo outras marcas equivalentes. Matching é **estrito por família**.
    """
    d = enrich(df)
    base_df = d[d["codigo"].astype(str).str.upper() == sku.upper()]
    if base_df.empty:
        raise ValueError(f"SKU {sku} não encontrado.")
    base = base_df.iloc[0]

    base_family = base["familia"]
    base_brand  = _norm(base.get("fabricante",""))

    # === ESTRITO POR FAMÍLIA ===
    pool = d[d["familia"] == base_family].copy()

    # Só itens com estoque
    pool = pool[pool["total_saldo"] > 0]

    if only_status:
        opts = [s.upper() for s in only_status]
        pool = pool[pool["status"].astype(str).str.upper().isin(opts)]
    if min_screen is not None:
        pool = pool[pool["screen_in"] >= float(min_screen)]

    # Avalia compatibilidade
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

    # Prioriza mesma marca, mas garante diversidade
    sb = [t for t in scored if t[0]]
    ob = [t for t in scored if not t[0]]

    sb.sort(key=lambda x: (x[1], x[2]), reverse=True)
    ob.sort(key=lambda x: (x[1], x[2]), reverse=True)

    reserve_other = min(10, max(3, topn // 3))
    reserve_other = min(reserve_other, len(ob))
    take_same = min(len(sb), topn - reserve_other)

    out_rows = sb[:take_same] + ob[:reserve_other]
    if len(out_rows) < topn:
        out_rows += sb[take_same: take_same + (topn - len(out_rows))]
    if len(out_rows) < topn:
        out_rows += ob[reserve_other: reserve_other + (topn - len(out_rows))]

    df_out = pd.DataFrame([row_specs(r) for _,_,_,r in out_rows[:topn]])
    return df_out

# =========================================================
# BATERIAS compatíveis para UM notebook
# =========================================================
def _battery_filter(d: pd.DataFrame) -> pd.DataFrame:
    """Retorna apenas as linhas que são baterias (pela flag 'eh_bateria')."""
    if "eh_bateria" not in d.columns:
        return d.iloc[0:0]
    return d[d["eh_bateria"] == True].copy()

def _battery_matches_model(batt_text: str, model_sig: str, brand: str) -> bool:
    """
    Heurística: bateria deve mencionar a mesma 'assinatura' de modelo
    (ex.: 'LATITUDE 3420', 'THINKPAD T14', 'E14'), e bater com fabricante.
    """
    t = _norm(batt_text)
    if brand and brand not in t:
        return False
    # exige aparecer a assinatura de modelo
    if model_sig and model_sig in t:
        return True
    # fallback: se assinatura for 'DELL 3420' tenta só o número com 'LATITUDE'/etc.
    m = re.search(r"\b(\d{3,4})\b", model_sig)
    if m and ("LATITUDE" in t or "THINKPAD" in t or "ELITEBOOK" in t or "PROBOOK" in t or "VOSTRO" in t):
        return m.group(1) in t
    return False

def find_batteries_for_notebook(df: pd.DataFrame, base_row: pd.Series, topn: int = 20) -> pd.DataFrame:
    d = enrich(df)
    # Só segue se for notebook
    if str(base_row.get("familia","")).upper() != "NOTEBOOK":
        return pd.DataFrame()

    model_sig = base_row.get("modelo_base","")
    brand     = _norm(base_row.get("fabricante",""))

    batt = _battery_filter(d)
    if batt.empty:
        return pd.DataFrame()

    out = []
    for _, r in batt.iterrows():
        txt = " ".join([
            str(r.get("descricao","")),
            str(r.get("descricao_subtitulo","")),
            str(r.get("modelo","")),
            str(r.get("ger_equipamento","")),
            str(r.get("tipo",""))
        ])
        if _battery_matches_model(txt, model_sig, brand):
            out.append({
                "codigo": r.get("codigo"),
                "descricao": r.get("descricao"),
                "fabricante": r.get("fabricante"),
                "modelo_mencionado": model_sig,
                "total_saldo": r.get("total_saldo",0),
                "status": r.get("status"),
                "valor": r.get("valor"),
            })
    df_out = pd.DataFrame(out)
    if df_out.empty:
        return df_out
    # ordena por saldo desc e, depois, por preço (se houver)
    if "valor" in df_out.columns:
        try:
            df_out["valor_num"] = pd.to_numeric(df_out["valor"].astype(str).str.replace("R$", "").str.replace(".", "").str.replace(",", "."), errors="coerce")
        except Exception:
            df_out["valor_num"] = None
        df_out = df_out.sort_values(by=["total_saldo","valor_num"], ascending=[False, True], na_position="last")
        df_out = df_out.drop(columns=["valor_num"])
    else:
        df_out = df_out.sort_values(by=["total_saldo"], ascending=False)
    return df_out.head(topn)
