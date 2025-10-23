# app.py
from __future__ import annotations
import os
import pandas as pd
import streamlit as st

from notebook_matcher import (
    enrich,
    recommend,
    find_batteries_for_notebook,
)

# ----------------------------
# Configuração da página
# ----------------------------
st.set_page_config(page_title="ArkMatch", layout="wide", page_icon="🔎")

# CSS: header centralizado + esconder botões do Streamlit
st.markdown(
    """
    <style>
      /* Esconde menu hamburger, deploy e rodapé */
      [data-testid="stToolbar"] { display: none !important; }
      [data-testid="stDecoration"] { display: none !important; }
      [data-testid="stStatusWidget"] { display: none !important; }
      header { visibility: hidden; }
      footer { visibility: hidden; }
      [data-testid="stBaseButton-headerNoPadding"] { display: none !important; }
      [data-testid="stActionButtonIcon"] { display: none !important; }

      /* Header visual */
      .ark-header {
        width: 100%;
        text-align: center;
        margin: 12px 0 6px 0;
      }
      .ark-header img {
        height: 40px;
        margin-bottom: 10px;
        opacity: 0.95;
      }
      .ark-title {
        text-align: center;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin: -2px 0 6px 0;
      }
      /* Tabelas */
      .small-note { font-size: 13px; opacity: .7; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Header visual
# ----------------------------
LOGO_URL = "https://arknet.arklok.com.br/assets/img/logos/logo-arklok.png"
st.markdown(
    f"""
    <div class="ark-header">
      <img src="{LOGO_URL}" alt="Arklok" />
      <div class="ark-title">ArkMatch</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# Carregar base integrada
# ----------------------------
@st.cache_data(show_spinner=False)
def load_base() -> pd.DataFrame:
    path = os.path.join("data", "produtos_final.xlsx")
    df = pd.read_excel(path, engine="openpyxl")
    # Garantir colunas esperadas como string
    if "codigo" in df.columns:
        df["codigo"] = df["codigo"].astype(str)
    return df


def _rename_cols_portugues(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "codigo": "Código",
        "descricao": "Descrição",
        "familia": "Família",
        "familia_conf": "Conf. família",
        "fabricante": "Fabricante",
        # cpu_score e cpu_class não são mostrados
        "ram_gb": "RAM (GB)",
        "storage_gb": "Armazenamento (GB)",
        "storage_type": "Tipo de armazenamento",
        "gpu_dedicated": "GPU dedicada",
        "screen_in": "Tela (pol)",
        "total_saldo": "Saldo total",
        "status": "Status",
        "valor": "Valor",
        "modelo_base": "Modelo base",
    }
    cols = [c for c in df.columns if c in rename]
    out = df[cols].rename(columns=rename)
    return out


# ============================
# UI
# ============================
df_raw = load_base()

st.markdown("### Buscar equivalentes")

sku = st.text_input("SKU base (ex.: NB3091 / DTOXXX / TB0XXX / SM0XXX)", value="", max_chars=50).strip().upper()
col_btn1, col_diag = st.columns([3, 1])
with col_diag:
    show_diag = st.checkbox("Mostrar diagnóstico", value=False)

buscar = col_btn1.button("Encontrar compatíveis", use_container_width=True)

# Espaço
st.markdown("<br/>", unsafe_allow_html=True)

if buscar:
    if not sku:
        st.error("Informe um SKU.")
        st.stop()

    try:
        # Enriquecer a base (uma vez para todo o fluxo)
        edf = enrich(df_raw)

        base = edf[edf["codigo"].astype(str).str.upper() == sku]
        if base.empty:
            st.error(f"SKU **{sku}** não encontrado.")
            st.stop()

        base_row = base.iloc[0]

        # Diagnóstico do SKU base (sem cpu_score/cpu_class)
        if show_diag:
            st.markdown("#### Interpretação do SKU base:")
            base_show = base.copy()
            base_show = base_show.drop(columns=[c for c in ["cpu_score", "cpu_class"] if c in base_show.columns])
            st.dataframe(_rename_cols_portugues(base_show), use_container_width=True, hide_index=True)

        # Recomendações
        try:
            rec = recommend(df_raw, sku=sku, topn=30)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        st.markdown("#### Opções compatíveis encontradas!")
        if rec.empty:
            st.warning("Nenhuma opção igual/superior encontrada com estoque para este SKU.")
        else:
            # Esconde colunas internas
            rec_show = rec.drop(columns=[c for c in ["cpu_score", "cpu_class"] if c in rec.columns])
            st.dataframe(_rename_cols_portugues(rec_show), use_container_width=True, hide_index=True)

        # Baterias para notebooks
        if str(base_row.get("familia", "")).upper() == "NOTEBOOK":
            st.markdown("#### Baterias compatíveis para este notebook")
            bat = find_batteries_for_notebook(df_raw, base_row, topn=20)
            if bat.empty:
                st.info("Nenhuma bateria compatível localizada na base.")
            else:
                st.dataframe(
                    bat.rename(
                        columns={
                            "codigo": "Código",
                            "descricao": "Descrição",
                            "fabricante": "Fabricante",
                            "modelo_mencionado": "Modelo (extraído)",
                            "total_saldo": "Saldo total",
                            "status": "Status",
                            "valor": "Valor",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    except Exception as ex:
        st.error(f"Ocorreu um erro: {ex}")

# Rodapé leve
st.markdown('<div class="small-note">Base integrada: <code>data/produtos_final.xlsx</code></div>', unsafe_allow_html=True)
