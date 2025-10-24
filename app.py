import streamlit as st
import pandas as pd
from notebook_matcher import enrich, recommend, find_batteries_for_notebook  # usa versão atualizada

# =========================================
# Config Streamlit
# =========================================
st.set_page_config(
    page_title="ArkMatch",
    page_icon="🔎",
    layout="wide",
)

# =========================================
# Carregar base integrada do repo
# =========================================
BASE_PATH = "data/produtos_final.xlsx"

@st.cache_data(show_spinner=False)
def load_data():
    raw = pd.read_excel(BASE_PATH)
    return raw

df_raw = load_data()
df_enriched = enrich(df_raw)

# métricas topo
total_itens        = len(df_enriched)
com_estoque        = (df_enriched["total_saldo"] > 0).sum()
notebooks_qtd      = (df_enriched["familia"].str.upper() == "NOTEBOOK").sum()
desktops_qtd       = (df_enriched["familia"].str.upper() == "DESKTOP").sum()
tablets_qtd        = (df_enriched["familia"].str.upper() == "TABLET").sum()
smartphones_qtd    = (df_enriched["familia"].str.upper() == "SMARTPHONE").sum()

# =========================================
# HEADER (logo grande centralizada)
# =========================================
header_col = st.columns([1,1,1])[1]
with header_col:
    st.markdown(
        """
        <div style="
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            margin-top:0.5rem;
            margin-bottom:1rem;
        ">
            <img src="https://arknet.arklok.com.br/assets/img/logos/logo-arklok.png"
                 style="height:64px;">
        </div>
        """,
        unsafe_allow_html=True,
    )

# métricas
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total de itens", f"{total_itens}")
c2.metric("Com estoque", f"{com_estoque}")
c3.metric("Notebooks", f"{notebooks_qtd}")
c4.metric("Desktops", f"{desktops_qtd}")
c5.metric("Tablets/Smartphones", f"{tablets_qtd + smartphones_qtd}")

st.markdown("---")

# =========================================
# Busca
# =========================================
st.header("Buscar equivalentes")

sku_input = st.text_input(
    "SKU base (ex.: NB3091 / DT0XXX / TB0XXX / SM0XXX)",
    value="",
)

show_diag = st.checkbox("Mostrar diagnóstico", value=False)

buscar_btn = st.button(
    "Encontrar compatíveis",
    type="primary",
    use_container_width=True,
)

# =========================================
# Execução
# =========================================
if buscar_btn:
    sku_upper = sku_input.strip().upper()
    if not sku_upper:
        st.error("Digite um SKU.")
    else:
        try:
            recs = recommend(
                df_raw,
                sku_upper,
                topn=30,
            )

            base_line = df_enriched[
                df_enriched["codigo"].astype(str).str.upper() == sku_upper
            ]
            if base_line.empty:
                st.error(f"SKU {sku_upper} não encontrado.")
            else:
                base_row = base_line.iloc[0]

                if show_diag:
                    st.subheader("Interpretação do SKU base:")
                    diag_cols = [
                        "codigo","descricao","familia",
                        "ram_gb","storage_gb","storage_type","gpu_dedicated",
                        "screen_in","total_saldo","status","fabricante"
                    ]
                    diag_df = base_line[diag_cols].copy()
                    diag_df = diag_df.rename(columns={
                        "codigo":"Código",
                        "descricao":"Descrição",
                        "familia":"Família",
                        "ram_gb":"RAM (GB)",
                        "storage_gb":"Armazenamento (GB)",
                        "storage_type":"Tipo de armazenamento",
                        "gpu_dedicated":"GPU dedicada",
                        "screen_in":"Tela (pol)",
                        "total_saldo":"Saldo total",
                        "status":"Status",
                        "fabricante":"Fabricante",
                    })
                    st.dataframe(
                        diag_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                # baterias compatíveis (notebook only)
                bat_df = find_batteries_for_notebook(df_raw, base_row, topn=20)

                if recs.empty:
                    st.warning(
                        "Nenhuma opção igual/superior encontrada com os critérios padrão."
                    )
                else:
                    st.subheader("Opções compatíveis encontradas!")
                    show_cols = [
                        "codigo","descricao","familia","familia_conf","fabricante",
                        "ram_gb","storage_gb","storage_type","gpu_dedicated",
                        "screen_in","total_saldo","status"
                    ]
                    out_table = recs[show_cols].copy()
                    out_table = out_table.rename(columns={
                        "codigo":"Código",
                        "descricao":"Descrição",
                        "familia":"Família",
                        "familia_conf":"Conf. família",
                        "fabricante":"Fabricante",
                        "ram_gb":"RAM (GB)",
                        "storage_gb":"Armazenamento (GB)",
                        "storage_type":"Tipo de armazenamento",
                        "gpu_dedicated":"GPU dedicada",
                        "screen_in":"Tela (pol)",
                        "total_saldo":"Saldo total",
                        "status":"Status",
                    })
                    st.dataframe(
                        out_table,
                        use_container_width=True,
                        hide_index=True,
                    )

                if show_diag and not bat_df.empty:
                    st.subheader("Baterias compatíveis (mesmo modelo):")
                    bat_table = bat_df.rename(columns={
                        "codigo":"Código",
                        "descricao":"Descrição",
                        "fabricante":"Fabricante",
                        "modelo_mencionado":"Modelo base",
                        "total_saldo":"Saldo total",
                        "status":"Status",
                        "valor":"Valor",
                    })
                    st.dataframe(
                        bat_table,
                        use_container_width=True,
                        hide_index=True,
                    )

        except ValueError as e:
            st.error(str(e))

st.markdown(
    f"""
    <div style="margin-top:2rem;font-size:0.8rem;color:#aaa;">
    Base integrada: <code>{BASE_PATH}</code>
    </div>
    """,
    unsafe_allow_html=True,
)
