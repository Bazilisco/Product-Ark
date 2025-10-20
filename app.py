# app.py
import streamlit as st
import pandas as pd
from notebook_matcher import enrich, recommend

st.set_page_config(page_title="Matcher de Equipamentos", layout="wide")

# CSS para remover Deploy/menus
st.markdown("""
<style>
header [data-testid="stHeaderActionElements"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
div[data-testid="stToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Matcher de Equipamentos (igual ou superior)")

DATA_PATH = "data/produtos_final.xlsx"
try:
    df = pd.read_excel(DATA_PATH)
except Exception:
    st.error(f"Não encontrei {DATA_PATH}. Coloque a planilha na pasta /data.")
    st.stop()

edf = enrich(df)

DISPLAY_MAP = {
    "codigo": "Código",
    "descricao": "Descrição",
    "familia": "Família",
    "cpu_score": "CPU (pontuação)",
    "ram_gb": "RAM (GB)",
    "storage_gb": "Armazenamento (GB)",
    "storage_type": "Tipo de armazenamento",
    "gpu_dedicated": "GPU dedicada",
    "screen_in": "Tela (pol)",
    "total_saldo": "Saldo total",
    "status": "Status",
    "valor": "Valor",
}
def pretty(df_: pd.DataFrame) -> pd.DataFrame:
    out = df_.copy()
    out.columns = [DISPLAY_MAP.get(c, c) for c in out.columns]
    return out

st.subheader("Resumo da base de produtos")
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Total de itens", len(edf))
with c2: st.metric("Com estoque", int((edf["total_saldo"] > 0).sum()))
with c3: st.metric("Notebooks", int((edf["familia"] == "NOTEBOOK").sum()))
with c4: st.metric("Desktops", int((edf["familia"] == "DESKTOP").sum()))
with c5: st.metric("Tablets / Smartphones", int((edf["familia"].isin(["TABLET", "SMARTPHONE"])).sum()))

with st.expander("Ver amostra da base (enriquecida)"):
    st.dataframe(pretty(edf.head(50)), use_container_width=True)

st.markdown("---")
st.subheader("Buscar equivalentes (igual ou superior)")

sku = st.text_input("SKU base (ex.: NB3091 / DT0XXX / TB0XXX / SM0XXX)").strip()

colA, colB, colC = st.columns(3)
with colA:
    status_filter = st.multiselect("Status permitido (opcional)", sorted(edf["status"].dropna().astype(str).unique()))
with colB:
    min_screen = st.number_input("Tela mínima (pol) — opcional", min_value=0.0, value=0.0, step=0.1)
with colC:
    allow_family_override = st.toggle("Permitir outras famílias (ex.: Notebook → Desktop)", value=False)

st.caption("Limites para evitar **muito superior** (ajuste se necessário):")
s1, s2, s3 = st.columns(3)
with s1:
    cpu_cap = st.slider("Máximo acima em CPU", 0.0, 5.0, 1.5, 0.1)
with s2:
    ram_cap = st.slider("RAM (máx. x vezes)", 1.0, 4.0, 2.0, 0.1)
with s3:
    stor_cap = st.slider("Armazenamento (máx. x vezes)", 1.0, 4.0, 2.0, 0.1)

if st.button("Encontrar compatíveis"):
    if not sku:
        st.error("Informe o SKU base.")
    else:
        try:
            only_status = tuple(status_filter) if status_filter else None
            ms = min_screen if min_screen > 0 else None
            recs = recommend(
                df, sku=sku, topn=100,
                only_status=only_status,
                min_screen=ms,
                allow_family_override=allow_family_override,
                cpu_cap=cpu_cap,
                ram_factor_cap=ram_cap,
                storage_factor_cap=stor_cap
            )
            if recs.empty:
                st.warning("Nenhuma opção igual/superior com os filtros e limites atuais.")
            else:
                st.success(f"Encontradas {len(recs)} opções.")
                st.dataframe(pretty(recs), use_container_width=True)
                st.download_button(
                    "Baixar CSV",
                    pretty(recs).to_csv(index=False).encode("utf-8"),
                    file_name=f"recomendacoes_{sku}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(str(e))

st.caption("A comparação considera CPU, RAM, Armazenamento (com tipo) e GPU. Tela conta como bônus.")
