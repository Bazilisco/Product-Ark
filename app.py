# app.py
import streamlit as st
import pandas as pd
from notebook_matcher import enrich, recommend

st.set_page_config(page_title="Product Ark — Matcher", layout="wide")

# Esconde menus/Deploy
st.markdown("""
<style>
header [data-testid="stHeaderActionElements"]{display:none!important;}
#MainMenu, footer, div[data-testid="stToolbar"]{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🔍 Product Ark — Equipamento igual ou superior")

DATA_PATH = "data/produtos_final.xlsx"
try:
    df = pd.read_excel(DATA_PATH)
except Exception:
    st.error(f"Não encontrei a base em **{DATA_PATH}**. Coloque o arquivo lá e recarregue.")
    st.stop()

# Enriquecimento (CPU/RAM/Storage/Tipo/Tela/Estoque/Família)
edf = enrich(df)

# Pequeno resumo
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Total de itens", len(edf))
with c2: st.metric("Com estoque", int((edf["total_saldo"] > 0).sum()))
with c3: st.metric("Notebooks", int((edf["familia"] == "NOTEBOOK").sum()))
with c4: st.metric("Desktops", int((edf["familia"] == "DESKTOP").sum()))
with c5: st.metric("Tablets/Smartphones", int(edf["familia"].isin(["TABLET","SMARTPHONE"]).sum()))

with st.expander("Ver amostra da base (enriquecida)"):
    tmp = edf.copy()
    # Oculta a pontuação de CPU na tabela de amostra
    if "cpu_score" in tmp.columns:
        tmp = tmp.drop(columns=["cpu_score"])
    st.dataframe(tmp.head(50), use_container_width=True)

st.markdown("---")
st.subheader("Buscar equivalentes")

sku = st.text_input("SKU base (ex.: NB3091 / DT0XXX / TB0XXX / SM0XXX)").strip()

if st.button("Encontrar compatíveis", type="primary", use_container_width=True):
    if not sku:
        st.error("Informe o SKU base.")
    else:
        try:
            # Usamos limites padrão internamente (sem mostrar na UI)
            recs = recommend(
                df,
                sku=sku,
                topn=100,
                only_status=None,
                min_screen=None,
                allow_family_override=False,   # mesma família por padrão
                cpu_cap=1.5,                   # até ~1.5 acima em CPU
                ram_factor_cap=2.0,            # até 2x RAM
                storage_factor_cap=2.0         # até 2x armazenamento
            )

            if recs.empty:
                st.warning("Nenhuma opção igual/superior encontrada com os critérios padrão.")
            else:
                # Remove a coluna de pontuação da visualização
                if "cpu_score" in recs.columns:
                    recs = recs.drop(columns=["cpu_score"])

                # Renomeia para português
                recs = recs.rename(columns={
                    "codigo": "Código",
                    "descricao": "Descrição",
                    "familia": "Família",
                    "ram_gb": "RAM (GB)",
                    "storage_gb": "Armazenamento (GB)",
                    "storage_type": "Tipo de armazenamento",
                    "gpu_dedicated": "GPU dedicada",
                    "screen_in": "Tela (pol)",
                    "total_saldo": "Saldo total",
                    "status": "Status",
                    "valor": "Valor",
                })

                st.success(f"Encontradas {len(recs)} opções.")
                st.dataframe(recs, use_container_width=True)

                st.download_button(
                    "Baixar CSV",
                    recs.to_csv(index=False).encode("utf-8"),
                    file_name=f"recomendacoes_{sku}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(str(e))
