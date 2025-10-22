import streamlit as st
import pandas as pd
from notebook_matcher import enrich, recommend, find_batteries_for_notebook

# =========================================
# Config e cabeçalho
# =========================================
st.set_page_config(page_title="ArkMatch", layout="wide")

st.markdown("""
<style>
header [data-testid="stHeaderActionElements"]{display:none!important;}
#MainMenu, footer, div[data-testid="stToolbar"]{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    display:flex; flex-direction:column; align-items:center; text-align:center;
    margin-top:-10px; margin-bottom:40px;">
  <div style="
      background-color:#101010; padding:25px 60px; border-radius:22px;
      display:flex; flex-direction:column; align-items:center; justify-content:center;
      box-shadow:0 0 25px rgba(255,255,255,0.05);">
    <img src="https://arknet.arklok.com.br/assets/img/logos/logo-arklok.png"
         width="130" style="margin-bottom:10px; opacity:0.95;">
    <h1 style="font-size:50px; font-weight:800; letter-spacing:-1px; margin:0; color:#fff;">
      ArkMatch
    </h1>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================================
# Carrega a base integrada
# =========================================
DATA_PATH = "data/produtos_final.xlsx"
try:
    df = pd.read_excel(DATA_PATH)
except Exception:
    st.error(f"❌ Não encontrei a base em {DATA_PATH}")
    st.stop()

edf = enrich(df)

# =========================================
# Métricas de topo
# =========================================
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Total de itens", len(edf))
with c2: st.metric("Com estoque", int((edf["total_saldo"] > 0).sum()))
with c3: st.metric("Notebooks", int((edf["familia"] == "NOTEBOOK").sum()))
with c4: st.metric("Desktops", int((edf["familia"] == "DESKTOP").sum()))
with c5: st.metric("Tablets/Smartphones", int(edf["familia"].isin(["TABLET","SMARTPHONE"]).sum()))
st.markdown("---")

# =========================================
# Busca principal
# =========================================
st.subheader("Buscar equivalentes")
sku = st.text_input("SKU base (ex.: NB3091 / DT0XXX / TB0XXX / SM0XXX)").strip()
show_diag = st.checkbox("Mostrar diagnóstico", value=False)

if st.button("Encontrar compatíveis", type="primary", use_container_width=True):
    if not sku:
        st.error("Informe o SKU base."); st.stop()

    base_rows = edf[edf["codigo"].astype(str).str.upper() == sku.upper()]
    if base_rows.empty:
        st.error(f"SKU {sku} não encontrado."); st.stop()

    base_row = base_rows.iloc[0]

    # ---------- diagnóstico opcional ----------
    if show_diag:
        diag = pd.DataFrame([{
            "Código": base_row["codigo"],
            "Descrição": base_row["descricao"],
            "Família": base_row["familia"],
            "RAM (GB)": base_row["ram_gb"],
            "Armazenamento (GB)": base_row["storage_gb"],
            "Tipo de armazenamento": base_row["storage_type"],
            "GPU dedicada": base_row["gpu_dedicated"],
            "Tela (pol)": base_row["screen_in"],
            "Saldo total": base_row["total_saldo"],
            "Status": base_row.get("status", ""),
            "Fabricante": base_row.get("fabricante",""),
            "Modelo base (p/ bateria)": base_row.get("modelo_base","")
        }])
        st.info("Interpretação do SKU base:")
        st.dataframe(diag, use_container_width=True)

    # ---------- resultados equivalentes ----------
    found = recommend(df, sku, topn=30)
    if found.empty:
        st.warning("Nenhuma opção igual/superior encontrada com os critérios atuais.")
    else:
        show = found.drop(columns=["cpu_score","cpu_class"], errors="ignore")
        st.success(f"✅ {len(show)} opções compatíveis encontradas!")
        st.dataframe(show, use_container_width=True)

        csv = show.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Baixar CSV",
            data=csv,
            file_name=f"recomendacoes_{sku}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ---------- baterias compatíveis (se for notebook) ----------
    if str(base_row.get("familia","")).upper() == "NOTEBOOK":
        st.markdown("### 🔋 Baterias compatíveis para este notebook")
        batt = find_batteries_for_notebook(df, base_row, topn=20)
        if batt.empty:
            st.info("Nenhuma bateria específica encontrada para este modelo na base.")
        else:
            st.dataframe(batt, use_container_width=True)
            csvb = batt.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Baixar lista de baterias",
                data=csvb,
                file_name=f"baterias_{sku}.csv",
                mime="text/csv",
                use_container_width=True
            )
