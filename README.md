# Notebook Matcher (igual ou superior)

Ferramenta para localizar, a partir de um SKU de notebook, **todos os notebooks iguais ou superiores** (CPU, RAM, armazenamento e GPU), **com saldo**. Inclui um app Streamlit e um módulo Python reutilizável.

## Como usar (local)

```bash
# 1) criar venv (opcional)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) instalar dependências
pip install -r requirements.txt

# 3) rodar o app
streamlit run app.py
```

Coloque a planilha `produtos_final.xlsx` em `data/` ou faça upload no próprio app.

## Estrutura do repositório

```
notebook-matcher/
  app.py                 # app Streamlit (UI)
  notebook_matcher.py   # funções de parsing e recomendação
  requirements.txt
  README.md
  data/produtos_final.xlsx (opcional)
  tools/check_update.py  # utilitário para avisar quando há atualização no repositório
```

## Lógica de compatibilidade

- **CPU**: mapeamos e pontuamos Intel (i3/i5/i7/i9, Core Ultra 5/7/9), AMD Ryzen (3/5/7/9) e Apple (M1..M4, Pro/Max/Ultra).
- **RAM** e **Armazenamento**: comparamos por GB (regex extrai da planilha/descrição).
- **GPU dedicada**: se houver evidência de RTX/GTX/Radeon/“PL VIDEO”, considera **superior** a integrada.
- **Tela**: não é obrigatória para compatibilidade; conta como desempate bônus.
- **Saldo**: filtra somente itens com `saldo_novo+saldo_seminovo+saldo_sustentacao > 0`.

> Observação: além das colunas dedicadas, a ferramenta também extrai dados da `descricao` para lidar com cadastros incompletos.

## Git - fluxo recomendado

1. **Criar repositório no GitHub** (vazio).
2. No terminal:

```bash
git init
git add .
git commit -m "feat: primeira versão do matcher"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/notebook-matcher.git
git push -u origin main
```

3. **Clonar na empresa**:

```bash
git clone https://github.com/SEU_USUARIO/notebook-matcher.git
```

4. **Receber avisos de atualização**: rode o utilitário abaixo em cada início de uso:

```bash
python tools/check_update.py
# se houver updates, ele sugere:
git pull --ff-only
```

Ou automatize com um script `.bat`/`.sh` que chama `git fetch` e `git pull` antes de abrir o app.

## Roadmap de melhorias

- Regras específicas por família (Latitude vs ThinkPad etc.).
- Pesos configuráveis por critério.
- Exportar relatório em PDF.
- API REST (FastAPI) para integrar ao ERP.
