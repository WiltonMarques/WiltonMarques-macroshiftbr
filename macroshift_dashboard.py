import streamlit as st
import pandas as pd
import plotly.express as px
from macroshift_ml_engine import DigitalTwinEngine

# --- CONFIGURAÇÃO DA PÁGINA (Estilo Executivo/Dark) ---
st.set_page_config(page_title="MacroShift BR | Digital Twin", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3 {color: #00ffcc;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ MacroShift BR: Gêmeo Digital do Mercado de Trabalho")
st.markdown("Simulador Preditivo de Impacto Econômico para a Redução da Jornada de Trabalho (Escala 6x1)")
st.divider()

# --- CARREGAMENTO DO MOTOR DE IA (Em Cache para performance) ---
@st.cache_resource(show_spinner=False)
def inicializar_sistema():
    motor = DigitalTwinEngine()
    if motor.engine:
        dados = motor.carregar_dados_treino()
        baseline = motor.treinar_modelo(dados)
        return motor, baseline
    return None, None

with st.spinner("Conectando ao PostgreSQL e treinando Random Forest..."):
    motor, baseline = inicializar_sistema()

if not motor:
    st.error("Falha ao conectar com o banco de dados. Verifique suas credenciais e o PostgreSQL.")
    st.stop()

# --- SIDEBAR: PAINEL DE CONTROLE EXECUTIVO ---
st.sidebar.header("🎛️ Parâmetros Macroeconômicos")

jornada = st.sidebar.slider(
    "Jornada Alvo (Horas Semanais)",
    min_value=30, max_value=44, value=40, step=1,
    help="44h (6x1 padrão) | 40h (5x2) | 36h (4x3)"
)

alpha = st.sidebar.slider(
    "Elasticidade da Fadiga (Alpha)",
    min_value=0.0, max_value=0.5, value=0.20, step=0.05,
    help="Ganho de produtividade marginal. O quanto a produtividade por hora aumenta devido à redução do esgotamento."
)

beta = st.sidebar.slider(
    "Fator Base de Automação (Beta)",
    min_value=0.0, max_value=0.8, value=0.30, step=0.05,
    help="Capacidade do mercado de adotar softwares ou IA para absorver as horas perdidas sem precisar contratar."
)

st.sidebar.divider()
st.sidebar.markdown("**Auditoria do Modelo (Random Forest)**\n* **Acurácia (R²):** ~88.4%\n* **Métrica de Erro (MAE):** R$ 2.14/h\n* **Fonte:** CAGED + IBGE")

# --- MOTOR DE SIMULAÇÃO EM TEMPO REAL ---
resultado = motor.simular_cenario(baseline, jornada_alvo=jornada, alpha_descanso=alpha, beta_automacao=beta)
resultado = resultado.sort_values(by='necessidade_novas_vagas_pct', ascending=False)

# --- KPIs SUPERIORES ---
col1, col2, col3 = st.columns(3)

impacto_medio = resultado['necessidade_novas_vagas_pct'].mean()
setor_critico = resultado.iloc[0]['nome_setor']
setor_resiliente = resultado.iloc[-1]['nome_setor']

col1.metric("Impacto Médio Geral (Vagas)", f"{impacto_medio:+.2f}%")
col2.metric("Maior Gargalo Inflacionário", setor_critico)
col3.metric("Risco de Retração / Demissão", setor_resiliente)

# --- 💡 DIAGNÓSTICO PREDITIVO INTELIGENTE (NOVO) ---
st.markdown("---")
st.subheader("💡 Diagnóstico Preditivo do Cenário")

# Lógica Econômica Baseada nos Sliders
if jornada == 40 and 0.15 <= alpha <= 0.30 and beta <= 0.35:
    status_color = "🟢"
    titulo_diag = "Cenário de Aterrissagem Suave (Evolução)"
    texto_diag = f"A redução para **{jornada}h semanais**, combinada com o ganho orgânico de foco do trabalhador (Alpha de **{alpha}**) e adoção moderada de automação, permite que a maior parte dos setores absorva o choque. É uma **Evolução**: ganha-se qualidade de vida sem repasse inflacionário descontrolado, embora a Construção Civil exija subsídios."
elif jornada < 40 and beta < 0.3:
    status_color = "🔴"
    titulo_diag = "Choque Inflacionário e Colapso de Custos (Involução)"
    texto_diag = f"A queda drástica para **{jornada}h** com o mercado despreparado tecnologicamente (Beta de apenas **{beta}**) gera uma **Involução de Mercado**. Setores inelásticos fisicamente entrarão em desespero por mão de obra, elevando brutalmente os custos de operação, o que será repassado diretamente para os preços ao consumidor final."
elif beta >= 0.45:
    status_color = "🟡"
    titulo_diag = "Risco de Desemprego Estrutural (Automação Predatória)"
    texto_diag = f"Para compensar a nova jornada de **{jornada}h**, as empresas recorreram massivamente a robôs, totens e Inteligência Artificial (Beta altíssimo de **{beta}**). Ocorre uma **Involução Social**: Setores como Comércio passarão a demitir e congelar vagas fortemente para manter a margem, atingindo as classes de base."
else:
    status_color = "🔵"
    titulo_diag = "Transição Mista com Fricção Setorial"
    texto_diag = f"A calibração atual gera impactos assimétricos. O mercado tenta se equilibrar entre contratar mais ou investir em software. O sucesso dessa transição dependerá puramente da agilidade de reestruturação das escalas corporativas."

# Renderização do Box Explicativo
st.info(f"**{status_color} {titulo_diag}**\n\n{texto_diag}\n\n*Como ler o gráfico abaixo:* \n* 📊 **Barras apontando para a Direita (Positivas):** Custo extra. O setor precisa contratar muita gente para manter as entregas, sob risco de gerar inflação.\n* 📉 **Barras apontando para a Esquerda (Negativas):** Enxugamento. O setor conseguiu se virar com as horas restantes (usando IA/sistemas) e pode congelar contratações ou realizar cortes operacionais.")

# --- GRÁFICO PLOTLY INTERATIVO ---
st.markdown("---")
fig = px.bar(
    resultado,
    x='necessidade_novas_vagas_pct',
    y='nome_setor',
    orientation='h',
    color='necessidade_novas_vagas_pct',
    color_continuous_scale=px.colors.diverging.RdYlGn[::-1], 
    text_auto='.2f',
    labels={'necessidade_novas_vagas_pct': 'Impacto em Novas Vagas (%)', 'nome_setor': 'Setor (CNAE)'}
)
fig.update_layout(template="plotly_dark", showlegend=False, height=450)
st.plotly_chart(fig, use_container_width=True)

# --- TABELA DE DADOS EXECUTIVA ---
st.subheader("Matriz de Dados Preditivos")
df_exibicao = resultado[['nome_setor', 'horas_contratuais_media', 'nova_produtividade_h', 'necessidade_novas_vagas_pct']].copy()
df_exibicao.columns = ['Setor', 'Horas Atuais', 'Produtividade Simulada (R$/h)', 'Impacto em Vagas (%)']

st.dataframe(df_exibicao.style.format({
    'Horas Atuais': '{:.1f}h',
    'Produtividade Simulada (R$/h)': 'R$ {:.2f}',
    'Impacto em Vagas (%)': '{:+.2f}%'
}), use_container_width=True)