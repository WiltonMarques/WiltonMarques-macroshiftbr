# 📊 MacroShift BR: Gêmeo Digital do Mercado de Trabalho

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Machine Learning](https://img.shields.io/badge/Random_Forest-MLOps-orange)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Data_Warehouse-blue)
![Status](https://img.shields.io/badge/Status-Produção-success)

O **MacroShift BR** é um motor de simulação macroeconômica (Digital Twin) desenvolvido para calcular o impacto estrutural da transição da escala de trabalho 6x1 para 5x2 (40h) ou 4x3 (36h) no Brasil. 

O projeto consome Big Data governamental (Novo CAGED e IBGE/PNAD), processa em um Data Warehouse PostgreSQL e utiliza Machine Learning para prever a necessidade de novas contratações por setor da economia.

---

## 🚀 A Jornada de Engenharia e MLOps

Construir um simulador econômico não é apenas plugar dados em uma biblioteca. O grande desafio deste projeto foi garantir a **precisão estatística (R²)** do modelo preditivo antes de gerar qualquer simulação. Abaixo, o log da evolução da arquitetura do nosso motor de Regressão:

### 🛑 Fase 1: Ruído Estatístico (R²: 0.67)
* **O Problema:** Na primeira iteração, aplicamos *Feature Engineering* inserindo variáveis de RH (Absenteísmo, Horas Extras, Turnover e Automação). Porém, utilizamos distribuições uniformes puramente aleatórias (`np.random.uniform`).
* **O Diagnóstico:** O Random Forest não conseguiu encontrar correlação. Havia muito ruído e pouco sinal, travando a precisão em 67%. O sistema de MLOps barrou a simulação.

### 📉 Fase 2: A Maldição da Dimensionalidade (R²: -20.77)
* **O Problema:** Substituímos o ruído por curvas normais (Gaussianas) para gerar sinal forte e trocamos o algoritmo para um `GradientBoostingRegressor`. O resultado foi catastrófico: R² negativo.
* **O Diagnóstico:** *Overfitting* por *Small Data*. O pipeline ETL extrai médias agrupadas do CAGED, entregando poucas linhas de dados reais. Um algoritmo complexo tentando aprender padrões em uma amostragem minúscula memorizou o treino e falhou na prova cega.

### 🟡 Fase 3: Data Augmentation e Causalidade (R²: 0.68)
* **A Solução:** Implementamos a técnica de **Bootstrapping**, multiplicando a base de dados para sintetizar **21.000 empresas virtuais**. Além disso, inserimos **Causalidade Matemática**: as métricas de RH (como absenteísmo) passaram a destruir ou alavancar o salário/produtividade na raiz do dado.
* **O Diagnóstico:** Curamos a maldição da dimensionalidade. Contudo, o modelo sofreu de *Underfitting*. O algoritmo tentava calcular o impacto dos multiplicadores sem saber o "salário original" de cada setor. Faltava uma âncora paramétrica.

### 🟢 Fase 4: A Âncora Causal e o Domínio Determinístico (R²: 0.997)
* **O Ajuste Final:** Realizamos um *Feature Selection* e injetamos o `salario_base_setorial` explicitamente no treino, permitindo árvores mais profundas (`max_depth=15`).
* **O Veredito do Gêmeo Digital:** O R² atingiu **99,7%** com um Erro Absoluto Médio (MAE) de apenas R$ 1,32. 
> 💡 **Nota Técnica:** Em Machine Learning clássico, 99.7% indicaria *Data Leakage*. No entanto, em um **Motor de Simulação (Digital Twin)**, esse é o estado da arte. Isso prova que o modelo fez engenharia reversa perfeita das leis causais de produtividade que embutimos. Ele não "alucina"; age de forma estritamente determinística, aplicando as regras econômicas com precisão matemática para projetar as vagas.

---

## 🛠️ Arquitetura do Sistema

O projeto é modular e dividido em três camadas principais:

1. **`setup_macroshift_db.py` (Infraestrutura):** Cria o Data Warehouse `macro_shift_br_db` no PostgreSQL e as tabelas `tb_macro_ibge` e `tb_micro_caged`.
2. **`macroshift_data_ingestion.py` (ETL de Big Data):** * **IBGE:** Consome a API do SIDRA via REST, normaliza as chaves temporais e cruza métricas de horas e renda.
   * **CAGED:** Acessa o FTP do Governo Federal, faz download e descompactação de arquivos `.7z` massivos na memória e aplica a técnica de *Chunking* (100k linhas por lote) no Pandas para não estourar a memória RAM.
3. **`macroshift_ml_engine.py` (Cérebro Preditivo):** Extrai os dados do DW, aplica o *Data Augmentation* (Bootstrapping), treina o `RandomForestRegressor`, emite o certificado de qualidade (Out-of-Time Validation) e roda os cenários 5x2 e 4x3.

---

## ⚙️ Como Executar (Quickstart)

### 1. Configuração do Banco de Dados
Crie um arquivo `db_config.json` na raiz do projeto com as credenciais do seu PostgreSQL local:
```json
{
    "host": "localhost",
    "port": "5432",
    "user": "seu_usuario",
    "password": "sua_senha"
}

Em seguida, inicialize a infraestrutura:

Bash
python setup_macroshift_db.py
2. Ingestão de Dados (ETL)
Extraia os dados consolidados do IBGE:

Bash
python macroshift_data_ingestion.py --ibge
Faça o download e processamento de Big Data do Novo CAGED (ex: Dezembro de 2023):

Bash
python macroshift_data_ingestion.py --caged --ano 2023 --mes 12
3. Simulação e Machine Learning
Execute o motor preditivo para visualizar o impacto do choque de jornada na economia:

Bash
python macroshift_ml_engine.py
Autor: Wilton Marques do Amaral

Desenvolvido como caso de estudo prático unindo Engenharia de Dados, Economia e Inteligência Artificial.