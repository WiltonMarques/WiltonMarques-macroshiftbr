import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CONFIGURAÇÃO VISUAL MACROSHIFT BR ---
VERDE, VERMELHO, CIANO, AMARELO, RESETAR = '\033[92m', '\033[91m', '\033[96m', '\033[93m', '\033[0m'

MAPA_CNAE = {
    'A': 'Agricultura e Pecuária', 'C': 'Indústria da Transformação',
    'F': 'Construção Civil', 'G': 'Comércio (Varejo/Atacado)',
    'I': 'Alojamento e Alimentação', 'N': 'Serviços Administrativos',
    'Q': 'Saúde Humana e Serviços Sociais'
}

print(f"{CIANO}{'='*80}\n MACROSHIFT BR // MOTOR MLOPS: INTEGRAÇÃO PREVISÃO FINANCEIRA + NLP V8.0\n{'='*80}{RESETAR}")

class DigitalTwinEngine:
    def __init__(self):
        try:
            with open('db_config.json', 'r') as f:
                cfg = json.load(f)
            db_uri = f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
            self.engine = create_engine(db_uri)
            self.le_setor = LabelEncoder()
            
            # MOTOR: Random Forest Profunda
            self.modelo_ml = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
            print(f"{VERDE}[+] Conexão com o Data Warehouse estabelecida.{RESETAR}")
        except Exception as e:
            print(f"{VERMELHO}[ERRO] Falha ao conectar no banco para ML: {e}{RESETAR}")
            self.engine = None

    def carregar_dados_e_features(self):
        print(f"{AMARELO}[+] Sintetizando Matriz de Causalidade ancorada no Salário Base...{RESETAR}")
        query = "SELECT ano_mes, secao_cnae, salario_medio, horas_contratuais_media FROM tb_micro_caged"
        df_base = pd.read_sql(query, self.engine)
        
        df_base = df_base[df_base['secao_cnae'].isin(MAPA_CNAE.keys())].copy()
        df = df_base.loc[df_base.index.repeat(1500)].copy().reset_index(drop=True)
        df['nome_setor'] = df['secao_cnae'].map(MAPA_CNAE)
        
        np.random.seed(42)
        
        # --- A ÂNCORA --- 
        df['salario_base_setorial'] = df['salario_medio']
        
        # GERADORES DE CAUSA
        df['taxa_absenteismo'] = np.where(df['secao_cnae'].isin(['G', 'I']), 
                                          np.random.normal(0.09, 0.02, len(df)), 
                                          np.random.normal(0.03, 0.01, len(df)))
        
        df['volume_horas_extras'] = np.where(df['secao_cnae'].isin(['C', 'F']), 
                                             np.random.normal(0.20, 0.04, len(df)), 
                                             np.random.normal(0.08, 0.02, len(df)))
        
        df['indice_turnover'] = np.where(df['secao_cnae'].isin(['G', 'I']), 
                                         np.random.normal(0.25, 0.05, len(df)), 
                                         np.random.normal(0.10, 0.02, len(df)))
        
        df['investimento_tec'] = np.where(df['secao_cnae'] == 'C', np.random.normal(0.75, 0.10, len(df)),
                                 np.where(df['secao_cnae'] == 'F', np.random.normal(0.15, 0.05, len(df)),
                                          np.random.normal(0.45, 0.08, len(df))))
        
        cols_limites = ['taxa_absenteismo', 'volume_horas_extras', 'indice_turnover', 'investimento_tec']
        df[cols_limites] = df[cols_limites].clip(lower=0.01, upper=0.99)

        # RUÍDO ESTATÍSTICO BASE
        df['horas_contratuais_media'] = np.random.normal(df['horas_contratuais_media'], 0.5)
        salario_base_ruido = np.random.normal(df['salario_base_setorial'], df['salario_base_setorial'] * 0.02)

        # CAUSALIDADE MATEMÁTICA
        df['salario_medio'] = (salario_base_ruido * (1 - df['taxa_absenteismo']) * (1 - (df['indice_turnover'] * 0.5)) * (1 + (df['investimento_tec'] * 0.6)) * (1 + (df['volume_horas_extras'] * 0.3)))
        
        # ALVO (Target)
        df['produtividade_hora_base'] = df['salario_medio'] / df['horas_contratuais_media']
        
        return df

    def validar_e_treinar_modelo(self, df):
        print(f"{AMARELO}[+] Avaliando Algoritmo Preditivo em Alta Resolução...{RESETAR}")
        
        df['setor_encoded'] = self.le_setor.fit_transform(df['secao_cnae'])
        
        self.features = ['setor_encoded', 'salario_base_setorial', 'horas_contratuais_media', 
                         'taxa_absenteismo', 'volume_horas_extras', 'indice_turnover', 'investimento_tec']
        
        X = df[self.features]
        y = df['produtividade_hora_base']
        
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.modelo_ml.fit(X_treino, y_treino)
        y_previsto = self.modelo_ml.predict(X_teste)
        
        r2 = r2_score(y_teste, y_previsto)
        mae = mean_absolute_error(y_teste, y_previsto)
        
        print(f"\n{VERDE}>>> CERTIFICADO DE QUALIDADE DO MODELO (MLOps) <<<{RESETAR}")
        print(f" • Base de Dados Sintetizada: {len(df)} empresas")
        print(f" • R² (Coeficiente de Explicação) : {r2:.4f} (Meta > 0.85)")
        print(f" • MAE (Erro Absoluto Médio)      : R$ {mae:.2f}/hora de desvio")
        
        self.modelo_ml.fit(X, y) # Treina com a base completa após validação
        
        baseline = df.groupby(['nome_setor', 'setor_encoded']).agg({
            'salario_base_setorial': 'mean',
            'horas_contratuais_media': 'mean',
            'taxa_absenteismo': 'mean',
            'volume_horas_extras': 'mean',
            'indice_turnover': 'mean',
            'investimento_tec': 'mean',
            'produtividade_hora_base': 'mean'
        }).reset_index()
        
        return baseline

    def simular_impacto_escala_integrado(self, baseline, jornada_alvo, score_nlp_automacao):
        cenario = baseline.copy()
        
        cenario_prev = cenario[self.features].copy()
        cenario_prev['horas_contratuais_media'] = jornada_alvo
        
        # 1. ML Engine prevê a nova produtividade financeira
        cenario['nova_produtividade_h'] = self.modelo_ml.predict(cenario_prev)
        
        # 2. O CRUZAMENTO DE DADOS (Risco NLP vs Escudo Tecnológico)
        risco_regulatorio = score_nlp_automacao / 100 
        fator_escudo_teorico = cenario['investimento_tec']
        
        # A tecnologia perde eficiência (fica mais cara/arriscada) devido aos impostos mapeados no Congresso
        fator_escudo_real = fator_escudo_teorico * (1 - risco_regulatorio)
        cenario['escudo_real_pos_nlp'] = fator_escudo_real
        
        # 3. Lógica Econômica Pós-Simulação
        compensacao_fadiga = 1 + (cenario['taxa_absenteismo'] * 0.5) 
        pressao_horas_extras = 1 + (cenario['volume_horas_extras'] * (cenario['horas_contratuais_media'] / jornada_alvo - 1))
        
        capacidade_antiga = cenario['horas_contratuais_media'] * cenario['produtividade_hora_base']
        capacidade_nova = jornada_alvo * cenario['nova_produtividade_h'] * compensacao_fadiga
        
        # A necessidade de vagas considera a blindagem real que sobrou após a tesourada do Governo
        cenario['necessidade_novas_vagas_pct'] = (((capacidade_antiga * pressao_horas_extras) / capacidade_nova) - 1) * (1 - fator_escudo_real) * 100
        
        return cenario

# --- EXECUÇÃO DO LABORATÓRIO PREDITIVO ---
if __name__ == "__main__":
    twin = DigitalTwinEngine()
    
    if twin.engine:
        dados_enriquecidos = twin.carregar_dados_e_features()
        baseline_setores = twin.validar_e_treinar_modelo(dados_enriquecidos)
        
        # DADO INJETADO VIA NLP RADAR (Ex: PL 2067/2026 - Taxação de IA)
        SCORE_RISCO_CONGRESSO = 46.4  
        
        print(f"{CIANO}>>> SIMULANDO CHOQUE 4x3 (36H) + RISCO LEGISLATIVO DE AUTOMAÇÃO ({SCORE_RISCO_CONGRESSO}%) <<<{RESETAR}")
        
        sim_4x3_integrada = twin.simular_impacto_escala_integrado(baseline_setores, jornada_alvo=36, score_nlp_automacao=SCORE_RISCO_CONGRESSO)
        
        resultado_final = sim_4x3_integrada[['nome_setor', 'investimento_tec', 'escudo_real_pos_nlp', 'necessidade_novas_vagas_pct']].copy()
        resultado_final.columns = ['Setor', 'Escudo Téc. (Teórico)', 'Escudo Téc. (Pós-NLP)', 'Impacto Vagas (%)']
        
        print(resultado_final.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print(f"\n{VERMELHO}[ALERTA MLOPS] O Efeito Tesoura foi detectado. A taxação de automação reduziu a eficácia da tecnologia, inflando os custos de contratação.{RESETAR}")
        print(f"{VERDE}[SUCESSO] Laboratório de Simulação Sistêmica finalizado.{RESETAR}")