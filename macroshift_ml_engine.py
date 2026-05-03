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

# Dicionário oficial do CNAE 2.0 (Foco nos setores mais afetados pela escala 6x1)
MAPA_CNAE = {
    'A': 'Agricultura e Pecuária', 'C': 'Indústria da Transformação',
    'F': 'Construção Civil', 'G': 'Comércio (Varejo/Atacado)',
    'I': 'Alojamento e Alimentação', 'N': 'Serviços Administrativos',
    'Q': 'Saúde Humana e Serviços Sociais'
}

print(f"{CIANO}{'='*75}\n MACROSHIFT BR // MOTOR DE MACHINE LEARNING E SIMULAÇÃO V2.0\n{'='*75}{RESETAR}")

class DigitalTwinEngine:
    def __init__(self):
        try:
            with open('db_config.json', 'r') as f:
                cfg = json.load(f)
            db_uri = f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/macro_shift_br_db"
            self.engine = create_engine(db_uri)
            self.le = LabelEncoder()
            self.modelo_ml = RandomForestRegressor(n_estimators=100, random_state=42)
            print(f"{VERDE}[+] Conexão com o Data Warehouse estabelecida.{RESETAR}")
        except Exception as e:
            print(f"{VERMELHO}[ERRO] Falha ao conectar no banco para ML: {e}{RESETAR}")
            self.engine = None

    def carregar_dados_treino(self):
        print(f"{AMARELO}[+] Extraindo matriz de conhecimento do PostgreSQL...{RESETAR}")
        query = "SELECT ano_mes, secao_cnae, salario_medio, horas_contratuais_media FROM tb_micro_caged"
        
        df = pd.read_sql(query, self.engine)
        
        # Filtra apenas os setores chave mapeados
        df = df[df['secao_cnae'].isin(MAPA_CNAE.keys())].copy()
        df['nome_setor'] = df['secao_cnae'].map(MAPA_CNAE)
        
        # Feature Engineering: Produtividade Horária Base (Valor gerado por hora)
        df['produtividade_hora_base'] = df['salario_medio'] / df['horas_contratuais_media']
        
        return df

    def treinar_modelo(self, df):
        print(f"{AMARELO}[+] Avaliando Algoritmo Preditivo (Validação Cruzada)...{RESETAR}")
        
        # O modelo aprende a prever a produtividade baseada no Setor e nas Horas Exigidas
        df['setor_encoded'] = self.le.fit_transform(df['secao_cnae'])
        
        X = df[['setor_encoded', 'horas_contratuais_media']]
        y = df['produtividade_hora_base']
        
        # O Segredo da Validação: Separando 20% dos dados para testar se o modelo não está decorando (overfitting)
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Treina com 80% para extrair as métricas de prova
        self.modelo_ml.fit(X_treino, y_treino)
        y_previsto = self.modelo_ml.predict(X_teste)
        
        # Extraindo as Métricas de Prova Estatística
        r2 = r2_score(y_teste, y_previsto)
        mae = mean_absolute_error(y_teste, y_previsto)
        rmse = np.sqrt(mean_squared_error(y_teste, y_previsto))
        
        print(f"{VERDE}[+] Cérebro de IA Validado! Métricas de Performance:{RESETAR}")
        print(f" -> R² (Acurácia de Variância): {r2:.4f}")
        print(f" -> MAE (Erro Médio Absoluto): R$ {mae:.2f}/hora")
        print(f" -> RMSE (Raiz do Erro Quadrático): R$ {rmse:.2f}/hora\n")
        
        # Retreina o modelo com 100% dos dados para que a simulação oficial tenha capacidade preditiva máxima
        self.modelo_ml.fit(X, y)
        
        # Calcula a média atual de cada setor para usar como linha de base (Baseline)
        baseline = df.groupby(['nome_setor', 'setor_encoded']).agg({
            'horas_contratuais_media': 'mean',
            'produtividade_hora_base': 'mean'
        }).reset_index()
        
        return baseline

    def simular_cenario(self, baseline, jornada_alvo, alpha_descanso, beta_automacao):
        """
        jornada_alvo: 40h (5x2) ou 36h (4x3)
        alpha_descanso: Ganho de produtividade por redução de fadiga (0.1 a 0.3)
        beta_automacao: Capacidade do setor absorver o choque com tecnologia (0.0 a 0.5)
        """
        cenario = baseline.copy()
        cenario['nova_jornada'] = jornada_alvo
        
        # 1. Previsão Base da IA (Como o modelo enxerga a produtividade nessa nova carga horária)
        X_simulado = cenario[['setor_encoded', 'nova_jornada']].copy()
        X_simulado = X_simulado.rename(columns={'nova_jornada': 'horas_contratuais_media'})
        
        previsao_ia = self.modelo_ml.predict(X_simulado)
        
        # 2. Motor Econômico (Aplicação da elasticidade da fadiga)
        # Formula: P_h = P_0 * (H_s / H_0)^(-alpha)
        cenario['nova_produtividade_h'] = previsao_ia * ((jornada_alvo / cenario['horas_contratuais_media']) ** (-alpha_descanso))
        
        # 3. Cálculo de Impacto em Vagas de Emprego (Delta E)
        # Formula: Delta E = ((H0 * P0) / (Hs * Ph) - 1) * (1 - beta)
        capacidade_antiga = cenario['horas_contratuais_media'] * cenario['produtividade_hora_base']
        capacidade_nova = jornada_alvo * cenario['nova_produtividade_h']
        
        # Ajustamos o Beta dinamicamente: Indústria tem alto beta (automação), Comércio tem baixo.
        ajuste_beta = np.where(cenario['nome_setor'].str.contains('Indústria'), beta_automacao * 1.5,
                      np.where(cenario['nome_setor'].str.contains('Comércio|Alojamento'), beta_automacao * 0.5, beta_automacao))
        
        cenario['necessidade_novas_vagas_pct'] = ((capacidade_antiga / capacidade_nova) - 1) * (1 - ajuste_beta) * 100
        
        return cenario

# --- EXECUÇÃO DO GÊMEO DIGITAL ---
if __name__ == "__main__":
    twin = DigitalTwinEngine()
    
    if twin.engine:
        # 1. Carrega e prepara dados
        dados_historicos = twin.carregar_dados_treino()
        
        # 2. Treina IA e exibe métricas de prova
        baseline_setores = twin.treinar_modelo(dados_historicos)
        
        # 3. Configuração de Parâmetros Macroeconômicos
        ALPHA = 0.20 # Elasticidade média (Ganho de 20% do potencial por redução de fadiga)
        BETA_MEDIO = 0.30 # 30% da perda de horas é absorvida por software/IA
        
        # 4. Rodando Simulações
        print(f"{CIANO}>>> SIMULANDO TRANSIÇÃO PARA ESCALA 5x2 (40 HORAS SEMANAIS) <<<{RESETAR}")
        sim_5x2 = twin.simular_cenario(baseline_setores, jornada_alvo=40, alpha_descanso=ALPHA, beta_automacao=BETA_MEDIO)
        
        resultado_5x2 = sim_5x2[['nome_setor', 'horas_contratuais_media', 'necessidade_novas_vagas_pct']].copy()
        resultado_5x2.columns = ['Setor da Economia', 'Horas Atuais (Média)', 'Impacto em Contratações (%)']
        print(resultado_5x2.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

        print(f"\n{CIANO}>>> SIMULANDO TRANSIÇÃO PARA ESCALA 4x3 (36 HORAS SEMANAIS) <<<{RESETAR}")
        sim_4x3 = twin.simular_cenario(baseline_setores, jornada_alvo=36, alpha_descanso=ALPHA, beta_automacao=BETA_MEDIO)
        
        resultado_4x3 = sim_4x3[['nome_setor', 'horas_contratuais_media', 'necessidade_novas_vagas_pct']].copy()
        resultado_4x3.columns = ['Setor da Economia', 'Horas Atuais (Média)', 'Impacto em Contratações (%)']
        print(resultado_4x3.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        
        print(f"\n{VERDE}[SUCESSO] Simulação MacroShift BR concluída.{RESETAR}")