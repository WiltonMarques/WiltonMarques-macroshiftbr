import pandas as pd
import json
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURAÇÃO VISUAL MACROSHIFT ---
VERDE, VERMELHO, CIANO, AMARELO, RESETAR = '\033[92m', '\033[91m', '\033[96m', '\033[93m', '\033[0m'

print(f"{CIANO}{'='*80}\n MACROSHIFT BR // RADAR NLP DE TRANSIÇÃO LABORAL E AUTOMAÇÃO V1.0\n{'='*80}{RESETAR}")

class MacroShiftRadar:
    def __init__(self):
        self._preparar_nltk()
        self.engine = self._conectar_banco()
        
        self.stop_words = set(stopwords.words('portuguese'))
        
        jargoes = {
            'altera', 'lei', 'dispoe', 'sobre', 'art', 'inciso', 'redacao', 'estabelece', 'providencias',
            'parecer', 'relator', 'relatora', 'comissao', 'requer', 'aprovacao', 'substitutivo', 'adotado',
            'exarado', 'deputado', 'deputada', 'nacional', 'realizacao', 'providencia', 'representante',
            'projeto', 'proposicao', 'institui', 'cria', 'acrescenta', 'paragrafo', 'federal', 'membro', 'mesa',
            'solicita', 'informacoes', 'ministerio', 'ministro', 'voto', 'votos', 'favoravel', 'contrario',
            'tramitacao', 'urgencia', 'apresentacao', 'ementa', 'texto', 'outras'
        }
        self.stop_words.update(jargoes)

    def _preparar_nltk(self):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

    def _conectar_banco(self, arquivo_config="db_config.json"):
        try:
            with open(arquivo_config, 'r') as f:
                cfg = json.load(f)
            db_uri = f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
            return create_engine(db_uri)
        except Exception as e:
            print(f"{VERMELHO}[ERRO] Falha ao conectar no banco para NLP: {e}{RESETAR}")
            return None

    def extrair_ementas(self):
        print(f"{AMARELO}[+] Fase 1: Ingestão de Texto Bruto (Filtrando Substantivos e Duplicatas)...{RESETAR}")
        query = """
        SELECT id, "siglaTipo", numero, ano, ementa 
        FROM tb_camara_proposicoes 
        WHERE ementa IS NOT NULL 
        AND "siglaTipo" IN ('PL', 'PEC', 'PLP', 'MPV')
        """
        try:
            df = pd.read_sql(query, self.engine)
            df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
            print(f"{VERDE}[OK] {len(df)} projetos de alta relevância extraídos e desduplicados.{RESETAR}")
            return df
        except Exception as e:
            print(f"{VERMELHO}[X] Erro ao extrair ementas. Verifique a tabela: {e}{RESETAR}")
            return None

    def limpar_texto(self, texto):
        if not isinstance(texto, str):
            return ""
        texto = texto.lower()
        texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
        texto = re.sub(r'[^a-z]', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        tokens = [palavra for palavra in texto.split() if palavra not in self.stop_words and len(palavra) > 3]
        return " ".join(tokens)

    def processar_matriz_nlp(self, df, num_clusters=4):
        print(f"{AMARELO}[+] Fase 2 e 3: Higienização e Construção da Matriz Vetorial (TF-IDF)...{RESETAR}")
        df['ementa_limpa'] = df['ementa'].apply(self.limpar_texto)
        vetorizador = TfidfVectorizer(max_features=1000, max_df=0.85, min_df=2)
        matriz_tfidf = vetorizador.fit_transform(df['ementa_limpa'])
        return df, vetorizador, matriz_tfidf

    def calcular_macro_score(self, df, vetorizador, matriz_tfidf):
        print(f"\n{AMARELO}[+] Fase 4: Aplicando Similaridade por Cosseno (Macro-Score Laboral)...{RESETAR}")
        
        # O NOVO MANIFESTO: Focado 100% no motor de simulação do MacroShift BR
        texto_ouro = "mercado trabalho jornada automacao inteligencia artificial transicao laboral produtividade escala emprego desemprego reducao horas trabalhador"
        texto_ouro_limpo = self.limpar_texto(texto_ouro)
        
        vetor_ouro = vetorizador.transform([texto_ouro_limpo])
        scores = cosine_similarity(matriz_tfidf, vetor_ouro).flatten()
        df['macro_score'] = scores * 100 
        
        df_ranking = df.sort_values(by='macro_score', ascending=False)
        
        print(f"{CIANO}\n>>> TOP 5 PROJETOS: IMPACTO DIRETO NA ESCALA E AUTOMAÇÃO DO TRABALHO <<<{RESETAR}")
        top_5 = df_ranking.head(5)
        for index, row in top_5.iterrows():
            if row['macro_score'] > 0:
                print(f" {VERDE}Score: {row['macro_score']:.1f}%{RESETAR} | {row['siglaTipo']} {row['numero']}/{row['ano']}")
                print(f" Resumo: {row['ementa'][:120]}...\n")
        
        return df_ranking

if __name__ == "__main__":
    radar = MacroShiftRadar()
    if radar.engine:
        df_leis = radar.extrair_ementas()
        if df_leis is not None and not df_leis.empty:
            df_classificado, modelo_tfidf, matriz = radar.processar_matriz_nlp(df_leis)
            df_final = radar.calcular_macro_score(df_classificado, modelo_tfidf, matriz)
            print(f"{CIANO}[+] Radar MacroShift BR finalizado. Monitoramento estratégico concluído.{RESETAR}")