import pandas as pd
import requests
import py7zr
import os
import urllib.request
import json
import argparse
from sqlalchemy import create_engine

# --- CONFIGURAÇÃO VISUAL MACROSHIFT BR ---
VERDE, VERMELHO, CIANO, AMARELO, RESETAR = '\033[92m', '\033[91m', '\033[96m', '\033[93m', '\033[0m'

class IBGEConnector:
    """ Tubulação para a API do SIDRA (IBGE) - PNAD Contínua """
    
    def __init__(self):
        self.base_url = "https://apisidra.ibge.gov.br/values"
        
    def extrair_dados_tabela(self, tabela, metrica_nome, periodos="all"):
        url = f"{self.base_url}/t/{tabela}/n1/all/v/all/p/{periodos}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            dados_raw = response.json()
            
            cabeçalho = dados_raw[0]
            df = pd.DataFrame(dados_raw[1:])
            
            # 1. BUSCA DINÂMICA DE TEMPO: Detecta se a tabela é Anual ou Trimestral
            chave_tempo = None
            for key, value in cabeçalho.items():
                val_lower = str(value).lower()
                if ('trimestre' in val_lower or 'ano' in val_lower) and 'código' in val_lower:
                    chave_tempo = key
                    break
                    
            if not chave_tempo:
                raise ValueError(f"Coluna de Tempo não encontrada no cabeçalho: {cabeçalho}")
            
            df_extraido = pd.DataFrame({
                'periodo_bruto': df[chave_tempo].astype(str),
                metrica_nome: pd.to_numeric(df['V'], errors='coerce')
            })
            
            # 2. NORMALIZAÇÃO TEMPORAL: 
            # Se for '202301' (Trimestre), vira '2023'. Se for '2023' (Ano), continua '2023'.
            df_extraido['ano'] = df_extraido['periodo_bruto'].str[:4]
            
            # 3. AGRUPAMENTO: Tira a média anual para padronizar os dados
            df_agrupado = df_extraido.groupby('ano')[metrica_nome].mean().reset_index()
            
            return df_agrupado
            
        except Exception as e:
            print(f"{VERMELHO}[IBGE ERRO] Falha ao ler a Tabela {tabela}: {e}{RESETAR}")
            return None

    def extrair_horas_e_renda(self):
        print(f"{AMARELO}[IBGE] Iniciando extração da PNAD Contínua (Tabelas 5436 e 9443)...{RESETAR}")
        
        # Puxamos um histórico maior para garantir cruzamento de anos completos
        df_renda = self.extrair_dados_tabela(5436, 'renda_media_real', 'last 12') # 12 trimestres (3 anos)
        df_horas = self.extrair_dados_tabela(9443, 'horas_semanais_media', 'last 3') # 3 anos
        
        if df_renda is not None and df_horas is not None:
            # Cruza as tabelas perfeitamente usando o 'ano' normalizado
            df_final = pd.merge(df_horas, df_renda, on='ano', how='inner')
            
            # Renomeia para 'trimestre' apenas para manter compatibilidade com a tabela do banco de dados já criada
            df_final = df_final.rename(columns={'ano': 'trimestre'})
            
            print(f"{VERDE}[IBGE OK] Granularidade temporal normalizada e dados cruzados com sucesso.{RESETAR}")
            return df_final
            
        return None

class CAGEDConnector:
    """ Tubulação de Big Data para o FTP do Ministério do Trabalho (Novo CAGED) """
    
    def __init__(self, diretorio_base="./data/caged"):
        self.ftp_base = "ftp://ftp.mtps.gov.br/pdet/microdados/NOVO CAGED"
        self.dir = diretorio_base
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            
    def baixar_e_agregar_mes(self, ano, mes):
        mes_str = f"{mes:02d}"
        arquivo_nome = f"CAGEDMOV{ano}{mes_str}.7z"
        url_ftp = f"{self.ftp_base}/{ano}/{ano}{mes_str}/{arquivo_nome}"
        caminho_local = os.path.join(self.dir, arquivo_nome)
        
        print(f"{AMARELO}[CAGED] Baixando microdados de {mes_str}/{ano} (Isso pode demorar)...{RESETAR}")
        
        try:
            if not os.path.exists(caminho_local):
                urllib.request.urlretrieve(url_ftp, caminho_local)
                
            print(f"{CIANO}[CAGED] Descompactando e processando chunks de {arquivo_nome}...{RESETAR}")
            with py7zr.SevenZipFile(caminho_local, mode='r') as z:
                z.extractall(path=self.dir)
                
            arquivo_txt = caminho_local.replace('.7z', '.txt')
            
            colunas_interesse = ['seção', 'salário', 'horascontratuais']
            agrupamento_setorial = pd.DataFrame()
            
            iter_csv = pd.read_csv(arquivo_txt, sep=';', usecols=colunas_interesse, 
                                   decimal=',', encoding='utf-8', chunksize=100000)
            
            for chunk in iter_csv:
                agg_chunk = chunk.groupby('seção').agg({
                    'salário': 'mean',
                    'horascontratuais': 'mean'
                }).reset_index()
                agrupamento_setorial = pd.concat([agrupamento_setorial, agg_chunk])
            
            resultado_final = agrupamento_setorial.groupby('seção').mean().reset_index()
            
            os.remove(arquivo_txt)
            
            print(f"{VERDE}[CAGED OK] Perfil de horas e salários por setor processado com sucesso.{RESETAR}")
            return resultado_final
            
        except Exception as e:
            print(f"{VERMELHO}[CAGED ERRO] Falha no processamento: {e}{RESETAR}")
            return None

class DWLoader:
    """ Responsável por carregar os dados limpos no Data Warehouse PostgreSQL """
    
    def __init__(self):
        try:
            with open('db_config.json', 'r') as f:
                cfg = json.load(f)
            db_uri = f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/macro_shift_br_db"
            self.engine = create_engine(db_uri)
        except Exception as e:
            print(f"{VERMELHO}[DW ERRO] Falha ao ler db_config.json ou conectar: {e}{RESETAR}")
            self.engine = None
            
    def salvar_ibge(self, df):
        if self.engine is None: return
        try:
            df.to_sql('tb_macro_ibge', self.engine, if_exists='append', index=False)
            print(f"{VERDE}[DW OK] Dados do IBGE gravados no Data Warehouse.{RESETAR}")
        except Exception as e:
            print(f"{AMARELO}[AVISO DW] IBGE: Registros já existem ou erro: {e}{RESETAR}")

    def salvar_caged(self, df, ano, mes):
        if self.engine is None: return
        try:
            df['ano_mes'] = f"{ano}{mes:02d}"
            df = df.rename(columns={
                'seção': 'secao_cnae', 
                'salário': 'salario_medio', 
                'horascontratuais': 'horas_contratuais_media'
            })
            
            df.to_sql('tb_micro_caged', self.engine, if_exists='append', index=False)
            print(f"{VERDE}[DW OK] Microdados do CAGED ({ano}/{mes}) arquivados na tb_micro_caged.{RESETAR}")
        except Exception as e:
            print(f"{AMARELO}[AVISO DW] CAGED: Registros já existem ou erro: {e}{RESETAR}")

# --- ORQUESTRAÇÃO MODULAR (CLI) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MacroShift BR - Pipeline de Ingestão de Dados")
    parser.add_argument('--ibge', action='store_true', help="Executa apenas a extração e carga da base IBGE")
    parser.add_argument('--caged', action='store_true', help="Executa apenas a extração e carga da base CAGED")
    parser.add_argument('--ano', type=int, default=2023, help="Ano de referência para o CAGED (ex: 2023)")
    parser.add_argument('--mes', type=int, default=10, help="Mês de referência para o CAGED (ex: 10)")
    
    args = parser.parse_args()

    if not (args.ibge or args.caged):
        print(f"{CIANO}{'='*70}\n MACROSHIFT BR // PIPELINE DE DADOS CONSOLIDADO (ETL) V3.4\n{'='*70}{RESETAR}")
        parser.print_help()
        exit()

    loader = DWLoader()

    # --- FLUXO IBGE ---
    if args.ibge:
        print(f"\n{CIANO}>>> INICIANDO FLUXO IBGE (MACRO) <<<{RESETAR}")
        ibge = IBGEConnector()
        dados_macro = ibge.extrair_horas_e_renda()
        if dados_macro is not None:
            print(dados_macro.tail(3))
            loader.salvar_ibge(dados_macro)

    # --- FLUXO CAGED ---
    if args.caged:
        print(f"\n{CIANO}>>> INICIANDO FLUXO CAGED (MICRO/SETORIAL) - Referência: {args.mes:02d}/{args.ano} <<<{RESETAR}")
        caged = CAGEDConnector()
        dados_setoriais = caged.baixar_e_agregar_mes(ano=args.ano, mes=args.mes)
        if dados_setoriais is not None:
            print(dados_setoriais.head(3))
            loader.salvar_caged(dados_setoriais, ano=args.ano, mes=args.mes)