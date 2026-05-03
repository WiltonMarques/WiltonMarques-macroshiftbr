import json
from sqlalchemy import create_engine, text
import os

# --- CONFIGURAÇÃO VISUAL ---
VERDE, VERMELHO, CIANO, AMARELO, RESETAR = '\033[92m', '\033[91m', '\033[96m', '\033[93m', '\033[0m'
print(f"{CIANO}{'='*70}\n MACROSHIFT BR // DATA WAREHOUSE SETUP V1.0\n{'='*70}{RESETAR}")

def carregar_credenciais():
    try:
        with open('db_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{VERMELHO}[ERRO] Arquivo 'db_config.json' não encontrado.{RESETAR}")
        return None

def criar_banco_de_dados(cfg):
    """ Conecta ao servidor e cria o banco macro_shift_br_db isolado do Aurum OS """
    print(f"{AMARELO}[+] Conectando ao servidor PostgreSQL principal...{RESETAR}")
    # Conecta ao banco padrão 'postgres' apenas para poder executar o CREATE DATABASE
    uri_admin = f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/postgres"
    engine_admin = create_engine(uri_admin, isolation_level="AUTOCOMMIT")
    
    nome_db = "macro_shift_br_db"
    
    try:
        with engine_admin.connect() as conn:
            resultado = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname='{nome_db}'")).fetchone()
            if not resultado:
                print(f"{CIANO}[+] Criando o Data Warehouse: {nome_db}...{RESETAR}")
                conn.execute(text(f"CREATE DATABASE {nome_db}"))
                print(f"{VERDE}[OK] Banco de dados criado com sucesso.{RESETAR}")
            else:
                print(f"{VERDE}[OK] Banco de dados '{nome_db}' já existe.{RESETAR}")
    except Exception as e:
        print(f"{VERMELHO}[ERRO] Falha ao criar o banco: {e}{RESETAR}")

def construir_esquema(cfg):
    """ Conecta ao novo banco e cria as tabelas estruturadas """
    uri_dw = f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/macro_shift_br_db"
    engine_dw = create_engine(uri_dw)
    
    print(f"{AMARELO}[+] Montando o esquema de tabelas (Schema)...{RESETAR}")
    
    tabela_ibge = """
    CREATE TABLE IF NOT EXISTS tb_macro_ibge (
        id SERIAL PRIMARY KEY,
        trimestre VARCHAR(10) UNIQUE NOT NULL,
        horas_semanais_media NUMERIC(10, 4),
        renda_media_real NUMERIC(12, 2),
        data_atualizacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    tabela_caged = """
    CREATE TABLE IF NOT EXISTS tb_micro_caged (
        id SERIAL PRIMARY KEY,
        ano_mes VARCHAR(6) NOT NULL,
        secao_cnae VARCHAR(255) NOT NULL,
        salario_medio NUMERIC(12, 2),
        horas_contratuais_media NUMERIC(10, 4),
        data_atualizacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (ano_mes, secao_cnae)
    );
    """
    
    try:
        with engine_dw.connect() as conn:
            conn.execute(text(tabela_ibge))
            conn.execute(text(tabela_caged))
            conn.commit()
            print(f"{VERDE}[OK] Tabelas 'tb_macro_ibge' e 'tb_micro_caged' operacionais.{RESETAR}")
    except Exception as e:
        print(f"{VERMELHO}[ERRO] Falha ao montar tabelas: {e}{RESETAR}")

if __name__ == "__main__":
    config = carregar_credenciais()
    if config:
        criar_banco_de_dados(config)
        construir_esquema(config)
        print(f"\n{CIANO}[+] Infraestrutura do MacroShift BR pronta para receber dados!{RESETAR}")