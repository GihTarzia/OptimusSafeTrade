import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Union
import threading
from colorama import Fore, Style
from pathlib import Path
import time  # Adiciona import do time

class DatabaseConnection:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path, timeout=20)
        self.conn.row_factory = sqlite3.Row
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
            
class DatabaseManager:
    def __init__(self, db_path: str = 'data/trading_bot.db'):
        # Garante que o diretório existe
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
        
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path, timeout=20)
        self.conn.row_factory = sqlite3.Row
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
            
    def _init_database(self):
        """Inicializa as tabelas do banco de dados"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Tabela de métricas de mercado
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metricas_mercado (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ativo TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    volatilidade REAL,
                    score_mercado REAL,
                    range_medio REAL,
                    tendencia_definida REAL,
                    movimento_consistente REAL,
                    lateralizacao REAL,
                    detalhes TEXT,
                    UNIQUE(ativo, timestamp)
                )
            ''')

            # Cria índice para melhor performance
            cursor.execute('''
                      CREATE INDEX IF NOT EXISTS idx_metricas_mercado_ativo_timestamp 
                ON metricas_mercado(ativo, timestamp)
            ''')

            # Tabela de sinais
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sinais (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ativo TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    direcao TEXT NOT NULL,
                    preco_entrada REAL NOT NULL,
                    tempo_expiracao INTEGER NOT NULL,
                    score REAL NOT NULL,
                    assertividade REAL NOT NULL,
                    resultado TEXT,
                    lucro REAL,
                    padroes TEXT,
                    indicadores TEXT,
                    ml_prob REAL,
                    volatilidade REAL
                )
            ''')
            
            # Tabela de preços históricos
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS precos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ativo TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL,
                    UNIQUE(ativo, timestamp)
                )
            ''')

            # Tabela de métricas
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metricas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    tipo TEXT NOT NULL,
                    valor REAL NOT NULL,
                    detalhes TEXT
                )
            ''')
            
            conn.commit()

    def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        """Executa uma query SELECT e retorna todos os resultados"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            # Obtém os nomes das colunas
            columns = [description[0] for description in cursor.description]

            # Converte os resultados em lista de dicionários
            results = cursor.fetchall()
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            print(f"Erro ao executar fetch_all: {str(e)}")
            return []
    def fetch_one(self, query: str, params: tuple = None) -> Dict:
        """Executa uma query SELECT e retorna um resultado"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            # Obtém os nomes das colunas
            columns = [description[0] for description in cursor.description]
            
            # Converte o resultado em dicionário
            result = cursor.fetchone()
            return dict(zip(columns, result)) if result else None
            
        except Exception as e:
            print(f"Erro ao executar fetch_one: {str(e)}")
                
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=20)  # Adiciona timeout
        conn.row_factory = sqlite3.Row
        return conn
    
    def execute_with_retry(self, query: str, params: tuple = None, max_retries: int = 3) -> bool:
        """Executa uma query com retry em caso de banco travado"""
        with self.lock:
            for attempt in range(max_retries):
                try:
                    with self.get_connection() as conn:
                        cursor = conn.cursor()
                        if params:
                            cursor.execute(query, params)
                        else:
                            cursor.execute(query)
                        conn.commit()
                        return True
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) and attempt < max_retries - 1:
                        time.sleep(1)  # Espera 1 segundo antes de tentar novamente
                        continue
                    raise
                except Exception as e:
                    print(f"Erro na tentativa {attempt + 1}: {str(e)}")
                    if attempt == max_retries - 1:
                        return False

    def registrar_sinal(self, ativo: str, direcao: str, momento_entrada: datetime,
                       tempo_expiracao: int, score: float, assertividade: float,
                       padroes: List[str], indicadores: Dict, ml_prob: float,
                       volatilidade: float) -> int:
        """Registra um novo sinal no banco de dados"""
        with self.lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO sinais (
                            ativo, timestamp, direcao, preco_entrada,
                            tempo_expiracao, score, assertividade, padroes,
                            indicadores, ml_prob, volatilidade
                        ) VALUES (?, datetime(?), ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ativo, momento_entrada.strftime('%Y-%m-%d %H:%M:%S'), direcao,
                        self.get_ultimo_preco(ativo),
                        tempo_expiracao, score, assertividade,
                        json.dumps(padroes),
                        json.dumps(indicadores),
                        ml_prob, volatilidade
                    ))
                    
                    return cursor.lastrowid
            except Exception as e:
                print(f"Erro ao registrar sinal: {str(e)}")
                return None

    def registrar_resultado_sinal(self, sinal_id: int, resultado: str, lucro: float):
        """Registra o resultado de um sinal"""
        with self.lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                    UPDATE sinais 
                    SET 
                        resultado = ?,
                        lucro = ?
                    WHERE id = ?
                    ''', (resultado, lucro, sinal_id))
                    conn.commit()
                    return True

            except Exception as e:
                print(f"Erro ao registrar resultado: {str(e)}")
                return False

    def get_preco(self, ativo: str, momento: datetime) -> float:
        """Busca o preço de um ativo em um determinado momento"""
        with self.lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    query = """
                    SELECT close 
                    FROM precos 
                    WHERE ativo = ? 
                    AND strftime('%Y-%m-%d %H:%M:%S', timestamp) <= ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                    """
                    momento_str = momento.strftime('%Y-%m-%d %H:%M:%S')
                    cursor.execute(query, (ativo, momento_str))
                    result = cursor.fetchone()
                    return float(result[0]) if result else None
            except Exception as e:
                print(f"Erro ao buscar preço: {str(e)}")
                return None
    
    def get_sinais_sem_resultado(self):
        """Busca sinais sem resultado registrado"""
        with self.lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    query = """
                    SELECT * FROM sinais 
                    WHERE resultado IS NULL 
                    AND timestamp < datetime('now')
                    """
                    cursor.execute(query)
                    columns = [description[0] for description in cursor.description]
                    results = cursor.fetchall()
                    return [dict(zip(columns, row)) for row in results]
            except Exception as e:
                print(f"Erro ao buscar sinais sem resultado: {str(e)}")
                return []

    def atualizar_resultado_sinal(self, sinal_id: int, resultado: str, lucro: float):
        """Atualiza o resultado de um sinal"""
        with self.lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE sinais
                        SET resultado = ?, lucro = ?
                        WHERE id = ?
                    ''', (resultado, lucro, sinal_id))
                    conn.commit()
            except Exception as e:
                print(f"Erro ao atualizar resultado: {str(e)}")

    def salvar_precos(self, ativo: str, dados: pd.DataFrame):
        """Salva dados históricos de preços"""
        with self.lock:
            try:
                with self.get_connection() as conn:
                    # Remove a coluna 'Adj Close' se existir
                    if 'Adj Close' in dados.columns:
                        dados = dados.drop('Adj Close', axis=1)

                    # Prepara os dados
                    dados_para_salvar = dados.copy()

                    # Renomeia as colunas para minúsculo
                    dados_para_salvar.columns = [col.lower() for col in dados_para_salvar.columns]

                    # Se a coluna 'datetime' existe, renomeia para 'timestamp'
                    if 'datetime' in dados_para_salvar.columns:
                        dados_para_salvar = dados_para_salvar.rename(columns={'datetime': 'timestamp'})
                                                                     
                    # Adiciona a coluna do ativo
                    dados_para_salvar['ativo'] = ativo

                    # Garante que temos todas as colunas necessárias
                    colunas_necessarias = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ativo']
                    for col in colunas_necessarias:
                        if col not in dados_para_salvar.columns:
                            if col == 'volume':
                                dados_para_salvar[col] = 0
                            else:
                                raise ValueError(f"Coluna obrigatória ausente: {col}")
                            
                    # Converte timestamp para string no formato correto
                    dados_para_salvar['timestamp'] = pd.to_datetime(dados_para_salvar['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                    # Salva no banco
                    dados_para_salvar.to_sql('precos', conn, if_exists='append', index=False)

            except Exception as e:
                print(f"Erro ao salvar preços para {ativo}: {str(e)}")
                print("Colunas disponíveis:", dados.columns.tolist())

    def get_dados_mercado(self, ativo: str, limite: int = 1000) -> pd.DataFrame:
        """Recupera dados históricos do mercado"""
        query = f'''
            SELECT timestamp, open, high, low, close, volume
            FROM precos
            WHERE ativo = ?
            ORDER BY timestamp DESC
            LIMIT {limite}
        '''
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(ativo,))

    def get_metricas_mercado(self, ativo: str, periodo: str = '1d') -> pd.DataFrame:
        """Recupera métricas históricas de mercado"""
        query = '''
            SELECT *
            FROM metricas_mercado
            WHERE ativo = ?
            AND timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC
        '''
        
        with self.get_connection() as conn:
            return pd.read_sql_query(
                query, 
                conn, 
                params=(ativo, f'-{periodo}'),
                parse_dates=['timestamp']
            )
    
    def get_resumo_metricas_mercado(self, ativo: str, periodo: str = '1d') -> Dict:
        """Retorna resumo estatístico das métricas de mercado"""
        df = self.get_metricas_mercado(ativo, periodo)
        if df.empty:
            return {}
            
        return {
            'score_medio': df['score_mercado'].mean(),
            'volatilidade_media': df['volatilidade'].mean(),
            'score_min': df['score_mercado'].min(),
            'score_max': df['score_mercado'].max(),
            'periodos_favoraveis': (df['score_mercado'] >= 0.75).sum(),
            'total_periodos': len(df),
            'qualidade_geral': (df['score_mercado'] >= 0.75).mean() * 100
        }

    def registrar_metricas_mercado(self, ativo: str, metricas: Dict):
        """Registra métricas de mercado para análise posterior"""
        with self.lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute('''
                        INSERT OR REPLACE INTO metricas_mercado (
                            ativo,
                            timestamp,
                            volatilidade,
                            score_mercado,
                            range_medio,
                            tendencia_definida,
                            movimento_consistente,
                            lateralizacao,
                            detalhes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ativo,
                        metricas['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        metricas.get('volatilidade', 0),
                        metricas.get('score_mercado', 0),
                        metricas.get('range_medio', 0),
                        metricas.get('tendencia_definida', 0),
                        metricas.get('movimento_consistente', 0),
                        metricas.get('lateralizacao', 0),
                        json.dumps(metricas.get('detalhes', {}))
                    ))

                    conn.commit()

            except Exception as e:
                self.logger.error(f"Erro ao registrar métricas de mercado: {str(e)}")



    def get_dados_treino(self) -> pd.DataFrame:
        """Recupera dados para treino do modelo ML"""
        try:
            query = '''
            SELECT DISTINCT
                p.timestamp,
                p.open as Open,
                p.high as High,
                p.low as Low,
                p.close as Close,
                p.volume as Volume,
                p.ativo
            FROM precos p
            ORDER BY p.timestamp DESC
            LIMIT 10000
            '''

            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn)
                print(f"Dados recuperados do banco: {len(df)} registros")
                if df.empty:
                    print("Nenhum dado encontrado no banco")
                else:
                    print("Colunas disponíveis:", df.columns.tolist())
                    print("Ativos únicos:", df['ativo'].unique().tolist())
                    print("Período dos dados:", df['timestamp'].min(), "até", df['timestamp'].max())
                return df
        except Exception as e:
            print(f"{Fore.RED}Erro ao recuperar dados de treino: {str(e)}{Style.RESET_ALL}")
            return pd.DataFrame()

    def get_estatisticas(self, periodo: str = '1d') -> Dict:
        """Recupera estatísticas de performance"""
        data_limite = datetime.now() - self._converter_periodo(periodo)
        
        query = '''
            SELECT 
                COUNT(*) as total_sinais,
                SUM(CASE WHEN resultado = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN resultado = 'LOSS' THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN resultado = 'WIN' THEN lucro ELSE 0 END) as lucro_medio_win,
                AVG(CASE WHEN resultado = 'LOSS' THEN lucro ELSE 0 END) as lucro_medio_loss,
                SUM(lucro) as lucro_total
            FROM sinais
            WHERE timestamp >= ?
        '''
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (data_limite,))
            resultado = cursor.fetchone()
            
            if resultado:
                total, wins, losses, lucro_win, lucro_loss, lucro_total = resultado
                return {
                    'total_operacoes': total,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': (wins / total * 100) if total > 0 else 0,
                    'lucro_medio_win': lucro_win or 0,
                    'lucro_medio_loss': lucro_loss or 0,
                    'lucro_total': lucro_total or 0
                }
            return None

    def get_horarios_sucesso(self, ativo: str) -> Dict[str, float]:
        """Recupera taxa de sucesso por horário"""
        query = '''
            SELECT 
                strftime('%H:%M', timestamp) as horario,
                COUNT(*) as total,
                SUM(CASE WHEN resultado = 'WIN' THEN 1 ELSE 0 END) as wins
            FROM sinais
            WHERE ativo = ?
            GROUP BY horario
            HAVING total >= 5
        '''
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (ativo,))
            resultados = cursor.fetchall()
            
            return {
                horario: wins/total
                for horario, total, wins in resultados
            }

    def get_assertividade_recente(self, ativo: str, direcao: str) -> float:
        """Recupera assertividade recente para um ativo/direção"""
        query = '''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN resultado = 'WIN' THEN 1 ELSE 0 END) as wins
            FROM sinais
            WHERE ativo = ?
                AND direcao = ?
                AND timestamp >= datetime('now', '-1 day')
        '''
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (ativo, direcao))
            total, wins = cursor.fetchone()
            
            return (wins / total * 100) if total > 0 else 0

    def get_ultimo_preco(self, ativo: str) -> float:
        """Recupera o último preço registrado para um ativo"""
        query = '''
            SELECT close
            FROM precos
            WHERE ativo = ?
            ORDER BY timestamp DESC
            LIMIT 1
        '''
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (ativo,))
            resultado = cursor.fetchone()
            
            return resultado[0] if resultado else None

    def _converter_periodo(self, periodo: str) -> timedelta:
        """Converte string de período em timedelta"""
        unidade = periodo[-1]
        valor = int(periodo[:-1])
        
        if unidade == 'd':
            return timedelta(days=valor)
        elif unidade == 'h':
            return timedelta(hours=valor)
        elif unidade == 'm':
            return timedelta(minutes=valor)
        else:
            return timedelta(days=1)  # padrão