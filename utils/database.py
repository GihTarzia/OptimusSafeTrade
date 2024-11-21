import sqlite3
import pandas as pd
from datetime import datetime
import json
from typing import Dict, List, Optional
import threading
from pathlib import Path
from queue import Queue
from contextlib import contextmanager
import yfinance as yf

class ConnectionPool:
    def __init__(self, db_path: str, max_connections: int = 5):
        self.db_path = db_path
        self.max_connections = max_connections
        self.connections = Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        
        # Inicializa o pool
        for _ in range(max_connections):
            conn = sqlite3.connect(db_path, timeout=30)
            conn.row_factory = sqlite3.Row
            self.connections.put(conn)
    
    @contextmanager
    def get_connection(self):
        connection = self.connections.get()
        try:
            yield connection
        finally:
            self.connections.put(connection)
    
    def close_all(self):
        while not self.connections.empty():
            conn = self.connections.get()
            conn.close()

class DatabaseManager:
    def __init__(self, logger, db_path: str = 'data/trading_bot.db'):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.lock = threading.Lock()
        self.pool = ConnectionPool(db_path)
        self.logger = logger
        # Cache para otimização
        self.cache = {
            'dados_mercado': {},
            'analises': {},
            'horarios': {}
        }
        self.cache_timeout = 300  # 5 minutos
        self.cache_last_update = {}
        
        self._init_database()
        self._optimize_database()
    
    def _optimize_database(self):
        """Otimiza o banco de dados"""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('PRAGMA journal_mode=WAL')  # Write-Ahead Logging
            cursor.execute('PRAGMA synchronous=NORMAL')  # Menos syncs para melhor performance
            cursor.execute('PRAGMA cache_size=-2000')  # 2MB de cache
            cursor.execute('PRAGMA temp_store=MEMORY')  # Temporários em memória
            conn.commit()
    
    def _init_database(self):
        """Inicializa as tabelas com otimizações"""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Tabela de preços com particionamento por data
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
                date_partition TEXT GENERATED ALWAYS AS (date(timestamp)) VIRTUAL,
                UNIQUE(ativo, timestamp)
                )
            ''')
  
            # Tabela de sinais com cache de resultados
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sinais (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ativo TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    direcao TEXT NOT NULL,
                    preco_entrada REAL NOT NULL,
                    preco_saida REAL,  -- Nova coluna
                    tempo_expiracao INTEGER NOT NULL,
                    score REAL NOT NULL,
                    assertividade REAL NOT NULL,
                    resultado TEXT,
                    lucro REAL,
                    padroes TEXT,
                    indicadores TEXT,
                    ml_prob REAL,
                    volatilidade REAL,
                    processado BOOLEAN DEFAULT 0,
                    data_processamento DATETIME
                )
            ''')

            # Tabela de métricas com resumos pré-calculados
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metricas_resumo (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ativo TEXT NOT NULL,
                    periodo TEXT NOT NULL,
                    data_calculo DATETIME NOT NULL,
                    win_rate REAL,
                    profit_factor REAL,
                    total_operacoes INTEGER,
                    media_assertividade REAL,
                    UNIQUE(ativo, periodo, data_calculo)
                )
            ''')
            
            # Nova tabela para resultados de backtest
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_resultados (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    total_trades INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    drawdown_maximo REAL NOT NULL,
                    retorno_total REAL NOT NULL,
                    metricas TEXT NOT NULL,
                    melhores_horarios TEXT NOT NULL,
                    evolucao_capital TEXT NOT NULL
                )
            ''')
            
            # Índices para sinais
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sinais_ativo_timestamp ON sinais(ativo, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sinais_processado ON sinais(processado)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_precos_ativo_date ON precos(ativo, date_partition)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_precos_timestamp ON precos(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sinais_resultado ON sinais(resultado) WHERE resultado IS NULL;')
          
            conn.commit()
            
    def get_horarios_sucesso(self, ativo: str) -> Dict[str, float]:
        """Retorna os horários com maior taxa de sucesso para o ativo"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT 
                        strftime('%H:%M', s.timestamp) AS horario,
                        COUNT(CASE WHEN s.resultado = 'WIN' THEN 1 END) * 1.0 / COUNT(*) AS taxa_sucesso
                    FROM sinais s
                    WHERE s.ativo = ?
                    GROUP BY strftime('%H:%M', s.timestamp)
                    ORDER BY taxa_sucesso DESC
                """
                
                cursor.execute(query, (ativo,))
                
                return {row[0]: row[1] for row in cursor.fetchall()}
        
        except Exception as e:
            self.logger.error(f"Erro ao obter horários de sucesso: {str(e)}")
            return {}
        
    def get_taxa_sucesso_horario(self, hora: int) -> float:
        """Retorna taxa de sucesso para um horário específico"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                query = """
                    SELECT 
                        COUNT(CASE WHEN resultado = 'WIN' THEN 1 END) * 1.0 / COUNT(*) as taxa_sucesso,
                        COUNT(*) as total_operacoes
                    FROM sinais
                    WHERE strftime('%H', timestamp) = ?
                    AND resultado IS NOT NULL
                    AND timestamp >= datetime('now', '-30 days')
                """

                cursor.execute(query, (f"{hora:02d}",))
                resultado = cursor.fetchone()

                if resultado and resultado[0] is not None:
                    total_ops = resultado[1]
                    # Só considera taxa se tiver mínimo de operações
                    if total_ops >= 10:  # Mínimo de 10 operações para considerar
                        return float(resultado[0])
                return 0.65  # Taxa neutra se não houver dados

        except Exception as e:
            self.logger.error(f"Erro ao obter taxa de sucesso do horário: {str(e)}")
            return 0.65
        
    def get_assertividade_recente(self, ativo: str, direcao: str, tempo_expiracao: int) -> Optional[float]:
        """Retorna a assertividade média recente para um ativo e direção"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT AVG(assertividade) AS assertividade_media
                    FROM sinais
                    WHERE ativo = ?
                    AND direcao = ?
                    AND tempo_expiracao = ?
                    AND timestamp >= datetime('now', '-7 days')
                """
                
                cursor.execute(query, (ativo, direcao, tempo_expiracao))
                result = cursor.fetchone()
                
                if result and result[0] is not None:
                    return float(result[0])
                return 50.0  # Valor padrão se não houver dados
        
        except Exception as e:
            self.logger.error(f"Erro ao obter assertividade recente: {str(e)}")
            return None
        
    # Adicionar novo método para salvar resultados
    async def salvar_resultados_backtest(self, resultados: Dict) -> bool:
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                
                query = '''
                    INSERT INTO backtest_resultados (
                        timestamp, total_trades, win_rate, profit_factor,
                        drawdown_maximo, retorno_total, metricas,
                        melhores_horarios, evolucao_capital
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
                
                metricas = resultados['metricas']
                
                cursor.execute(query, (
                    datetime.now(),
                    metricas['total_trades'],
                    metricas['win_rate'],
                    metricas['profit_factor'],
                    metricas['drawdown_maximo'],
                    metricas['retorno_total'],
                    json.dumps(metricas),
                    json.dumps(resultados['melhores_horarios']),
                    json.dumps(resultados['evolucao_capital'])
                ))
                
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao salvar resultados do backtest: {str(e)}")
            return False


    @contextmanager
    def transaction(self):
        """Gerenciador de contexto para transações"""
        with self.pool.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
    
            
    async def get_dados_mercado(self, ativo: str) -> pd.DataFrame:
        cache_key = f"mercado_{ativo}"
        try:
            # Verifica cache
            if cache_key in self.cache['dados_mercado']:
                data, timestamp = self.cache['dados_mercado'][cache_key]
                if (datetime.now() - timestamp).total_seconds() < self.cache_timeout:
                    return data

            # Busca dados novos
            dados = await self._baixar_dados_mercado(ativo)
            
            # Salva os novos dados no banco
            await self.salvar_precos_novos(ativo, dados)
            
            if not dados.empty:
                self.logger.info(f"Dados obtidos para {ativo}: {len(dados)} registros")
                self.cache['dados_mercado'][cache_key] = (dados, datetime.now())
                return dados

            else:
                self.logger.warning(f"Nenhum dado retornado do yfinance para {ativo}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Erro ao obter dados de mercado: {str(e)}")
            return pd.DataFrame()

    async def _baixar_dados_mercado(self, ativo: str) -> pd.DataFrame:
        """Baixa dados de mercado para um ativo"""
        try:
            # Baixa dados usando a biblioteca yfinance
            df = yf.download(
                ativo,
                period="1d",
                interval="1m",
                progress=False
            )

            # Certifica-se de que todas as colunas necessárias estão presentes
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in df.columns for col in required_columns):
                return df
            else:
                self.logger.error(f"Colunas ausentes no DataFrame para {ativo}: {', '.join(required_columns)}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Erro ao baixar dados de mercado para {ativo}: {str(e)}")
            return pd.DataFrame()

    async def get_preco(self, ativo: str, momento: datetime) -> Optional[float]:
        """Recupera preço para um momento específico"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT close
                    FROM precos
                    WHERE ativo = ?
                    AND timestamp = ?
                    LIMIT 1
                '''
                
                cursor.execute(query, (
                    ativo,
                    momento.strftime('%Y-%m-%d %H:%M:%S')
                ))
                
                resultado = cursor.fetchone()
                 
                if resultado:
                    return resultado[0]

                # Se não encontrar o exato, busca o preço mais próximo
                # dentro de uma janela de 10 segundos
                query = '''
                    SELECT close, timestamp,
                           ABS(STRFTIME('%s', timestamp) - STRFTIME('%s', ?)) as diff
                    FROM precos
                    WHERE ativo = ?
                    AND timestamp BETWEEN datetime(?, '-30 seconds') AND datetime(?, '+40000 seconds')
                    ORDER BY diff ASC
                    LIMIT 1
                '''

                cursor.execute(query, (
                    momento.strftime('%Y-%m-%d %H:%M:%S'),
                    ativo,
                    momento.strftime('%Y-%m-%d %H:%M:%S'),
                    momento.strftime('%Y-%m-%d %H:%M:%S')
                ))
                
                resultado = cursor.fetchone()
                if resultado:
                    diff_segundos = resultado[2]
                    if diff_segundos <= 40000:  # Só retorna se estiver dentro da janela de 10 segundos
                        return resultado[0]
                
                return None
                
        except Exception as e:
            self.logger.error(f"Erro ao recuperar preço: {str(e)}")
            return None


    async def atualizar_resultado_sinal(self, sinal_id: int, **kwargs) -> bool:
        """Atualiza resultado de um sinal"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                query = '''
                    UPDATE sinais
                    SET resultado = ?,
                        lucro = ?,
                        preco_saida = ?,
                        processado = 1,
                        data_processamento = ?
                    WHERE id = ?
                '''

                cursor.execute(query, (
                    kwargs['resultado'],
                    kwargs['lucro'],
                    kwargs['preco_saida'],
                    kwargs['data_processamento'].strftime('%Y-%m-%d %H:%M:%S'),
                    sinal_id
                ))

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Erro ao atualizar resultado do sinal: {str(e)}")
            return False

    async def registrar_sinal(self, sinal: Dict) -> Optional[int]:
        """Registra novo sinal no banco de dados"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                query = '''
                    INSERT INTO sinais (
                        ativo, timestamp, direcao, preco_entrada,
                        tempo_expiracao, score, assertividade, padroes,
                        ml_prob, volatilidade, indicadores,
                        processado
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''

                assertividade = min(100, max(0, sinal['assertividade']))

                cursor.execute(query, (
                    sinal['ativo'],
                    sinal['momento_entrada'].strftime('%Y-%m-%d %H:%M:%S'),
                    sinal['direcao'],
                    sinal['preco_entrada'],
                    sinal['tempo_expiracao'],
                    sinal['score'],
                    assertividade,
                    sinal['padroes_forca'],
                    sinal['ml_prob'],
                    sinal['volatilidade'],
                    json.dumps(sinal['indicadores']),
                    False
                ))

                conn.commit()
                return cursor.lastrowid

        except Exception as e:
            self.logger.error(f"Erro ao registrar sinal: {str(e)}")
            return None
    
    async def get_sinais_sem_resultado(self) -> List[Dict]:
        """Recupera sinais pendentes de forma assíncrona"""
        try:
            with self.pool.get_connection() as conn:
                query = '''
                    SELECT DISTINCT s.* 
                    FROM sinais s
                    WHERE s.resultado IS NULL 
                    AND s.processado = 0
                    AND datetime(s.timestamp, '+' || s.tempo_expiracao || ' minutes') <= datetime('now')
                    AND s.timestamp >= datetime('now', '-1 day')
                '''
                
                cursor = conn.cursor()
                cursor.execute(query)

                
                sinais = []
                for row in cursor.fetchall():
                    sinal = dict(row)
                    if isinstance(sinal['timestamp'], str):
                        sinal['timestamp'] = datetime.strptime(
                            sinal['timestamp'], 
                            '%Y-%m-%d %H:%M:%S'
                        )
                    sinais.append(sinal)
                
                return sinais

        except Exception as e:
            self.logger.error(f"Erro ao recuperar sinais pendentes: {str(e)}")
            return []
    
    async def salvar_precos_novos(self, ativo: str, dados: pd.DataFrame) -> bool:
        try:
            if not all(col in dados.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                self.logger.error(f"Colunas necessárias ausentes para {ativo}")
                return False    

            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                for timestamp, row in dados.iterrows():
                    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')

                    # Tenta atualizar primeiro
                    cursor.execute("""
                        UPDATE precos 
                        SET open = ?, high = ?, low = ?, close = ?, volume = ?
                        WHERE ativo = ? AND timestamp = ?
                    """, (
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        float(row['Volume']),
                        ativo,
                        timestamp_str
                    ))

                    # Se não atualizou nenhum registro, insere novo
                    if cursor.rowcount == 0:
                        cursor.execute("""
                            INSERT INTO precos (ativo, timestamp, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            ativo,
                            timestamp_str,
                            float(row['Open']),
                            float(row['High']),
                            float(row['Low']),
                            float(row['Close']),
                            float(row['Volume'])
                        ))  

                conn.commit()
                return True 

        except Exception as e:
            self.logger.error(f"Erro ao salvar preços para {ativo}: {str(e)}")
            return False
    
    # Correção da função salvar_precos
    async def salvar_precos(self, ativo: str, dados: pd.DataFrame) -> bool:
        """Salva dados de preços no banco de dados de forma assíncrona"""
        try:
            # Verifica colunas necessárias
            colunas_requeridas = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in dados.columns for col in colunas_requeridas):
                self.logger.error(f"Erro: DataFrame deve conter as colunas: {colunas_requeridas}")
                return False
    
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
    
                # Prepara dados para inserção
                rows = []
                for timestamp, row in dados.iterrows():
                    # Converte timestamp para string
                    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    
                    rows.append((
                        ativo,
                        timestamp_str,
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        float(row['Volume'])
                    ))
    
                # Insere dados em lote
                cursor.executemany('''
                    INSERT OR REPLACE INTO precos (
                        ativo, timestamp, open, high, low, close, volume
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', rows)
    
                conn.commit()
                return True
    
        except Exception as e:
            self.logger.error(f"Erro ao salvar preços para {ativo}: {str(e)}")
            return False  
    # Correção da função get_dados_historicos
    async def get_dados_historicos(self, dias: int = 30) -> pd.DataFrame:
        """Recupera dados históricos do banco de dados"""
        try:
            with self.pool.get_connection() as conn:
                query = '''
                    SELECT 
                        timestamp,
                        ativo,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM precos
                    WHERE timestamp >= datetime('now', ? || ' days')
                    ORDER BY timestamp DESC
                '''

                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(-dias,),
                    parse_dates=['timestamp']
                )

                if not df.empty:
                    df.set_index('timestamp', inplace=True)

                return df

        except Exception as e:
            self.logger.error(f"Erro ao recuperar dados históricos: {str(e)}")
            return pd.DataFrame()
        
    async def get_dados_treino(self) -> pd.DataFrame:
        """Recupera dados de treinamento do banco de dados"""
        try:
            with self.pool.get_connection() as conn:
                query = '''
                    SELECT 
                        timestamp,
                        ativo,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM precos
                    WHERE timestamp >= datetime('now', '-90 days')
                    ORDER BY timestamp ASC
                '''

                df = pd.read_sql_query(
                    query,
                    conn,
                    parse_dates=['timestamp']
                )

                if not df.empty:
                    # Pivota os dados para formato mais adequado para treino
                    df_pivot = df.pivot(
                        index='timestamp',
                        columns='ativo',
                        values=['open', 'high', 'low', 'close', 'volume']
                    )

                    # Achata os níveis das colunas
                    df_pivot.columns = [f"{col[1]}_{col[0]}" for col in df_pivot.columns]

                return df_pivot if not df.empty else pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Erro ao recuperar dados de treino: {str(e)}")
            return pd.DataFrame()
    
    async def get_operacoes_periodo(self, inicio: str, fim: str) -> List[Dict]:
        """Recupera operações em um período específico"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                query = '''
                    SELECT s.*,
                        julianday(data_processamento) - julianday(timestamp) as duracao
                    FROM sinais s
                    WHERE strftime('%H:%M', timestamp) BETWEEN ? AND ?
                    AND resultado IS NOT NULL
                    AND timestamp >= datetime('now', '-30 days')
                    ORDER BY timestamp DESC
                '''

                cursor.execute(query, (inicio, fim))

                operacoes = []
                for row in cursor.fetchall():
                    operacao = dict(row)
                    # Converte timestamp para datetime se necessário
                    if isinstance(operacao['timestamp'], str):
                        operacao['timestamp'] = datetime.strptime(
                            operacao['timestamp'],
                            '%Y-%m-%d %H:%M:%S'
                        )
                    operacoes.append(operacao)

                return operacoes

        except Exception as e:
            self.logger.error(f"Erro ao recuperar operações do período: {str(e)}")
            return []