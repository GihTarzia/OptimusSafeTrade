import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Union
import threading
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path: str = 'data/trading_bot.db'):
        # Garante que o diretório existe
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()

    def _init_database(self):
        """Inicializa as tabelas do banco de dados"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
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

    def get_connection(self):
        """Retorna uma conexão com o banco de dados"""
        return sqlite3.connect(self.db_path)

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
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ativo, momento_entrada, direcao,
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
                    dados.to_sql('precos', conn, if_exists='append', index=False)
            except Exception as e:
                print(f"Erro ao salvar preços: {str(e)}")

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

    def get_dados_treino(self) -> pd.DataFrame:
        """Recupera dados para treino do modelo ML"""
        query = '''
            SELECT s.*, p.open, p.high, p.low, p.close, p.volume
            FROM sinais s
            JOIN precos p ON s.ativo = p.ativo 
                AND s.timestamp = p.timestamp
            WHERE s.resultado IS NOT NULL
            ORDER BY s.timestamp DESC
            LIMIT 10000
        '''
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

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