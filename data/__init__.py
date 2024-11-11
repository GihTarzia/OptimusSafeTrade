from pathlib import Path
import sqlite3
import json
from datetime import datetime

class DataManager:
    def __init__(self):
        self.data_dir = Path(__file__).parent
        self.db_path = self.data_dir / 'historico.db'
        self.models_dir = self.data_dir / 'modelos'
        
        # Cria diretórios necessários
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def init_database(self):
        """Inicializa o banco de dados se não existir"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Cria tabelas necessárias se não existirem
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS modelos_ml (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome TEXT NOT NULL,
                    data_criacao DATETIME NOT NULL,
                    metricas TEXT,
                    caminho_arquivo TEXT NOT NULL
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_backup DATETIME NOT NULL,
                    tipo TEXT NOT NULL,
                    caminho_arquivo TEXT NOT NULL
                )
            ''')

            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Erro ao inicializar banco de dados: {str(e)}")

    def salvar_modelo(self, nome: str, modelo_path: str, metricas: dict):
        """Salva informações sobre um modelo treinado"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO modelos_ml (nome, data_criacao, metricas, caminho_arquivo)
                VALUES (?, ?, ?, ?)
            ''', (
                nome,
                datetime.now().isoformat(),
                json.dumps(metricas),
                modelo_path
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Erro ao salvar modelo: {str(e)}")

    def get_ultimo_modelo(self, nome: str = None):
        """Recupera informações do último modelo salvo"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if nome:
                cursor.execute('''
                    SELECT * FROM modelos_ml
                    WHERE nome = ?
                    ORDER BY data_criacao DESC
                    LIMIT 1
                ''', (nome,))
            else:
                cursor.execute('''
                    SELECT * FROM modelos_ml
                    ORDER BY data_criacao DESC
                    LIMIT 1
                ''')
            
            resultado = cursor.fetchone()
            conn.close()
            
            if resultado:
                return {
                    'id': resultado[0],
                    'nome': resultado[1],
                    'data_criacao': resultado[2],
                    'metricas': json.loads(resultado[3]),
                    'caminho_arquivo': resultado[4]
                }
            return None
            
        except Exception as e:
            print(f"Erro ao recuperar modelo: {str(e)}")
            return None

    def registrar_backup(self, tipo: str, caminho: str):
        """Registra informações sobre backup realizado"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO backups (data_backup, tipo, caminho_arquivo)
                VALUES (?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                tipo,
                caminho
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Erro ao registrar backup: {str(e)}")

    def get_estatisticas_modelos(self):
        """Retorna estatísticas sobre os modelos salvos"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM modelos_ml')
            total_modelos = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT AVG(CAST(json_extract(metricas, '$.win_rate') AS FLOAT))
                FROM modelos_ml
            ''')
            win_rate_medio = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_modelos': total_modelos,
                'win_rate_medio': win_rate_medio,
                'espaco_disco': self._calcular_espaco_disco()
            }
            
        except Exception as e:
            print(f"Erro ao obter estatísticas: {str(e)}")
            return {}

    def _calcular_espaco_disco(self):
        """Calcula espaço em disco usado pelos modelos"""
        try:
            total_bytes = sum(f.stat().st_size for f in self.models_dir.glob('**/*') if f.is_file())
            return total_bytes / (1024 * 1024)  # Converte para MB
        except Exception:
            return 0