import logging
from datetime import datetime
import os
from pathlib import Path
import json
import threading
from colorama import init, Fore, Style
from logging.handlers import RotatingFileHandler

init()  # Inicializa colorama

class TradingLogger:
    def __init__(self, log_dir: str = 'logs', max_files: int = 5):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuração básica do logger
        self.logger = logging.getLogger('TradingBot')
        self.logger.setLevel(logging.DEBUG)
        
        # Formato do log
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para arquivo com rotação
        file_handler = RotatingFileHandler(
            self.log_dir / 'trading_bot.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=max_files
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Métricas e estatísticas
        self.metricas = {}
        self.alertas_criticos = []
        self.lock = threading.Lock()
        
        # Níveis de alerta
        self.niveis_alerta = {
            'CRITICAL': {'cor': Fore.RED},
            'ERROR': {'cor': Fore.RED},
            'WARNING': {'cor': Fore.YELLOW},
            'INFO': {'cor': Fore.CYAN},
            'DEBUG': {'cor': Fore.WHITE}
        }

    def _log_mensagem(self, nivel: str, mensagem: str, dados: dict = None):
        """Processa e registra uma mensagem de log"""
        with self.lock:
            try:
                # Formata a mensagem
                msg_formatada = f"{self.niveis_alerta[nivel]['cor']}{mensagem}{Style.RESET_ALL}"
                if dados:
                    msg_formatada += f"\nDados: {json.dumps(dados, indent=2)}"
                
                # Registra no logger
                getattr(self.logger, nivel.lower())(mensagem)
                
                # Exibe no console com cores
                print(msg_formatada)
                
                # Registra métricas
                self._registrar_metrica(nivel, mensagem, dados)
                
            except Exception as e:
                print(f"Erro ao registrar log: {str(e)}")

    def _registrar_metrica(self, nivel: str, mensagem: str, dados: dict = None):
        """Registra métricas para análise"""
        timestamp = datetime.now().isoformat()
        
        with self.lock:
            if nivel not in self.metricas:
                self.metricas[nivel] = []
                
            metrica = {
                'timestamp': timestamp,
                'mensagem': mensagem,
                'dados': dados
            }
            
            self.metricas[nivel].append(metrica)
            
            # Mantém apenas últimas 1000 métricas por nível
            if len(self.metricas[nivel]) > 1000:
                self.metricas[nivel].pop(0)
            
            # Registra alertas críticos
            if nivel in ['CRITICAL', 'ERROR']:
                self.alertas_criticos.append(metrica)
                if len(self.alertas_criticos) > 100:
                    self.alertas_criticos.pop(0)

    def critical(self, mensagem: str, dados: dict = None):
        """Registra mensagem crítica"""
        self._log_mensagem('CRITICAL', mensagem, dados)

    def error(self, mensagem: str, dados: dict = None):
        """Registra erro"""
        self._log_mensagem('ERROR', mensagem, dados)

    def warning(self, mensagem: str, dados: dict = None):
        """Registra aviso"""
        self._log_mensagem('WARNING', mensagem, dados)

    def info(self, mensagem: str, dados: dict = None):
        """Registra informação"""
        self._log_mensagem('INFO', mensagem, dados)

    def debug(self, mensagem: str, dados: dict = None):
        """Registra mensagem de debug"""
        self._log_mensagem('DEBUG', mensagem, dados)

    def registrar_operacao(self, operacao: dict):
        """Registra detalhes de uma operação"""
        nivel = 'INFO' if operacao.get('resultado') == 'WIN' else 'WARNING'
        self._log_mensagem(nivel, "Nova Operação", operacao)

    def registrar_erro_sistema(self, erro: Exception, contexto: dict = None):
        """Registra erro do sistema"""
        dados = {
            'erro': str(erro),
            'tipo': type(erro).__name__,
            'contexto': contexto or {}
        }
        self._log_mensagem('ERROR', "Erro do Sistema", dados)

    def get_metricas(self) -> dict:
        """Retorna métricas acumuladas"""
        with self.lock:
            return {
                'total_logs': sum(len(logs) for logs in self.metricas.values()),
                'logs_por_nivel': {nivel: len(logs) for nivel, logs in self.metricas.items()},
                'alertas_criticos': len(self.alertas_criticos),
                'ultimo_alerta': self.alertas_criticos[-1] if self.alertas_criticos else None
            }

    def exportar_logs(self, caminho: str = None):
        """Exporta logs para arquivo JSON"""
        if not caminho:
            caminho = self.log_dir / f'logs_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
        with self.lock:
            try:
                dados_export = {
                    'metricas': self.metricas,
                    'alertas_criticos': self.alertas_criticos,
                    'export_timestamp': datetime.now().isoformat()
                }
                
                with open(caminho, 'w') as f:
                    json.dump(dados_export, f, indent=4)
                    
                self.info(f"Logs exportados para: {caminho}")
                
            except Exception as e:
                self.error(f"Erro ao exportar logs: {str(e)}")

    def limpar_logs_antigos(self, dias: int = 30):
        """Remove logs mais antigos que X dias"""
        try:
            data_limite = datetime.now().timestamp() - (dias * 24 * 60 * 60)
            
            for arquivo in self.log_dir.glob('*.log.*'):
                if arquivo.stat().st_mtime < data_limite:
                    arquivo.unlink()
                    self.info(f"Log antigo removido: {arquivo}")
                    
        except Exception as e:
            self.error(f"Erro ao limpar logs antigos: {str(e)}")