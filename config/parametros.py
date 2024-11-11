import yaml
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, time
import json

class Config:
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config_path = Path(config_path)
        self.last_reload = datetime.now()
        self.reload_interval = 300  # 5 minutos
        
        # Configurações padrão
        self.DEFAULT_CONFIG = {
            'sistema': {
                'modo': 'producao',  # 'producao' ou 'teste'
                'debug': False,
                'auto_restart': True,
                'max_memoria': 1024,  # MB
                'backup_interval': 3600  # 1 hora
            },
            'trading': {
                'saldo_inicial': 1000,
                'risco_por_operacao': 0.02,  # 2%
                'stop_diario': -0.1,  # -10%
                'meta_diaria': 0.05,  # 5%
                'max_operacoes_dia': 15,
                'min_intervalo_operacoes': 300,  # 5 minutos
                'tempo_expiracao_padrao': 5,  # minutos
                'martingale': {
                    'ativo': False,
                    'multiplicador': 2.0,
                    'max_niveis': 2
                }
            }
        }
        
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()

    def load_config(self):
        """Carrega configurações do arquivo YAML"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    self._update_config(yaml_config)
            else:
                self._save_default_config()
        except Exception as e:
            print(f"Erro ao carregar configurações: {str(e)}")

    def _update_config(self, new_config: Dict):
        """Atualiza configurações mantendo a estrutura"""
        def update_dict(base: Dict, new: Dict):
            for key, value in new.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    update_dict(base[key], value)
                else:
                    base[key] = value
        
        update_dict(self.config, new_config)

    def _save_default_config(self):
        """Salva configurações padrão no arquivo"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self.DEFAULT_CONFIG, f, default_flow_style=False)
        except Exception as e:
            print(f"Erro ao salvar configurações padrão: {str(e)}")

    def save_config(self):
        """Salva configurações atuais no arquivo"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            print(f"Erro ao salvar configurações: {str(e)}")

    def get(self, path: str, default: Any = None) -> Any:
        """Obtém valor de configuração por caminho"""
        try:
            value = self.config
            for key in path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, path: str, value: Any):
        """Define valor de configuração por caminho"""
        try:
            keys = path.split('.')
            current = self.config
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = value
            self.save_config()
        except Exception as e:
            print(f"Erro ao definir configuração: {str(e)}")

    def is_horario_operacional(self) -> bool:
        """Verifica se está em horário operacional"""
        agora = datetime.now()
        dia_semana = agora.strftime('%A')
        
        if dia_semana not in self.get('horarios.dias_operacionais'):
            return False
            
        hora_atual = agora.time()
        inicio = datetime.strptime(self.get('horarios.inicio_operacoes'), '%H:%M').time()
        fim = datetime.strptime(self.get('horarios.fim_operacoes'), '%H:%M').time()
        
        # Verifica horários bloqueados
        for bloqueio in self.get('horarios.horarios_bloqueados'):
            inicio_bloq, fim_bloq = bloqueio.split('-')
            inicio_bloq = datetime.strptime(inicio_bloq, '%H:%M').time()
            fim_bloq = datetime.strptime(fim_bloq, '%H:%M').time()
            
            if inicio_bloq <= hora_atual <= fim_bloq:
                return False
        
        return inicio <= hora_atual <= fim

    def get_ativos_ativos(self) -> List[str]:
        """Retorna lista de todos os ativos ativos"""
        try:
            ativos = self.config.get('ativos', {})
            if not ativos:
                # Lista padrão se não houver configuração
                return [
                    "EURUSD=X",
                    "GBPUSD=X",
                    "USDJPY=X",
                    "AUDUSD=X",
                    "USDCAD=X",
                    "NZDUSD=X"
                ]
                
            todos_ativos = []
            for categoria in ativos.values():
                if isinstance(categoria, list):
                    todos_ativos.extend(categoria)
            
            return todos_ativos
            
        except Exception as e:
            print(f"Erro ao obter lista de ativos: {str(e)}")
            # Retorna lista mínima em caso de erro
            return ["EURUSD=X", "GBPUSD=X"]