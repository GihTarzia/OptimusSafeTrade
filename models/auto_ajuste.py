import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score
import json
from dataclasses import dataclass
import optuna
from concurrent.futures import ThreadPoolExecutor

@dataclass
class ResultadoOtimizacao:
    parametros: Dict
    win_rate: float
    profit_factor: float
    drawdown: float
    score_final: float
    data_otimizacao: datetime

class AutoAjuste:
    def __init__(self, config, db_manager, logger):
        self.config = config
        self.db = db_manager
        self.logger = logger
        self.historico_otimizacoes = []
        self.parametros_atuais = {}
        self.melhor_resultado = None
        
        # Configurações adaptativas
        self.win_rate_minimo = 0.55
        self.fator_kelly = 0.5  # Fração de Kelly conservadora
        self.drawdown_maximo = 0.15  # 15% de drawdown máximo
        
        # Métricas de desempenho
        self.metricas = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'drawdown_atual': 0.0,
            'drawdown_maximo': 0.0,
            'resultado_dia': 0.0,
            'sequencia_atual': 0,
            'maior_sequencia_loss': 0
        }
        
        # Controle de horários
        self.horarios_ruins = set()
        self.melhores_horarios = {}

    def otimizar_parametros(self) -> ResultadoOtimizacao:
        """Realiza otimização completa dos parâmetros"""
        try:
            self.logger.info("Iniciando otimização de parâmetros...")
            
            # Obtém dados históricos
            dados = self.db.get_dados_treino()
            if dados.empty:
                raise ValueError("Dados insuficientes para otimização")
            
            # Divide dados em treino e validação
            dados_treino, dados_validacao = self._dividir_dados(dados)
            
            # Cria estudo Optuna
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler()
            )
            
            # Função objetivo para otimização
            def objetivo(trial):
                params = self._criar_parametros_trial(trial)
                return self._avaliar_parametros(params, dados_treino, dados_validacao)
            
            # Executa otimização
            study.optimize(objetivo, n_trials=100, timeout=3600)  # 1 hora máximo
            
            # Obtém melhores parâmetros
            melhores_params = study.best_params
            melhor_valor = study.best_value
            
            # Valida resultados
            resultado_final = self._validar_parametros(melhores_params, dados_validacao)
            
            # Registra resultado
            otimizacao = ResultadoOtimizacao(
                parametros=melhores_params,
                win_rate=resultado_final['win_rate'],
                profit_factor=resultado_final['profit_factor'],
                drawdown=resultado_final['drawdown'],
                score_final=melhor_valor,
                data_otimizacao=datetime.now()
            )
            
            self.historico_otimizacoes.append(otimizacao)
            self._atualizar_parametros(otimizacao)
            
            return otimizacao
            
        except Exception as e:
            self.logger.error(f"Erro na otimização: {str(e)}")
            return None

    def _dividir_dados(self, dados: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Divide dados em treino e validação"""
        ponto_divisao = int(len(dados) * 0.7)
        return dados[:ponto_divisao], dados[ponto_divisao:]

    def _criar_parametros_trial(self, trial) -> Dict:
        """Cria conjunto de parâmetros para teste"""
        return {
            'rsi_periodo': trial.suggest_int('rsi_periodo', 5, 21),
            'rsi_sobrevenda': trial.suggest_int('rsi_sobrevenda', 20, 35),
            'rsi_sobrecompra': trial.suggest_int('rsi_sobrecompra', 65, 80),
            'bb_periodo': trial.suggest_int('bb_periodo', 10, 30),
            'bb_desvio': trial.suggest_float('bb_desvio', 1.8, 2.8),
            'ma_curta': trial.suggest_int('ma_curta', 5, 21),
            'ma_media': trial.suggest_int('ma_media', 15, 50),
            'ma_longa': trial.suggest_int('ma_longa', 30, 100),
            'score_minimo': trial.suggest_float('score_minimo', 1.5, 3.0)
        }

    def _avaliar_parametros(self, params: Dict, dados_treino: pd.DataFrame, 
                          dados_validacao: pd.DataFrame) -> float:
        """Avalia um conjunto de parâmetros"""
        # Simula operações com os parâmetros
        resultados_treino = self._simular_operacoes(params, dados_treino)
        resultados_validacao = self._simular_operacoes(params, dados_validacao)
        
        # Calcula métricas
        score_treino = self._calcular_score(resultados_treino)
        score_validacao = self._calcular_score(resultados_validacao)
        
        # Penaliza overfitting
        diferenca_scores = abs(score_treino - score_validacao)
        score_final = score_validacao * (1 - diferenca_scores)
        
        return score_final

    def _simular_operacoes(self, params: Dict, dados: pd.DataFrame) -> Dict:
        """Simula operações com os parâmetros dados"""
        resultados = {
            'operacoes': [],
            'saldo': 1000,  # Saldo inicial
            'max_drawdown': 0
        }
        
        for i in range(len(dados)):
            sinais = self._gerar_sinais(dados.iloc[i:i+1], params)
            if sinais['operar']:
                resultado = self._simular_operacao(dados.iloc[i:], sinais['direcao'])
                resultados['operacoes'].append(resultado)
                resultados['saldo'] += resultado['resultado']
                
        return resultados

    def _calcular_score(self, resultados: Dict) -> float:
        """Calcula score final dos resultados"""
        if not resultados['operacoes']:
            return 0
            
        win_rate = len([op for op in resultados['operacoes'] if op['resultado'] > 0]) / len(resultados['operacoes'])
        profit_factor = resultados['saldo'] / 1000  # Relativo ao saldo inicial
        drawdown = resultados['max_drawdown']
        
        # Combina métricas
        score = (
            win_rate * 0.4 +
            profit_factor * 0.4 +
            (1 - drawdown) * 0.2
        )
        
        return score

    def _validar_parametros(self, params: Dict, dados: pd.DataFrame) -> Dict:
        """Valida conjunto de parâmetros em dados de validação"""
        resultados = self._simular_operacoes(params, dados)
        
        total_ops = len(resultados['operacoes'])
        if total_ops == 0:
            return {'win_rate': 0, 'profit_factor': 0, 'drawdown': 1}
            
        wins = len([op for op in resultados['operacoes'] if op['resultado'] > 0])
        
        return {
            'win_rate': wins / total_ops,
            'profit_factor': resultados['saldo'] / 1000,
            'drawdown': resultados['max_drawdown']
        }

    def _atualizar_parametros(self, otimizacao: ResultadoOtimizacao):
        """Atualiza parâmetros do sistema"""
        # Verifica se melhoria é significativa
        if (self.melhor_resultado is None or
            otimizacao.score_final > self.melhor_resultado.score_final * 1.05):
            
            self.parametros_atuais = otimizacao.parametros
            self.melhor_resultado = otimizacao
            
            # Atualiza configurações
            for param, valor in otimizacao.parametros.items():
                self.config.set(f'analise.{param}', valor)
            
            self.logger.info("Parâmetros atualizados com sucesso")

    def get_estatisticas(self) -> Dict:
        """Retorna estatísticas das otimizações"""
        if not self.historico_otimizacoes:
            return {}
            
        return {
            'total_otimizacoes': len(self.historico_otimizacoes),
            'ultima_otimizacao': self.historico_otimizacoes[-1].data_otimizacao,
            'melhor_win_rate': max(r.win_rate for r in self.historico_otimizacoes),
            'melhor_profit_factor': max(r.profit_factor for r in self.historico_otimizacoes),
            'menor_drawdown': min(r.drawdown for r in self.historico_otimizacoes)
        }