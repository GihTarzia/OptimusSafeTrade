import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import optuna
from sklearn.model_selection import TimeSeriesSplit

@dataclass
class ResultadoOtimizacao:
    parametros: Dict
    win_rate: float
    profit_factor: float
    drawdown: float
    sharpe_ratio: float
    data_otimizacao: datetime

class AutoOtimizador:
    def __init__(self, config, db_manager, logger):
        self.config = config
        self.db = db_manager
        self.logger = logger
        self.resultados_historicos = []
        
        # Ranges para otimização
        self.param_ranges = {
            'analise': {
                'rsi_periodo': (5, 21),
                'rsi_sobrevenda': (20, 35),
                'rsi_sobrecompra': (65, 80),
                'bb_periodo': (10, 30),
                'bb_desvio': (1.8, 2.8),
                'ma_curta': (5, 21),
                'ma_media': (15, 50),
                'ma_longa': (30, 100)
            },
            'operacional': {
                'min_score_entrada': (1.5, 3.0),
                'tempo_expiracao': (3, 15),
                'min_volatilidade': (0.0005, 0.002)
            }
        }

    def otimizar(self, periodo_dias: int = 30) -> Optional[ResultadoOtimizacao]:
        """Executa otimização completa dos parâmetros"""
        try:
            print("Iniciando processo de otimização...")
            
            # Carrega dados históricos
            dados = self.db.get_dados_historicos(dias=periodo_dias)
            if dados.empty:
                raise ValueError("Dados insuficientes para otimização")

            # Cria estudo Optuna
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler()
            )

            # Define função objetivo
            def objetivo(trial):
                params = self._criar_parametros_trial(trial)
                return self._avaliar_parametros(params, dados)

            # Executa otimização
            study.optimize(objetivo, n_trials=100)

            # Processa resultados
            melhor_resultado = ResultadoOtimizacao(
                parametros=study.best_params,
                win_rate=study.best_value,
                profit_factor=self._calcular_profit_factor(study.best_params, dados),
                drawdown=self._calcular_drawdown(study.best_params, dados),
                sharpe_ratio=self._calcular_sharpe_ratio(study.best_params, dados),
                data_otimizacao=datetime.now()
            )

            self.resultados_historicos.append(melhor_resultado)
            self._atualizar_configuracoes(melhor_resultado)

            return melhor_resultado

        except Exception as e:
            self.logger.error(f"Erro durante otimização: {str(e)}")
            return None

    def _criar_parametros_trial(self, trial) -> Dict:
        """Cria conjunto de parâmetros para teste"""
        params = {}
        
        # Parâmetros de análise
        for param, (min_val, max_val) in self.param_ranges['analise'].items():
            if 'periodo' in param or 'ma_' in param:
                params[param] = trial.suggest_int(param, min_val, max_val)
            else:
                params[param] = trial.suggest_float(param, min_val, max_val)
                
        # Parâmetros operacionais
        for param, (min_val, max_val) in self.param_ranges['operacional'].items():
            params[param] = trial.suggest_float(param, min_val, max_val)
            
        return params

    def _avaliar_parametros(self, params: Dict, dados: pd.DataFrame) -> float:
        """Avalia um conjunto de parâmetros"""
        # Usa validação cruzada temporal
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, test_idx in tscv.split(dados):
            train_data = dados.iloc[train_idx]
            test_data = dados.iloc[test_idx]
            
            # Simula operações
            resultados = self._simular_operacoes(params, test_data)
            scores.append(self._calcular_score(resultados))
        
        return np.mean(scores)

    def _simular_operacoes(self, params: Dict, dados: pd.DataFrame) -> Dict:
        """Simula operações com os parâmetros dados"""
        resultados = {
            'operacoes': [],
            'saldo': 1000,
            'max_drawdown': 0
        }
        
        for i in range(len(dados)):
            sinais = self._gerar_sinais(dados.iloc[i:i+1], params)
            if sinais['operar']:
                resultado = self._simular_operacao(
                    dados.iloc[i:i+params['tempo_expiracao']],
                    sinais['direcao']
                )
                resultados['operacoes'].append(resultado)
                resultados['saldo'] += resultado['resultado']
                
        return resultados

    def _calcular_score(self, resultados: Dict) -> float:
        """Calcula score final dos resultados"""
        if not resultados['operacoes']:
            return 0
            
        win_rate = len([op for op in resultados['operacoes'] if op['resultado'] > 0]) / len(resultados['operacoes'])
        profit_factor = self._calcular_profit_factor(resultados)
        drawdown = resultados['max_drawdown']
        
        # Combina métricas
        score = (
            win_rate * 0.4 +
            profit_factor * 0.4 +
            (1 - drawdown) * 0.2
        )
        
        return score

    def _atualizar_configuracoes(self, resultado: ResultadoOtimizacao):
        """Atualiza configurações do sistema com os melhores parâmetros"""
        try:
            # Atualiza configurações apenas se houver melhoria significativa
            if len(self.resultados_historicos) > 1:
                ultimo_resultado = self.resultados_historicos[-2]
                if resultado.win_rate < ultimo_resultado.win_rate * 1.05:
                    print("Melhoria insuficiente, mantendo parâmetros atuais")
                    return

            # Atualiza configurações
            for categoria, params in resultado.parametros.items():
                for param, valor in params.items():
                    self.config.set(f"{categoria}.{param}", valor)

            print("Configurações atualizadas com sucesso")

        except Exception as e:
            self.logger.error(f"Erro ao atualizar configurações: {str(e)}")

    def get_estatisticas(self) -> Dict:
        """Retorna estatísticas das otimizações"""
        if not self.resultados_historicos:
            return {}
            
        return {
            'total_otimizacoes': len(self.resultados_historicos),
            'ultima_otimizacao': self.resultados_historicos[-1].data_otimizacao,
            'melhor_win_rate': max(r.win_rate for r in self.resultados_historicos),
            'melhor_profit_factor': max(r.profit_factor for r in self.resultados_historicos),
            'menor_drawdown': min(r.drawdown for r in self.resultados_historicos)
        }