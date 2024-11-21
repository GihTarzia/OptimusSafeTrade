import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import optuna
import json
from collections import deque
import asyncio

@dataclass
class ResultadoOtimizacao:
    parametros: Dict
    win_rate: float
    profit_factor: float
    drawdown: float
    score_final: float
    data_otimizacao: datetime
    volatilidade_media: float  # Novo campo
    tempo_medio_operacao: int  # Novo campo
    horarios_otimos: List[str] # Novo campo


    
class AutoAjuste:
    def __init__(self, config, db_manager, logger, metricas):
        self.config = config
        self.db = db_manager
        self.logger = logger
        self.metricas = metricas

        # Histórico com limite de memória
        self.historico_otimizacoes = deque(maxlen=100)
        self.parametros_atuais = {}
        self.melhor_resultado = None
        
        # Configurações adaptativas melhoradas
        self.configuracoes = {
            'win_rate_minimo': 0.58,  # Aumentado
            'fator_kelly': 0.3,       # Mais conservador
            'drawdown_maximo': 0.10,  # 10% máximo
            'volatilidade_min': 0.0002,
            'volatilidade_max': 0.006,
            'tempo_min_entre_sinais': 5,  # 5 minutos
            'max_sinais_hora': 10,
            'min_operacoes_validacao': 30
        }
    
        
        # Controle de horários e períodos
        self.periodos_analise = {
            'manha': ['09:00', '12:00'],
            'tarde': ['13:00', '16:00'],
            'noite': ['17:00', '20:00']
        }
        
        # Inicializa estudos Optuna
        self.estudos = {}
        self._inicializar_otimizadores()
        
    async def ajustar_filtros(self, direcao: str):
        """Ajusta os filtros de entrada com base na direção"""
        try:
            if direcao == 'aumentar':
                # Aumenta o limite mínimo de score de entrada
                self.config.set('analise.min_score_entrada', 
                               self.config.get('analise.min_score_entrada') + 0.05)
            else:
                # Diminui o limite mínimo de score de entrada
                self.config.set('analise.min_score_entrada',
                               self.config.get('analise.min_score_entrada') - 0.05)
        except Exception as e:
            self.logger.error(f"Erro ao ajustar filtros: {str(e)}")
            
    def _inicializar_otimizadores(self):
        """Inicializa otimizadores por período"""
        for periodo in self.periodos_analise:
            self.estudos[periodo] = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner()
            )

    async def _ajustar_filtros(self, direcao: str):
        """Ajusta filtros de entrada"""
        ajustes = {
            'analise.min_score_entrada': 0.05,
            'analise.min_confirmacoes': 1,
            'analise.min_volatilidade': 0.0001
        }

        for param, valor in ajustes.items():
            atual = self.config.get(param)
            novo = atual * (1 + valor) if direcao == 'aumentar' else atual * (1 - valor)
            self.config.set(param, novo)

    async def _ajustar_gestao_risco(self, direcao: str):
        """Ajusta parâmetros de gestão de risco"""
        ajustes = {
            'risco_por_operacao': 0.002,
            'stop_diario': 0.01,
            'max_operacoes_dia': 2
        }

        for param, valor in ajustes.items():
            atual = self.config.get(f'trading.{param}')
            novo = atual * (1 - valor) if direcao == 'reduzir' else atual * (1 + valor)
            self.config.set(f'trading.{param}', novo)

    async def _analisar_periodos(self):
        """Analisa performance por período do dia""" 
        for periodo, (inicio, fim) in self.periodos_analise.items():
            operacoes = await self.db.get_operacoes_periodo(inicio, fim)
            if len(operacoes) >= 20:
                metricas = self._calcular_metricas_periodo(operacoes)
                await self._otimizar_periodo(periodo, metricas)

    def _calcular_metricas_periodo(self, operacoes: List[Dict]) -> Dict:
        """Calcula métricas para um período específico"""
        total = len(operacoes)
        wins = len([op for op in operacoes if op['resultado'] == 'WIN'])

        return {
            'win_rate': wins / total if total > 0 else 0,
            'volume_medio': np.mean([op['volume'] for op in operacoes]),
            'tempo_medio': np.mean([op['duracao'].total_seconds() for op in operacoes]),
            'volatilidade': np.std([op['retorno'] for op in operacoes])
        }

    async def _otimizar_periodo(self, periodo: str, metricas: Dict):
        """Otimiza parâmetros para um período específico"""
        estudo = self.estudos[periodo]

        def objetivo(trial):
            params = self._criar_parametros_trial(trial)
            return self._avaliar_parametros_periodo(params, metricas)

        await asyncio.to_thread(
            estudo.optimize,
            objetivo,
            n_trials=50,
            timeout=1800
        )

        melhores_params = estudo.best_params
        self._atualizar_parametros_periodo(periodo, melhores_params)

    async def otimizar_parametros(self) -> Optional[ResultadoOtimizacao]:
        """Realiza otimização completa dos parâmetros com validação rigorosa"""
        try:
            self.logger.debug("Iniciando otimização de parâmetros...")

            # Obtém dados históricos
            dados = self.db.get_dados_treino()
            if dados.empty:
                self.logger.error("Dados insuficientes para otimização")
                raise ValueError("Dados insuficientes para otimização")

            # Valida quantidade mínima de dados
            if len(dados) < self.configuracoes['min_operacoes_validacao']:
                self.logger.warning("Dados insuficientes para otimização confiável")
                return None

            # Divide dados em treino e validação 
            dados_treino, dados_validacao = self._dividir_dados(dados)

            # Cria estudo Optuna com configurações melhoradas
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=10,
                    interval_steps=3
                )
            )

            # Função objetivo
            def objetivo(trial):
                params = self._criar_parametros_trial(trial)
                return self._avaliar_parametros(params, dados_treino, dados_validacao)

            # Executa otimização com timeout e callbacks
            study.optimize(
                objetivo,
                n_trials=100,
                timeout=3600,
                callbacks=[self._optuna_callback],
                show_progress_bar=True
            )

            # Análise dos resultados
            melhores_params = study.best_params
            melhor_valor = study.best_value

            # Validação extendida
            resultado_final = self._validar_parametros_extendido(
                melhores_params, 
                dados_validacao
            )

            # Análise de horários ótimos
            horarios_otimos = self._analisar_horarios_otimos(
                dados_validacao,
                melhores_params
            )

            # Cria resultado
            otimizacao = ResultadoOtimizacao(
                parametros=melhores_params,
                win_rate=resultado_final['win_rate'],
                profit_factor=resultado_final['profit_factor'], 
                drawdown=resultado_final['drawdown'],
                score_final=melhor_valor,
                data_otimizacao=datetime.now(),
                volatilidade_media=resultado_final['volatilidade_media'],
                tempo_medio_operacao=resultado_final['tempo_medio_operacao'],
                horarios_otimos=horarios_otimos
            )

            # Atualiza histórico
            self.historico_otimizacoes.append(otimizacao)

            # Atualiza parâmetros se houver melhoria significativa
            if self._validar_melhoria(otimizacao):
                self._atualizar_parametros(otimizacao)
                self.logger.info("Parâmetros atualizados com sucesso")
            else:
                self.logger.warning("Melhoria insuficiente, mantendo parâmetros atuais")

            return otimizacao

        except Exception as e:
            self.logger.error(f"Erro na otimização: {str(e)}")
            return None

    def _criar_parametros_trial(self, trial) -> Dict:
        """Cria conjunto de parâmetros para teste com ranges otimizados"""
        return {
            'analise': {
                'rsi': {
                    'periodo': trial.suggest_int('rsi_periodo', 7, 21),
                    'sobrevenda': trial.suggest_int('rsi_sobrevenda', 25, 35),
                    'sobrecompra': trial.suggest_int('rsi_sobrecompra', 65, 75)
                },
                'medias_moveis': {
                    'curta': trial.suggest_int('ma_curta', 5, 15),
                    'media': trial.suggest_int('ma_media', 15, 30),
                    'longa': trial.suggest_int('ma_longa', 30, 80)
                },
                'bandas_bollinger': {
                    'periodo': trial.suggest_int('bb_periodo', 12, 26),
                    'desvio': trial.suggest_float('bb_desvio', 1.8, 2.5)
                },
                'momentum': {
                    'periodo': trial.suggest_int('momentum_periodo', 8, 20),
                    'limite': trial.suggest_float('momentum_limite', 0.001, 0.005)
                }
            },
            'operacional': {
                'score_minimo': trial.suggest_float('score_minimo', 0.6, 0.85),
                'min_confirmacoes': trial.suggest_int('min_confirmacoes', 2, 4),
                'tempo_expiracao': trial.suggest_int('tempo_expiracao', 3, 10)
            }
        }

    def _validar_parametros_extendido(self, params: Dict, dados: pd.DataFrame) -> Dict:
        """Validação mais completa dos parâmetros"""
        resultados = self._simular_operacoes_completo(params, dados)

        if not resultados['operacoes']:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'drawdown': 1,
                'volatilidade_media': 0,
                'tempo_medio_operacao': 0
            }

        # Cálculos básicos
        total_ops = len(resultados['operacoes'])
        wins = len([op for op in resultados['operacoes'] if op['resultado'] > 0])

        # Cálculos avançados
        ganhos = sum(op['resultado'] for op in resultados['operacoes'] if op['resultado'] > 0)
        perdas = abs(sum(op['resultado'] for op in resultados['operacoes'] if op['resultado'] < 0))

        # Volatilidade
        retornos = [op['resultado'] / op['entrada'] for op in resultados['operacoes']]
        volatilidade = np.std(retornos) if retornos else 0

        # Tempo médio
        tempos = [(op['saida'] - op['entrada']).total_seconds() 
                 for op in resultados['operacoes']]
        tempo_medio = np.mean(tempos) if tempos else 0

        return {
            'win_rate': wins / total_ops,
            'profit_factor': ganhos / perdas if perdas > 0 else float('inf'),
            'drawdown': resultados['max_drawdown'],
            'volatilidade_media': volatilidade,
            'tempo_medio_operacao': int(tempo_medio)
        }

    def _analisar_horarios_otimos(self, dados: pd.DataFrame, params: Dict) -> List[str]:
        """Analisa horários com melhor performance"""
        resultados_hora = {}

        for hora in range(9, 21):  # 9h às 20h
            ops_hora = [op for op in dados['operacoes'] 
                       if op['timestamp'].hour == hora]

            if len(ops_hora) >= 10:  # Mínimo de operações para análise
                wins = len([op for op in ops_hora if op['resultado'] > 0])
                win_rate = wins / len(ops_hora)

                resultados_hora[hora] = {
                    'win_rate': win_rate,
                    'total_ops': len(ops_hora)
                }

        # Seleciona horários com win rate acima de 60%  
        horarios_otimos = [
            f"{hora:02d}:00"
            for hora, res in resultados_hora.items()
            if res['win_rate'] >= 0.6 and res['total_ops'] >= 20
        ]

        return horarios_otimos

    def _validar_melhoria(self, nova_otimizacao: ResultadoOtimizacao) -> bool:
        """Valida se nova otimização representa melhoria significativa"""
        if not self.melhor_resultado:
            return True

        # Critérios de melhoria
        melhorias = {
            'win_rate': nova_otimizacao.win_rate > self.melhor_resultado.win_rate * 1.05,
            'profit_factor': nova_otimizacao.profit_factor > self.melhor_resultado.profit_factor * 1.1,
            'drawdown': nova_otimizacao.drawdown < self.melhor_resultado.drawdown * 0.9,
            'score': nova_otimizacao.score_final > self.melhor_resultado.score_final * 1.05
        }

        # Precisa melhorar em pelo menos 2 critérios
        return sum(melhorias.values()) >= 2

    def _optuna_callback(self, study, trial):
        """Callback para monitoramento da otimização"""
        if trial.number % 10 == 0:
            self.logger.info(f"Trial {trial.number}: score = {trial.value:.4f}")

        # Salva estado intermediário
        if trial.number % 20 == 0:
            self._salvar_estado_otimizacao(study)

    def _salvar_estado_otimizacao(self, study):
        """Salva estado intermediário da otimização"""
        estado = {
            'timestamp': datetime.now().isoformat(),
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials)
        }

        try:
            with open('data/otimizacao_estado.json', 'w') as f:
                json.dump(estado, f, indent=2)
        except Exception as e:
            self.logger.error(f"Erro ao salvar estado: {str(e)}")

    

    def _necessita_otimizacao(self) -> bool:
        """Verifica se é necessário otimizar"""
        if not self.historico_otimizacoes:
            return True

        ultima = self.historico_otimizacoes[-1]
        tempo_passado = (datetime.now() - ultima.data_otimizacao).total_seconds()

        return (tempo_passado > 86400 or  # 24 horas
                ultima.win_rate < self.configuracoes['win_rate_minimo'] or
                ultima.drawdown > self.configuracoes['drawdown_maximo'])

    def _salvar_estado(self):
        """Salva estado atual do auto ajuste"""
        estado = {
            'timestamp': datetime.now().isoformat(),
            'parametros_atuais': self.parametros_atuais,
            'melhor_resultado': self.melhor_resultado._asdict() if self.melhor_resultado else None,
            'configuracoes': self.configuracoes,
            'metricas': self.metricas
        }

        try:
            self.db.salvar_estado_auto_ajuste(estado)
        except Exception as e:
            self.logger.error(f"Erro ao salvar estado: {str(e)}")

    def _simular_operacoes_completo(self, params: Dict, dados: pd.DataFrame) -> Dict:
        """Executa simulação completa para avaliação de parâmetros"""
        resultados = {
            'operacoes': [], 
            'max_drawdown': 0,
            'saldo': 1000 # Saldo inicial
        }

        # Simula operações 
        for i in range(len(dados) - 1):
            sinais = self._calcular_sinais(dados.iloc[i:i+1], params)

            for sinal in sinais:
                if self._validar_entrada(sinal, params):
                    op = self._simular_operacao(
                        sinal,
                        dados.iloc[i+1:i+params['operacional']['tempo_expiracao']]
                    )
                    if op:
                        resultados['operacoes'].append(op)

                        # Atualiza saldo e drawdown
                        resultados['saldo'] += op['resultado'] 
                        dd = (resultados['saldo'] - 1000) / 1000
                        resultados['max_drawdown'] = min(resultados['max_drawdown'], dd)

        return resultados

    def _calcular_sinais(self, dados: pd.DataFrame, params: Dict) -> List[Dict]:
        """Calcula sinais baseado nos parâmetros"""
        sinais = []

        # Análise RSI
        rsi = self._calcular_rsi(dados, params['analise']['rsi'])
        if rsi['sinal']:
            sinais.append(rsi)

        # Análise Médias Móveis
        mm = self._calcular_medias_moveis(dados, params['analise']['medias_moveis']) 
        if mm['sinal']:
            sinais.append(mm)

        # Análise Bollinger
        bb = self._calcular_bollinger(dados, params['analise']['bandas_bollinger'])
        if bb['sinal']:
            sinais.append(bb)

        # Análise Momentum
        mom = self._calcular_momentum(dados, params['analise']['momentum'])
        if mom['sinal']:
            sinais.append(mom)

        return sinais

    def _validar_entrada(self, sinal: Dict, params: Dict) -> bool:
        """Valida se sinal atende critérios mínimos"""
        return (
            sinal['score'] >= params['operacional']['score_minimo'] and
            sinal['confirmacoes'] >= params['operacional']['min_confirmacoes']
        )

    def _simular_operacao(self, sinal: Dict, dados_futuros: pd.DataFrame) -> Optional[Dict]:
        """Simula resultado de uma operação"""
        if dados_futuros.empty:
            return None

        preco_entrada = dados_futuros.iloc[0]['close']
        preco_saida = dados_futuros.iloc[-1]['close']

        resultado = (preco_saida - preco_entrada) if sinal['direcao'] == 'CALL' else (preco_entrada - preco_saida)

        return {
            'entrada': preco_entrada,
            'saida': preco_saida,
            'resultado': resultado,
            'timestamp': dados_futuros.index[0],
            'duracao': dados_futuros.index[-1] - dados_futuros.index[0],
            'retorno': resultado / preco_entrada,
            'volume': dados_futuros['volume'].mean() if 'volume' in dados_futuros else 0
        }

    def _atualizar_parametros(self, otimizacao: ResultadoOtimizacao):
        """Atualiza parâmetros do sistema com resultados otimizados"""
        self.parametros_atuais = otimizacao.parametros
        self.melhor_resultado = otimizacao

        # Atualiza configurações no sistema
        for categoria, params in otimizacao.parametros.items():
            if isinstance(params, dict):
                for param, valor in params.items():
                    self.config.set(f"{categoria}.{param}", valor)
            else:
                self.config.set(categoria, params)

        self._salvar_estado()