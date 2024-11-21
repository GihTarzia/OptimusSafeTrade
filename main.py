import sys
import traceback
import pandas as pd
import numpy as np
import ta
import asyncio
import yfinance as yf
from pathlib import Path
from datetime import datetime, time

# Adiciona o diretório raiz ao PATH
project_root = Path(__file__).parent
sys.path.append(str(project_root))
from tqdm import tqdm
from datetime import datetime, timedelta
from dataclasses import dataclass
from colorama import init, Fore, Style
from utils.notificador import Notificador
from typing import Dict, List, Optional
from models.ml_predictor import MLPredictor
from models.analise_padroes import AnalisePadroesComplexos
from models.gestao_risco import GestaoRiscoAdaptativo
from models.auto_ajuste import AutoAjuste
from utils.logger import TradingLogger
from utils.database import DatabaseManager
from config.parametros import Config

class Metricas:
    def __init__(self):
        self.metricas = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'drawdown': 0.0,
            'volume_operacoes': 0,
            'assertividade_media': 0.0,
            'tempo_medio_operacao': 0
        }
        self.historico_operacoes = []
        
    def atualizar(self, operacao: Dict):
        self.historico_operacoes.append(operacao)
        self._recalcular_metricas()
        
    def _recalcular_metricas(self):
        # Implementa cálculo das métricas
        pass
    
class TradingSystem:
    def __init__(self):
        self.logger = TradingLogger()
        self.logger.info(f"Iniciando Trading Bot...")
        self.db = DatabaseManager(self.logger)
        self.config = Config(self.logger)
        self.min_tempo_entre_analises= 5
        # Novos parâmetros de controle
        self.ultima_analise = {}  # Registro do momento da última análise por ativo

        # Estatísticas e histórico
        self.melhores_horarios = {}
        
        # Inicializa atributos que serão preenchidos posteriormente
        self.notificador = None
        self.ml_predictor = None
        self.analise_padroes = None
        self.gestao_risco = None
        self.auto_ajuste = None

    async def inicializar(self):
        """Inicializa componentes de forma assíncrona"""
        try:
            self.logger.debug("Iniciando inicialização dos componentes...")
            # Configura notificador
            token = self.config.get('notificacoes.telegram.token')
            chat_id = self.config.get('notificacoes.telegram.chat_id')
            self.notificador = Notificador(token, chat_id)
            self.logger.info("Notificador configurado com sucesso")

            # Inicializa componentes principais
            self.ml_predictor = MLPredictor(
                self.config,
                self.logger
            )
            self.analise_padroes = AnalisePadroesComplexos(self.config, self.logger)
            self.gestao_risco = GestaoRiscoAdaptativo(self.config.get('trading.saldo_inicial', 1000), self.logger)

            # Inicializa otimizadores
            self.logger.debug(f"\nConfigurando otimizadores...")
            self.auto_ajuste = AutoAjuste(self.config, self.db, self.logger, Metricas)
            self.logger.info("Componentes principais inicializados")

        except Exception as e:
            self.logger.critical(f"Erro na inicialização: {str(e)}")
            raise


    async def executar_backtest(self, dias: int = 30) -> Dict:
        """Executa backtesting com processamento otimizado"""
        self.logger.debug("Iniciando processo de backtesting...")
        timeout = 1800  # 30 minutos de timeout máximo

        try:
            # Carrega dados históricos
            dados = await self.db.get_dados_historicos(dias=dias)
            if dados.empty:
                self.logger.error("Sem dados suficientes para backtest")
                return {}
            
            # Adicionar verificação de dados mínimos
            if len(dados) < 20:  # Mínimo de 20 candles
                self.logger.error(f"Dados insuficientes para backtest: {len(dados)} candles")
                return {}
            
            # Agrupa dados por ativo
            dados_por_ativo = dados.groupby('ativo')
            resultados_por_ativo = {}
            # Cria tasks para processar cada ativo em paralelo
            tasks = []
            
            # Cria tasks para processar cada ativo em paralelo
            tasks = [
                asyncio.create_task(self._executar_backtest_ativo(ativo, dados_ativo))
                for ativo, dados_ativo in dados_por_ativo
            ]

            # Aguarda todas as tasks concluírem e obtém os resultados
            resultados_por_ativo = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=timeout
            )
            # Consolida resultados
            resultados_consolidados = self._consolidar_resultados_backtest(resultados_por_ativo)

            # Exibe e salva resultados
            if len(resultados_por_ativo) > 0:
                await self._salvar_resultados_backtest(resultados_consolidados)
                self._exibir_resultados_backtest(resultados_consolidados)
                return resultados_consolidados
            else:
                raise Exception("Nenhum resultado válido obtido no backtest")

            return resultados_consolidados

        except Exception as e:
            self.logger.error(f"Erro crítico durante backtesting: {str(e)}")
            return {}

    async def monitorar_desempenho(self):
        """Monitora desempenho e ajusta parâmetros"""
        while True:
            try:
                metricas = self.gestao_risco.get_estatisticas()
                
                # Verifica drawdown
                if metricas['metricas']['drawdown_atual'] > self.config.get('trading.max_drawdown'):
                    await self.pausar_operacoes()
                    await self.auto_ajuste.otimizar_parametros()
                
                # Verifica win rate
                #if metricas['metricas']['win_rate'] < self.config.get('trading.win_rate_minimo'):
                #    await self.auto_ajuste.ajustar_filtros('aumentar')
                
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Erro no monitoramento: {str(e)}")
                await asyncio.sleep(60)

    async def pausar_operacoes(self):
        """Pausa operações temporariamente"""
        self.operacoes_ativas = False
        await self.notificador.enviar_mensagem(
            "⚠️ Operações pausadas por atingir drawdown máximo"
        )
        
    def _consolidar_resultados_backtest(self, resultados_por_ativo: List[Dict]) -> Dict:
        resultados_consolidados = {
            'metricas_gerais': {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'drawdown_maximo': 0.0,
                'retorno_total': 0.0
            },
            'resultados_por_ativo': {}
        }

        for resultado in resultados_por_ativo:
            if resultado is None or 'ativo' not in resultado:
                continue
            
            ativo = resultado['ativo']
            resultados_consolidados['resultados_por_ativo'][ativo] = resultado

            resultados_consolidados['metricas_gerais']['total_trades'] += resultado['total_trades']
            resultados_consolidados['metricas_gerais']['wins'] += resultado['wins']
            resultados_consolidados['metricas_gerais']['losses'] += resultado['losses']
            resultados_consolidados['metricas_gerais']['drawdown_maximo'] = max(
                resultados_consolidados['metricas_gerais']['drawdown_maximo'],
                resultado['drawdown_maximo']
            )
            resultados_consolidados['metricas_gerais']['retorno_total'] += resultado['retorno_total']

        if resultados_consolidados['metricas_gerais']['total_trades'] > 0:
            resultados_consolidados['metricas_gerais']['win_rate'] = (
                resultados_consolidados['metricas_gerais']['wins'] / 
                resultados_consolidados['metricas_gerais']['total_trades']
            )
            resultados_consolidados['metricas_gerais']['profit_factor'] = (
                resultados_consolidados['metricas_gerais']['retorno_total'] / 
                abs(resultados_consolidados['metricas_gerais']['drawdown_maximo']) 
                if resultados_consolidados['metricas_gerais']['drawdown_maximo'] != 0 else float('inf')
            )

        return resultados_consolidados

    async def _executar_backtest_ativo(self, ativo: str, dados: pd.DataFrame) -> Dict:
        """Executa backtest para um ativo específico"""
        resultados = {
            'trades': [],
            'total_trades': 0,
            'wins': 0,
            'losses': 0
        }

        try:
            self.logger.debug(f"Iniciando backtest para {ativo}")
            
            for i in range(len(dados) - 1):
                dados_ate_momento = dados.iloc[:i+1]
                dados_futuros = dados.iloc[i+1:i+13]

                # Executa análises
                analise = await self._analisar_periodo(
                    ativo,
                    dados_ate_momento,
                    dados_futuros
                )

                if analise and analise.get('trade'):
                    trade = analise['trade']
                    resultados['trades'].append(trade)
                    resultados['total_trades'] += 1
                    
                    if trade.resultado == 'WIN':
                        resultados['wins'] += 1
                    else:
                        resultados['losses'] += 1

        except Exception as e:
            self.logger.error(f"Erro no backtest de {ativo}: {str(e)}")
            return resultados

        return resultados

    async def _analisar_periodo(self, ativo: str, dados_historicos: pd.DataFrame, 
                              dados_futuros: pd.DataFrame) -> Optional[Dict]:
        """Análise unificada de período para backtest"""
        try:
            # Análise ML
            sinal_ml = await self.ml_predictor.prever(dados_historicos, ativo)
            if not sinal_ml:
                return None

            # Análise padrões
            analise_tecnica = self.analise_padroes.analisar(
                dados=dados_historicos,
                ativo=ativo
            )
            if not analise_tecnica:
                return None

            # Validação do sinal
            if not self._validar_sinal(sinal_ml, analise_tecnica):
                return None

            # Simulação do trade
            trade = self._simular_trade(
                timestamp=dados_historicos.index[-1],
                dados_futuros=dados_futuros,
                sinal_ml=sinal_ml,
                analise_tecnica=analise_tecnica
            )

            return {'trade': trade} if trade else None

        except Exception as e:
            self.logger.error(f"Erro na análise do período: {str(e)}")
            return None


    def _exibir_resultados_backtest(self, resultados: Dict):
        """Exibe resultados do backtest"""
        self.logger.info(f"\n=== Resultados do Backtest ===")
        self.logger.info(f"Total de trades: {resultados['metricas_gerais']['total_trades']}")
        self.logger.info(f"Win Rate: {resultados['metricas_gerais']['win_rate']:.2%}")
        self.logger.info(f"Profit Factor: {resultados['metricas_gerais']['profit_factor']:.2f}")
        self.logger.info(f"Drawdown Máximo: {resultados['metricas_gerais']['drawdown_maximo']:.2f}%")
        self.logger.info(f"Retorno Total: {resultados['metricas_gerais']['retorno_total']:.2f}%")
        
        self.logger.info(f"\nMelhores Horários:")
        for hora, stats in resultados['melhores_horarios'].items():
            self.logger.info(f"• {hora}:00 - Win Rate: {stats['win_rate']:.2f}% ({stats['total_trades']} trades)")
    
    async def _salvar_resultados_backtest(self, resultados: Dict):
        """Salva resultados do backtest no banco de dados"""
        try:
            await self.db.salvar_resultados_backtest({
                'timestamp': datetime.now(),
                'metricas': resultados['metricas_gerais'],
                'melhores_horarios': resultados['melhores_horarios'],
                'evolucao_capital': resultados['evolucao_capital']
            })
        except Exception as e:
            self.logger.error(f"Erro ao salvar resultados do backtest: {str(e)}")
    
    def _validar_sinal(self, sinal_ml: Dict, analise_tecnica: Dict) -> bool:
        """Valida se sinal atende critérios mínimos"""
        try:
            
            if sinal_ml['direcao'] != analise_tecnica['direcao']:
                self.logger.warning(f"Sinal rejeitado: direção ML ({sinal_ml['direcao']}) != direção técnica ({analise_tecnica['direcao']})")
                return False
        
            # Score mínimo
            if sinal_ml['probabilidade'] < self.config.get('analise.min_score_entrada'):
                return False
            
            # Direções concordantes
            if sinal_ml['direcao'] != analise_tecnica['direcao']:
                return False

            # Volatilidade em range aceitável
            volatilidade = float(sinal_ml.get('volatilidade', 0))
            if volatilidade > 0.008 or volatilidade < 0.001:
                return False

            # Força mínima dos padrões
            if analise_tecnica.get('forca_sinal', 0) < 0.7:  # Mínimo de 70%
                return False

            # Confirmações técnicas
            if len(analise_tecnica.get('padroes', [])) < 3:  # Mínimo de 3 confirmações
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na validação: {str(e)}")
            return False

    def _simular_trade(self, timestamp: datetime, dados_futuros: pd.DataFrame, 
                      sinal_ml: Dict, analise_tecnica: Dict) -> Optional[Dict]:
        """Simula uma operação completa"""
        try:
            if dados_futuros.empty:
                return None
                
            preco_entrada = dados_futuros.iloc[0]['Open']
            tempo_exp = analise_tecnica.get('tempo_expiracao', 5)
            
            # Encontra candle de expiração
            idx_exp = min(int(tempo_exp * 12), len(dados_futuros) - 1)  # 12 candles = 1 hora
            if idx_exp < 1:
                return None
                
            preco_saida = dados_futuros.iloc[idx_exp]['Close']
            
            # Determina resultado
            if sinal_ml['direcao'] == 'CALL':
                resultado = 'WIN' if preco_saida > preco_entrada else 'LOSS'
            else:  # PUT
                resultado = 'WIN' if preco_saida < preco_entrada else 'LOSS'
                
            # Calcula lucro
            variacao = abs(preco_saida - preco_entrada) / preco_entrada
            lucro = variacao * 100 if resultado == 'WIN' else -variacao * 100
            
            return BacktestTrade(
                entrada_timestamp=timestamp,
                saida_timestamp=dados_futuros.index[idx_exp],
                ativo=sinal_ml['ativo'],
                direcao=sinal_ml['direcao'],
                preco_entrada=preco_entrada,
                preco_saida=preco_saida,
                resultado=resultado,
                lucro=lucro,
                score_entrada=sinal_ml['probabilidade'],
                assertividade_prevista=sinal_ml.get('score', 0)
            )
            
        except Exception as e:
            self.logger.error(f"Erro na simulação: {str(e)}")
            return None

    async def _notificar_resultado(self, operacao: Dict):
        """Envia notificação de resultado"""
        mensagem = self.notificador.formatar_resultado(operacao)
        await self.notificador.enviar_mensagem(mensagem)

    def calcular_timing_entrada(self, ativo: str, sinal: Dict) -> Dict:
        """Calcula o melhor momento para entrada baseado nos padrões históricos"""
        try:
            self.logger.debug(f"\nCalculando timing para {ativo}...")
            agora = datetime.now()
            
            # Analisa padrões de tempo mais favoráveis
            horarios_sucesso = self.db.get_horarios_sucesso(ativo)
            
            if not horarios_sucesso:
                self.logger.warning(f"Sem histórico de horários. Usando tempo padrão.")
                return {
                    'momento_ideal': agora + timedelta(minutes=1),
                    'tempo_espera': timedelta(minutes=1),
                    'taxa_sucesso_horario': 0.5
                }
            
            # Encontra melhor horário
            melhor_horario = None
            maior_taxa_sucesso = 0
            
            hora_atual = agora.time()
            for horario, taxa in horarios_sucesso.items():
                try:
                    horario_dt = datetime.strptime(horario, "%H:%M").time()
                    if taxa > maior_taxa_sucesso:
                        maior_taxa_sucesso = taxa
                        melhor_horario = horario_dt
                except ValueError:
                    continue
            
            # Calcula tempo de espera
            if melhor_horario:
                self.logger.info(f"Melhor horário encontrado: {melhor_horario.strftime('%H:%M')}")
                self.logger.info(f"Taxa de sucesso no horário: {maior_taxa_sucesso:.1%}")
                
                if hora_atual < melhor_horario:
                    tempo_espera = datetime.combine(agora.date(), melhor_horario) - datetime.combine(agora.date(), hora_atual)
                else:
                    tempo_espera = timedelta(minutes=1)
            else:
                self.logger.warning(f"Nenhum horário ótimo encontrado. Usando tempo padrão.")
                tempo_espera = timedelta(minutes=1)
            
            momento_entrada = agora + tempo_espera
            
            return {
                'momento_ideal': momento_entrada,
                'tempo_espera': tempo_espera,
                'taxa_sucesso_horario': maior_taxa_sucesso if maior_taxa_sucesso else 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular timing: {str(e)}")
            return {
                'momento_ideal': agora + timedelta(minutes=1),
                'tempo_espera': timedelta(minutes=1),
                'taxa_sucesso_horario': 0.5
            }
            
    def calcular_assertividade(self, ativo: str, sinal: Dict) -> float:
        """Calcula a probabilidade de sucesso do sinal"""
        try:
            self.logger.debug(f"\nCalculando assertividade para {ativo}...")
            
            # Componentes da assertividade
            prob_ml = sinal.get('ml_prob', 0)
            forca_padroes = float(sinal.get('score', 0))  # Usando 'score' como backup
            
            # Extrai probabilidade ML dos indicadores se disponível
            if 'indicadores' in sinal:
                prob_ml = float(sinal['indicadores'].get('ml_prob', prob_ml))
                forca_padroes = float(sinal['indicadores'].get('padroes_forca', forca_padroes))

            # Garante valores entre 0 e 1
            prob_ml = min(1.0, max(0.0, prob_ml))
            forca_padroes = min(1.0, max(0.0, forca_padroes))
            
            # Histórico específico para o tempo de expiração
            tempo_exp = sinal.get('tempo_expiracao', 5)
            historico = float(self.db.get_assertividade_recente(
                ativo, 
                sinal['direcao'],
                tempo_expiracao=tempo_exp
            ) or 50) / 100
            
            # Verifica momento do dia
            hora_atual = datetime.now().hour
            horarios_sucesso = self.db.get_horarios_sucesso(ativo)
            taxa_horario = horarios_sucesso.get(f"{hora_atual:02d}:00", 0.5)
            
            # Analisa volatilidade
            volatilidade = float(sinal.get('volatilidade', 0))
            volatilidade_score = 1.0
            if volatilidade > 0:
                if 0.001 <= volatilidade <= 0.005:  # Faixa ideal
                    volatilidade_score = 1.0
                elif 0.0005 <= volatilidade < 0.001:  # Baixa demais
                    volatilidade_score = 0.7
                elif 0.005 < volatilidade <= 0.01:  # Alta mas aceitável
                    volatilidade_score = 0.8
                else:  # Muito alta ou muito baixa
                    volatilidade_score = 0.5
            
            # Verifica tendência
            tendencia_match = sinal.get('tendencia') == sinal.get('direcao', '')
            tendencia_score = 1.2 if tendencia_match else 0.8
            
            # Cálculo ponderado com pesos ajustados
            base_score = (
                prob_ml * 0.40 +            # Probabilidade ML (40%)
                forca_padroes * 0.25 +      # Força dos padrões (25%)
                historico * 0.20 +          # Histórico recente (20%)
                volatilidade_score * 0.15 + # Score de volatilidade (15%)
                taxa_horario * 0.20         # Performance no horário
            )
            
            # Aplica multiplicador de tendência
            assertividade = base_score * tendencia_score

            # Limita entre 0 e 100
            assertividade = min(100, max(0, assertividade * 100))
            
            self.logger.info(f"Componentes da assertividade:")
            self.logger.info(f"ML: {prob_ml:.1%}")
            self.logger.info(f"Padrões: {forca_padroes:.1%}")
            self.logger.info(f"Histórico: {historico:.1%}")
            self.logger.info(f"Volatilidade Score: {volatilidade_score:.1%}")
            self.logger.info(f"Tendência Match: {tendencia_match}")
            self.logger.info(f"Score Final: {assertividade:.1f}%")
            self.logger.info(f"Taxa horário ({hora_atual}h): {taxa_horario:.1%}")
            
            return round(assertividade, 2)
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular assertividade: {str(e)}")
            return 0

    async def analisar_mercado(self):
        """Análise contínua do mercado"""
        ativos_falha = set()  # Conjunto para controlar ativos com problemas

        while True:
            try:
                # Obtém lista de ativos ativos
                ativos = self.config.get_ativos_ativos()
                
                # Remove ativos que falharam recentemente
                ativos_analise = [a for a in ativos if a not in ativos_falha]
                if not ativos_analise:
                    self.logger.warning("Nenhum ativo disponível para análise")
                    await asyncio.sleep(self.min_tempo_entre_analises * 2)
                    ativos_falha.clear()  # Limpa lista de falhas após espera
                    continue
            
                # Cria tasks apenas uma vez para cada ativo
                tasks = [
                    asyncio.create_task(self._analisar_ativo(ativo))
                    for ativo in ativos_analise
                ]

                # Executa análises em paralelo
                resultados = await asyncio.gather(*tasks, return_exceptions=True)

                # Processa resultados válidos
                for i, resultado in enumerate(resultados):
                    if isinstance(resultado, Exception):
                        self.logger.error(f"Erro ao analisar {ativos_analise[i]}: {str(resultado)}")
                        ativos_falha.add(ativos_analise[i])
                        self.logger.error(f"Stack trace completo: {traceback.format_exc()}")
                    elif resultado:
                        self.logger.info(f"Sinal gerado para {ativos_analise[i]}: {resultado}")
                        await self._processar_sinal(resultado)
                    else:
                        self.logger.warning(f"Nenhum sinal gerado para {ativos_analise[i]}")


                # Limpa ativos com falha periodicamente
                if len(ativos_falha) > 0 and len(resultados) % 10 == 0:
                    ativos_falha.clear()

                await asyncio.sleep(self.min_tempo_entre_analises)

            except Exception as e:
                self.logger.error(f"Erro no ciclo de análise: {str(e)}")
        
    async def _processar_sinal(self, sinal: Dict):
        """Processa sinal identificado"""
        try:
            
            # Valida horário novamente antes de processar o sinal
            if not self._validar_horario_operacao(datetime.now()):
                self.logger.warning("Sinal ignorado devido ao horário inadequado")
                return
        
            # Calcula melhor horário
            timing = self.calcular_timing_entrada(sinal['ativo'], sinal)

            # Calcula assertividade
            assertividade = self.calcular_assertividade(
                sinal['ativo'], 
                sinal
            )
            
            # Adiciona assertividade ao sinal
            sinal['assertividade'] = assertividade
        
            dados_mercado = await self.db.get_dados_mercado(sinal['ativo'])
            if dados_mercado.empty:
                self.logger.error(f"Não foi possível obter dados para {sinal['ativo']}")
                return None

            preco_entrada = dados_mercado['Close'].iloc[-1]
            volatilidade = dados_mercado['Close'].pct_change().std() * np.sqrt(252)

            # Registra sinal no banco de dados
            sinal_id = await self.db.registrar_sinal({
                'ativo': sinal['ativo'],
                'direcao': sinal['direcao'],
                'momento_entrada': timing['momento_ideal'],
                'tempo_expiracao': sinal.get('tempo_expiracao', 5),
                'score': sinal['score'],
                'assertividade': assertividade,
                'ml_prob': float(sinal['indicadores'].get('ml_prob', 0)),
                'padroes_forca': float(sinal['indicadores'].get('padroes_forca', 0)),
                'indicadores': sinal.get('indicadores', {}),
                'processado': False,
                'preco_entrada':preco_entrada,
                'volatilidade':volatilidade,
            })

            # Formata mensagem completa
            sinal_formatado = {
                'id': sinal_id,
                'ativo': sinal['ativo'],
                'direcao': sinal['direcao'],
                'momento_entrada': timing['momento_ideal'].strftime('%H:%M:%S'),
                'tempo_expiracao': sinal['tempo_expiracao'],
                'score': sinal['score'],
                'assertividade': assertividade,
                'indicadores': sinal['indicadores'],
                'preco_entrada': preco_entrada,
                'volatilidade': volatilidade,
            }

            # Notifica via telegram
            mensagem = self.notificador.formatar_sinal(sinal_formatado)
            await self.notificador.enviar_mensagem(mensagem)

        except Exception as e:
            self.logger.error(f"Erro ao processar sinal: {str(e)}")

    async def _analisar_ativo(self, ativo: str) -> Optional[Dict]:
        """Analisa um único ativo de forma assíncrona"""
        try:
            self.logger.debug(f"Iniciando análise de {ativo}")
            
            # Valida horário atual
            agora = datetime.now()
            if not self._validar_horario_operacao(agora):
                self.logger.warning(f"Horário não apropriado para análise de {ativo}")
                return None
            
            
            # Obtém dados do mercado de forma assíncrona
            dados_mercado = await self.db.get_dados_mercado(ativo)
            if dados_mercado is None or dados_mercado.empty:
                self.logger.warning(f"Sem dados para {ativo}")
                return None 

            # Análises em paralelo
            analises = await asyncio.gather(
                self.ml_predictor.prever(dados_mercado, ativo),
                self.analise_padroes.analisar(dados_mercado, ativo=ativo)
            )
            
            sinal_ml, analise_tecnica = analises
            
            if not sinal_ml or not analise_tecnica:
                return None
                
            # Combina análises
            sinal_combinado = self._combinar_analises(
                ativo, sinal_ml, analise_tecnica, dados_mercado
            )
            if not sinal_combinado:
                return None
            
            if sinal_combinado:
                self.ultima_analise[ativo] = datetime.now()
                
            # Retorna o sinal combinado no formato esperado por _processar_sinal
            return {
                'ativo': ativo,
                'direcao': sinal_combinado['direcao'],
                'score': sinal_combinado['score'],
                'tempo_expiracao': sinal_combinado['tempo_expiracao'],
                'indicadores': {
                    'ml_prob': sinal_combinado['ml_prob'],
                    'padroes_forca': sinal_combinado['padroes_forca'],
                    'tendencia': sinal_combinado['tendencia'],
                    'volatilidade': sinal_combinado['volatilidade'],
                    'score': sinal_combinado['score']
                },
                'volatilidade': sinal_combinado['volatilidade']
            }
            
        except Exception as e:
            self.logger.error(f"1Erro na análise de {ativo}: {str(e)}")
            return None

    def _combinar_analises(self, ativo: str, sinal_ml: Dict, analise_padroes: Dict, dados_mercado: pd.DataFrame) -> Dict:
        """Combina análises ML e técnica"""
        try:
            self.logger.debug(f"\nCombinando análises para {ativo}...")
            self.logger.info(f"Sinal ML: {sinal_ml}")
            self.logger.info(f"Análise Técnica: {analise_padroes}")            
            # Verifica dados de entrada
            if not all([sinal_ml, analise_padroes]):
                self.logger.warning(f"Dados insuficientes para análise completa")
                return None

            if dados_mercado is None or dados_mercado.empty:
                self.logger.warning(f"Sem dados de mercado disponíveis")
                return None

            # Verifica se temos as colunas necessárias
            if 'Close' not in dados_mercado.columns:
                self.logger.warning(f"Dados de mercado inválidos. Colunas disponíveis: {dados_mercado.columns.tolist()}")
                return None
            
            # Direção predominante
            direcao_ml = sinal_ml.get('direcao')
            direcao_padroes = analise_padroes.get('direcao')
            
            if not all([direcao_ml, direcao_padroes]):
                self.logger.warning(f"Direções não definidas")
                return None
            
            # Análise de tendência
            tendencia = self._analisar_tendencia(dados_mercado)
            
            # Normaliza scores individuais
            score_ml = min(sinal_ml.get('probabilidade', 0), 0.85)  # Limita em 85%
            score_padroes = min(analise_padroes.get('forca_sinal', 0), 0.75)  # Limita em 75%
            score_tendencia = min(tendencia.get('forca', 0), 0.6)  # Limita em 60%
                  
                  
            # NOVO: Bloqueio imediato se direção e tendência divergirem
            if tendencia['direcao'] != direcao_ml and tendencia['direcao'] != 'NEUTRO':
                self.logger.warning(f"Sinal descartado: divergência direção ({direcao_ml}) vs tendência ({tendencia['direcao']})")
                return None      
              
            # Score base ponderado
            score_final = (
                score_ml * 0.45 +          # 45% peso ML (reduzido)
                score_padroes * 0.25 +     # 25% peso padrões
                score_tendencia * 0.30     # 30% peso tendência (aumentado)
            )


            # Se chegou até aqui e a tendência for neutra, penaliza o score
            if tendencia['direcao'] == 'NEUTRO':
                score_final *= 0.8  # Penalização de 20% para tendência neutra


            # Bônus mais agressivos para concordância
            if direcao_ml == direcao_padroes:
                score_final *= 1.25  # +25% (aumentado)
            if tendencia['direcao'] == direcao_ml:
                score_final *= 0,6   # Penaliza em 40% quando há divergência

            # Limita score final
            score_final = min(0.95, max(0.1, score_final))

            # Calcula volatilidade corretamente
            volatilidade = dados_mercado['Close'].pct_change().rolling(20).std() * np.sqrt(252)
            volatilidade = float(volatilidade.iloc[-1]) if not volatilidade.empty else 0

            # Determina tempo de expiração baseado na volatilidade
            tempo_expiracao = self._calcular_tempo_expiracao(volatilidade)

            resultado = {
                'ativo': ativo,
                'direcao': direcao_ml,
                'score': score_final,
                'ml_prob': score_ml,
                'padroes_forca': score_padroes,
                'tendencia': tendencia['direcao'],
                'sinais': analise_padroes.get('padroes', []),
                'tempo_expiracao': tempo_expiracao,
                'volatilidade': volatilidade,
                'indicadores': {
                    'ml_prob': score_ml,
                    'padroes_forca': score_padroes,
                    'tendencia': tendencia['direcao']
                }
            }
            self.logger.info(f"Resultado combinado: {resultado}")
            return resultado
            
        except Exception as e:
            self.logger.error(f"\nErro ao combinar análises: {str(e)}")
            return None
    
    def _validar_horario_operacao(self, timestamp: datetime) -> bool:
        try:
            hora = timestamp.hour
            minuto = timestamp.minute
            horario_atual = timestamp.time()

            # Verifica período do dia
            is_madrugada = hora >= 0 and hora < 7
            is_noite = hora >= 20

            # Ajusta requisitos baseado no período
            min_taxa_sucesso = self.config.get('horarios.analise_horarios.win_rate_minimo_horario', 0.60)

            if is_madrugada or is_noite:
                min_taxa_sucesso *= 0.9  # Reduz requisito em 10% para horários alternativos
                self.logger.info(f"Operando em horário alternativo: {horario_atual} - Min taxa: {min_taxa_sucesso:.1%}")

            # Verifica taxa de sucesso
            taxa_sucesso = self.db.get_taxa_sucesso_horario(hora)

            if taxa_sucesso < min_taxa_sucesso:
                self.logger.warning(
                    f"Taxa de sucesso insuficiente para horário {hora}h: {taxa_sucesso:.1%} "
                    f"(mínimo: {min_taxa_sucesso:.1%})"
                )
                return False

            # Evita horários de alta volatilidade apenas em horário comercial
            if not (is_madrugada or is_noite):
                horarios_volateis = [
                    (8, 30, 9, 30),   # Abertura NY
                    (14, 30, 15, 30), # Fechamento Europa
                    (15, 45, 16, 15)  # Alta volatilidade NY
                ]

                for inicio_h, inicio_m, fim_h, fim_m in horarios_volateis:
                    inicio = time(inicio_h, inicio_m)
                    fim = time(fim_h, fim_m)
                    if inicio <= horario_atual <= fim:
                        self.logger.warning(f"Horário volátil detectado: {horario_atual}")
                        return False

            # Evita últimos 5 minutos de cada hora
            if minuto >= 55:
                self.logger.warning("Últimos 5 minutos da hora")
                return False

            self.logger.info(
                f"Horário validado: {horario_atual} "
                f"(Taxa sucesso: {taxa_sucesso:.1%}, "
                f"Min requerido: {min_taxa_sucesso:.1%})"
            )
            return True

        except Exception as e:
            self.logger.error(f"Erro ao validar horário: {str(e)}")
            return False

    
    def _analisar_tendencia(self, dados: pd.DataFrame) -> Dict:
        """Analisa a tendência atual do ativo"""
        try:
            if dados is None or dados.empty:
                return {'direcao': 'NEUTRO', 'forca': 0}
                
            # Certifica que estamos usando as colunas corretas
            if 'Close' not in dados.columns and 'close' not in dados.columns:
                self.logger.warning(f"Colunas disponíveis: {dados.columns.tolist()}")
                return {'direcao': 'NEUTRO', 'forca': 0}
                
                
            # Padroniza nome da coluna
            close_col = 'Close' if 'Close' in dados.columns else 'close'
            close = dados[close_col]
            
            # Calcula indicadores de tendência
            ema9 = ta.trend.EMAIndicator(close, window=9).ema_indicator()
            ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
            macd = ta.trend.MACD(close).macd()
            rsi = ta.momentum.RSIIndicator(close).rsi()
            
            # Analisa inclinações das médias (últimos 3 períodos para mais sensibilidade)
            inclinacao_9 = (ema9.iloc[-1] - ema9.iloc[-3]) / ema9.iloc[-3]
            inclinacao_21 = (ema21.iloc[-1] - ema21.iloc[-3]) / ema21.iloc[-3]
            
            # Reduz o limiar para detecção de tendência
            limiar_inclinacao = 0.0005  # 0.05% de variação

            # Analisa MACD
            macd_positivo = macd.iloc[-1] > 0
            macd_crescente = macd.iloc[-1] > macd.iloc[-2]
                           
            # Analisa RSI
            rsi_ultimo = rsi.iloc[-1]
            rsi_crescente = rsi.iloc[-1] > rsi.iloc[-2]

            # Sistema de pontos para determinar tendência
            pontos = 0

            # Análise da inclinação das médias
            if inclinacao_9 > limiar_inclinacao: pontos += 2
            if inclinacao_21 > limiar_inclinacao: pontos += 2
            if inclinacao_9 < -limiar_inclinacao: pontos -= 2
            if inclinacao_21 < -limiar_inclinacao: pontos -= 2

            # Análise do MACD
            if macd_positivo: pontos += 1
            else: pontos -= 1
            if macd_crescente: pontos += 1
            else: pontos -= 1

            # Análise do RSI
            if rsi_ultimo > 50 and rsi_crescente: pontos += 1
            if rsi_ultimo < 50 and not rsi_crescente: pontos -= 1

            # Calcula força da tendência (normalizada entre 0 e 1)
            forca = abs(pontos) / 8  # 8 é a pontuação máxima possível
            forca = min(1.0, max(0.0, forca))

            # Determina direção
            if pontos >= 2:
                return {'direcao': 'CALL', 'forca': forca}
            elif pontos <= -2:
                return {'direcao': 'PUT', 'forca': forca}
            else:
                return {'direcao': 'NEUTRO', 'forca': forca}
        
        except Exception as e:
            self.logger.error(f"Erro ao analisar tendência: {str(e)}")
            self.logger.error(f"Colunas disponíveis: {dados.columns.tolist() if dados is not None else 'None'}")
            return {'direcao': 'NEUTRO', 'forca': 0}
    
    async def verificar_resultados(self):
        """Verifica resultados das operações de forma otimizada"""
        try:
            while True:
                sinais_pendentes = await self.db.get_sinais_sem_resultado()
                self.logger.debug(f"Verificando {len(sinais_pendentes)} sinais pendentes")


                for sinal in sinais_pendentes:
                    try:
                        self.logger.info(f"\nProcessando sinal ID {sinal.get('id')} - {sinal.get('ativo')}")

                        # Verifica se timestamp já é datetime ou precisa converter
                        if isinstance(sinal['timestamp'], datetime):
                            momento_entrada = sinal['timestamp']
                        else:
                            momento_entrada = datetime.strptime(sinal['timestamp'], '%Y-%m-%d %H:%M:%S')
                  
                        tempo_expiracao = sinal['tempo_expiracao']
                        momento_expiracao = momento_entrada + timedelta(minutes=tempo_expiracao)

                        if datetime.now() > momento_expiracao:
                            self.logger.info(f"Sinal {sinal['id']} expirado, calculando resultado...")

                            # Busca preços
                            preco_entrada = sinal['preco_entrada']
                            preco_saida = await self.db.get_preco(sinal['ativo'], momento_expiracao)

                            if preco_entrada and preco_saida:
                                self.logger.info(f"Preços obtidos - Entrada: {preco_entrada}, Saída: {preco_saida}")

                                # Calcula resultado
                                if sinal['direcao'] == 'CALL':
                                    resultado = 'WIN' if preco_saida > preco_entrada else 'LOSS'
                                else:  # PUT
                                    resultado = 'WIN' if preco_saida < preco_entrada else 'LOSS'

                                # Calcula lucro fixo baseado na configuração
                                payout = self.config.get('trading.payout', 0.85)  # 85% padrão
                                valor_entrada = self.config.get('trading.valor_entrada', 100)
                                
                                lucro = valor_entrada * payout if resultado == 'WIN' else -valor_entrada

                                self.logger.info(f"Resultado calculado: {resultado} (lucro: {lucro})")

                                # Atualiza sinal no banco de dados
                                await self.db.atualizar_resultado_sinal(
                                    sinal['id'],
                                    resultado=resultado,
                                    lucro=lucro,
                                    preco_saida=preco_saida,
                                    data_processamento=datetime.now()
                                )

                                # Notifica resultado
                                await self._notificar_resultado({
                                    'ativo': sinal['ativo'],
                                    'direcao': sinal['direcao'],
                                    'resultado': resultado,
                                    'lucro': lucro,
                                    'preco_entrada': preco_entrada,
                                    'preco_saida': preco_saida,
                                    'id': sinal['id'],
                                })

                    except Exception as e:
                        self.logger.error(f"Erro ao processar sinal {sinal['id']}: {str(e)}")

                await asyncio.sleep(10)  # Espera 10 segundos entre verificações

        except Exception as e:
            self.logger.error(f"Erro no verificador de resultados: {str(e)}")      
            
    def _calcular_tempo_expiracao(self, volatilidade: float) -> int:
        """Define tempo de expiração para opções binárias"""
        try:
            if volatilidade < 0.003:  # Volatilidade baixa
                return 15  # Mais tempo para o movimento se desenvolver
            elif volatilidade > 0.008:  # Volatilidade muito alta
                return 1   # Tempo curto para evitar reversões
            elif volatilidade > 0.006:  # Volatilidade alta
                return 3   # Tempo moderado-curto
            else:  # Volatilidade ideal
                return 5   # Tempo padrão
                
        except Exception as e:
            self.logger.error(f"Erro ao definir tempo de expiração: {str(e)}")
            return 5
   
    # TradingSystem - Correção da função baixar_dados_historicos
    async def baixar_dados_historicos(self):
        """Baixa dados históricos iniciais para todos os ativos"""
        self.logger.debug(f"Iniciando download de dados históricos...")

        ativos = self.config.get_ativos_ativos()
        dados_salvos = False
        hoje = datetime.now()
        dfs = []

        for ativo in tqdm(ativos, desc="Baixando dados"):
            try:
                # Download de 30 dias de dados em intervalos de 1 minuto
                # Divide em 4 períodos de 7 dias para obter dados de 1 minuto
                for i in range(4):
                    end = hoje - timedelta(days=i*7)
                    start = end - timedelta(days=7)

                    df = yf.download(
                        ativo,
                        start=start,
                        end=end,
                        interval="1m",
                        progress=False
                    )
                    if not df.empty:
                        dfs.append(df)

                if len(dfs) > 0:  # Verifica se temos dados
                    dados_combinados = pd.concat(dfs).sort_index()
                    dados_combinados = dados_combinados[~dados_combinados.index.duplicated(keep='first')]

                    if not dados_combinados.empty:
                        dados_combinados.columns = [col if col == 'Volume' else col.title() for col in dados_combinados.columns]
                        dados_combinados = dados_combinados[['Open', 'High', 'Low', 'Close', 'Volume']]

                        sucesso = await self.db.salvar_precos(ativo, dados_combinados)
                        if sucesso:
                            dados_salvos = True
                            self.logger.info(f"Dados salvos com sucesso para {ativo}: {len(dados_combinados)} registros")
                    else:
                        self.logger.error(f"Erro ao salvar dados para {ativo}")
                else:
                    self.logger.warning(f"Nenhum dado disponível para {ativo}")

            except Exception as e:
                self.logger.error(f"Erro ao baixar dados para {ativo}: {str(e)}")

        if dados_salvos:
            self.logger.info(f"Download de dados históricos concluído com sucesso")
            return True
        else:
            self.logger.warning(f"Nenhum dado foi salvo durante o processo")
            return False

    async def executar(self):
        """Loop principal do sistema"""
        try:
            self.logger.info(f"\nIniciando sequência de inicialização...")

            # Inicializa componentes básicos
            await self.inicializar()

            # Fase 1: Baixa dados históricos
            self.logger.info(f"\nFase 1: Baixando dados históricos...")
            sucesso_download = await self.baixar_dados_historicos()
            if not sucesso_download:
                raise Exception("Falha ao baixar dados históricos")

            # Fase 2: Inicializa ML com dados baixados
            self.logger.info(f"\nFase 2: Inicializando modelos ML...")
            dados_historicos = await self.db.get_dados_historicos(dias=30)
            if dados_historicos.empty:
                self.logger.error(f"Sem dados históricos para treinar modelos")
                raise Exception("Sem dados históricos para treinar modelos")

            sucesso_ml = await self.ml_predictor.inicializar(dados_historicos)
            if not sucesso_ml:
                self.logger.error(f"Falha ao inicializar modelos ML")
                raise Exception("Falha ao inicializar modelos ML")

            # Fase 3: Executa backtesting
            #self.logger.info(f"\nFase 3: Executando backtesting...")
            #resultados_backtest = await self.executar_backtest(dias=30)

            # Fase 4: Inicia monitoramento
            self.logger.info(f"\nSistema pronto! Iniciando monitoramento contínuo...")

            # Cria tasks para monitoramento e verificação
            tasks = [
                asyncio.create_task(self.analisar_mercado()),
                asyncio.create_task(self.verificar_resultados()),
                asyncio.create_task(self.monitorar_desempenho())  # Adiciona monitoramento
            ]

            # Executa tasks
            await asyncio.gather(*tasks)

        except Exception as e:
            self.logger.error(f"Erro crítico: {str(e)}")
            raise
        except KeyboardInterrupt:
            self.logger.info("Sistema encerrado pelo usuário")
      
if __name__ == "__main__":
    init()  # Inicializa colorama
    
    # Limpa qualquer event loop residual
    if asyncio._get_running_loop() is not None:
        asyncio._set_running_loop(None)
    
    sistema = TradingSystem()
    
    try:
        asyncio.run(sistema.executar())
    except KeyboardInterrupt:
        print("Sistema encerrado pelo usuário")
    
    
@dataclass
class BacktestTrade:
    entrada_timestamp: datetime
    saida_timestamp: datetime
    ativo: str
    direcao: str  # 'CALL' ou 'PUT'
    preco_entrada: float
    preco_saida: float
    resultado: str  # 'WIN' ou 'LOSS'
    lucro: float
    score_entrada: float
    assertividade_prevista: float