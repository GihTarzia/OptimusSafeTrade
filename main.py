import os
import sys
import schedule
import time
from datetime import datetime, timedelta
from colorama import init, Fore, Style
from pathlib import Path
from tqdm import tqdm
import yfinance as yf
from utils.notificador import Notificador
import pandas as pd
import numpy as np
from typing import Dict, List, Union
import ta
import asyncio

# Adiciona o diretório raiz ao PATH
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.auto_otimizador import AutoOtimizador
from models.ml_predictor import MLPredictor
from models.analise_padroes import AnalisePadroesComplexos
from models.gestao_risco import GestaoRiscoAdaptativo
from models.auto_ajuste import AutoAjuste
from utils.logger import TradingLogger
from utils.database import DatabaseManager
from config.parametros import Config

class TradingSystem:
    def __init__(self):
        print(f"{Fore.CYAN}Iniciando Trading Bot...{Style.RESET_ALL}")
        self.logger = TradingLogger()
        self.db = DatabaseManager()
        self.config = Config()

        # Inicializa o notificador
        token = self.config.get('notificacoes.telegram.token')
        chat_id = self.config.get('notificacoes.telegram.chat_id')
        self.notificador = Notificador(token, chat_id)
        
        # Inicializa componentes principais com barra de progresso
        componentes = [
            ("Logger", self.logger),
            ("Banco de Dados", self.db),
            ("Configurações", self.config),
            ("ML Predictor", MLPredictor()),
            ("Análise de Padrões", AnalisePadroesComplexos()),
            ("Gestão de Risco", GestaoRiscoAdaptativo(self.config.get('trading.saldo_inicial', 1000))),
        ]

        print(f"\n{Fore.YELLOW}Inicializando componentes...{Style.RESET_ALL}")
        for nome, componente in tqdm(componentes, desc="Progresso"):
            if nome == "ML Predictor":
                self.ml_predictor = componente
            elif nome == "Análise de Padrões":
                self.analise_padroes = componente
            elif nome == "Gestão de Risco":
                self.gestao_risco = componente

        # Inicializa otimizadores
        print(f"\n{Fore.YELLOW}Configurando otimizadores...{Style.RESET_ALL}")
        self.auto_otimizador = AutoOtimizador(self.config, self.db, self.logger)
        self.auto_ajuste = AutoAjuste(self.config, self.db, self.logger)
        
        # Estatísticas e histórico
        self.historico_sinais = {}
        self.assertividade_por_ativo = {}
        self.melhores_horarios = {}

    async def _notificar_sinal(self, sinal: Dict):
        """Envia notificação de sinal"""
        try:
            mensagem = self.notificador.formatar_sinal(sinal)
            await self.notificador.enviar_mensagem(mensagem)
        except Exception as e:
            self.logger.error(f"Erro ao notificar sinal: {str(e)}")

    async def _notificar_resultado(self, operacao: Dict):
        """Envia notificação de resultado"""
        mensagem = self.notificador.formatar_resultado(operacao)
        await self.notificador.enviar_mensagem(mensagem)

    def calcular_timing_entrada(self, ativo: str, sinal: Dict) -> Dict:
        """Calcula o melhor momento para entrada baseado nos padrões históricos"""
        try:
            print(f"\nCalculando timing para {ativo}...")
            agora = datetime.now()
            
            # Analisa padrões de tempo mais favoráveis
            horarios_sucesso = self.db.get_horarios_sucesso(ativo)
            
            if not horarios_sucesso:
                print(f"Sem histórico de horários. Usando tempo padrão.")
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
                print(f"Melhor horário encontrado: {melhor_horario.strftime('%H:%M')}")
                print(f"Taxa de sucesso no horário: {maior_taxa_sucesso:.1%}")
                
                if hora_atual < melhor_horario:
                    tempo_espera = datetime.combine(agora.date(), melhor_horario) - datetime.combine(agora.date(), hora_atual)
                else:
                    tempo_espera = timedelta(minutes=1)
            else:
                print(f"Nenhum horário ótimo encontrado. Usando tempo padrão.")
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
            print(f"\nCalculando assertividade para {ativo}...")
            
            # Componentes da assertividade
            prob_ml = sinal.get('ml_prob', 0)
            forca_padroes = sinal.get('padroes_força', 0)
            historico = self.db.get_assertividade_recente(ativo, sinal['direcao'])
            volatilidade_ok = 0.001 <= sinal.get('volatilidade', 0) <= 0.005
            
            # Pesos
            peso_ml = 0.4
            peso_padroes = 0.3
            peso_historico = 0.2
            peso_volatilidade = 0.1
            
            # Cálculo ponderado
            assertividade = (
                prob_ml * peso_ml +
                forca_padroes * peso_padroes +
                (historico/100) * peso_historico +
                (1 if volatilidade_ok else 0) * peso_volatilidade
            ) * 100
            
            print(f"Componentes da assertividade:")
            print(f"ML: {prob_ml:.1%}")
            print(f"Padrões: {forca_padroes:.1%}")
            print(f"Histórico: {historico:.1f}%")
            print(f"Volatilidade OK: {'Sim' if volatilidade_ok else 'Não'}")
            print(f"Assertividade final: {assertividade:.1f}%")
            
            return round(assertividade, 2)
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular assertividade: {str(e)}")
            return 0
        
    def baixar_dados_historicos(self):
        """Baixa dados históricos iniciais para todos os ativos"""
        print(f"{Fore.YELLOW}Iniciando download de dados históricos...{Style.RESET_ALL}")

        ativos = self.config.get_ativos_ativos()
        for ativo in tqdm(ativos, desc="Baixando dados"):
            try:
                # Download de 30 dias de dados em intervalos de 5 minutos
                dados = yf.download(
                    ativo,
                    period="30d",
                    interval="5m",
                    progress=False
                )

                if not dados.empty:
                    print(f"\nBaixados {len(dados)} registros para {ativo}")
                    #print("Colunas disponíveis:", dados.columns.tolist())
                    # Prepara os dados para salvar
                    dados_para_salvar = dados.reset_index()
                    dados_para_salvar['ativo'] = ativo

                    # Salva no banco de dados
                    self.db.salvar_precos(ativo, dados_para_salvar)
                    print(f"{Fore.GREEN}Dados baixados com sucesso para {ativo}: {len(dados)} registros{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Nenhum dado disponível para {ativo}{Style.RESET_ALL}")

            except Exception as e:
                print(f"{Fore.RED}Erro ao baixar dados para {ativo}: {str(e)}{Style.RESET_ALL}")

    def _avaliar_condicoes_mercado(self, ativo: str, dados: pd.DataFrame) -> Dict:
        """Avalia se o mercado está em boas condições para operar"""
        try:
            resultados = {
                'operar': False,
                'motivo': '',
                'score_mercado': 0
            }

            # 1. Volatilidade
            volatilidade = dados['close'].pct_change().std()
            volatilidade_ok = 0.0001 <= volatilidade <= 0.005  # ajuste estes valores

            # 2. Tendência definida
            ema_curta = ta.trend.EMAIndicator(dados['close'], window=9).ema_indicator()
            ema_longa = ta.trend.EMAIndicator(dados['close'], window=21).ema_indicator()
            tendencia_definida = abs((ema_curta.iloc[-1] - ema_longa.iloc[-1]) / ema_longa.iloc[-1]) > 0.0001

            # 3. Movimento consistente
            ultimos_candles = dados.tail(20)
            range_medio = (ultimos_candles['high'] - ultimos_candles['low']).mean()
            movimento_consistente = range_medio > dados['close'].iloc[-1] * 0.0003

            # 4. Não está lateralizado
            max_20 = dados['high'].tail(20).max()
            min_20 = dados['low'].tail(20).min()
            nao_lateral = (max_20 - min_20) / min_20 > 0.001

            # Calcula score do mercado
            score = 0
            if volatilidade_ok: score += 0.25
            if tendencia_definida: score += 0.25
            if movimento_consistente: score += 0.25
            if nao_lateral: score += 0.25

            resultados['score_mercado'] = score

            # Define se deve operar
            if score >= 0.75:  # pelo menos 3 condições atendidas
                resultados['operar'] = True
                resultados['motivo'] = "Mercado favorável"
            else:
                resultados['motivo'] = f"Mercado não ideal (score: {score:.2f})"
                if not volatilidade_ok: resultados['motivo'] += " - Volatilidade inadequada"
                if not tendencia_definida: resultados['motivo'] += " - Sem tendência clara"
                if not movimento_consistente: resultados['motivo'] += " - Movimento inconsistente"
                if not nao_lateral: resultados['motivo'] += " - Mercado lateral"


            # Registra métricas
            self.db.registrar_metricas_mercado(ativo, {
                'timestamp': datetime.now(),
                'volatilidade': volatilidade,
                'score_mercado': score,
                'range_medio': range_medio,
                'tendencia_definida': 1 if tendencia_definida else 0,
                'movimento_consistente': 1 if movimento_consistente else 0,
                'lateralizacao': 0 if nao_lateral else 1,
                'detalhes': {
                    'ema_diff': float((ema_curta.iloc[-1] - ema_longa.iloc[-1]) / ema_longa.iloc[-1]),
                    'max_20': float(max_20),
                    'min_20': float(min_20)
                }
            })

            return resultados

        except Exception as e:
            return {'operar': False, 'motivo': f"Erro na análise: {str(e)}", 'score_mercado': 0}
    

    async def analisar_mercado(self):
        """Análise principal do mercado"""
        print(f"\n{Fore.CYAN}=== Nova Análise ==={Style.RESET_ALL}")
        print(f"Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        sinais_validos = []
        ativos_analisados = 0
        
        for ativo in self.config.get_ativos_ativos():
            try:
                print(f"\n{Fore.CYAN}Analisando {ativo}...{Style.RESET_ALL}")
                ativos_analisados += 1
                
                # Verifica horário de operação
                if not self.config.is_horario_operacional():
                    print(f"{Fore.YELLOW}Fora do horário operacional{Style.RESET_ALL}")
                    return
                
                # Análise ML
                if ativo not in self.ml_predictor.models:
                    print(f"{Fore.YELLOW}Modelo não disponível para {ativo}{Style.RESET_ALL}")
                    continue
                    
                sinal_ml = self.ml_predictor.analisar(ativo)
                if not sinal_ml:
                    print(f"{Fore.YELLOW}Sem sinal ML para {ativo}{Style.RESET_ALL}")
                    continue
                if sinal_ml:
                    print(f"Sinal ML obtido - Direção: {sinal_ml.get('direcao')} - Prob: {sinal_ml.get('probabilidade', 0):.2%}")
          
                
                # Análise de padrões
                dados_mercado = self.db.get_dados_mercado(ativo, limite=100)
                analise_padroes = self.analise_padroes.analisar(ativo)

                # Avalia condições do mercado
                condicoes = self._avaliar_condicoes_mercado(ativo, dados_mercado)
                print(f"Condições de mercado: {condicoes['motivo']}")
                print(f"Score do mercado: {condicoes['score_mercado']:.2f}")
                
                # Se mercado não está bom, ainda treina ML mas não gera sinais
                if not condicoes['operar']:
                    print(f"{Fore.YELLOW}Mercado não favorável para {ativo} - Apenas atualizando ML{Style.RESET_ALL}")
                    # Aqui ainda faz o treino do ML para manter o aprendizado
                    if ativo in self.ml_predictor.models:
                        self.ml_predictor.atualizar_modelo(ativo, dados_mercado)
                    continue

                if not analise_padroes:
                    print(f"{Fore.YELLOW}Sem padrões detectados para {ativo}{Style.RESET_ALL}")
                    continue
                
                # Combina análises
                sinal_combinado = self._combinar_analises(
                    ativo, sinal_ml, analise_padroes, dados_mercado
                )
                
                if not sinal_combinado:
                    print(f"{Fore.YELLOW}Sem sinal combinado para {ativo}{Style.RESET_ALL}")
                    continue
                
                if sinal_combinado['score'] >= self.config.get('analise.min_score_entrada', 0.7):
                    print(f"Score suficiente: {sinal_combinado['score']:.2%} >= {self.config.get('analise.min_score_entrada', 0.7):.2%}")

                    # Calcula timing e assertividade
                    timing = self.calcular_timing_entrada(ativo, sinal_combinado)
                    assertividade = self.calcular_assertividade(ativo, sinal_combinado)    
                    
                    print(f"Timing calculado - Momento: {timing['momento_ideal'].strftime('%H:%M:%S')}")
                    print(f"Assertividade calculada: {assertividade:.1f}%")

                    # Aplica filtros de qualidade
                    if self._validar_sinal(ativo, sinal_combinado, timing, assertividade):
                        print(f"Sinal validado com sucesso")

                        # Calcula risco
                        risco = self.gestao_risco.calcular_risco_operacao(
                            ativo,
                            sinal_combinado['score'],
                            assertividade
                        )
                        
                        if risco:
                            sinais_validos.append({
                                'ativo': ativo,
                                'sinal': sinal_combinado,
                                'timing': timing,
                                'assertividade': assertividade,
                                'risco': risco,
                                'score_final': self._calcular_score_final(
                                    sinal_combinado, timing, assertividade
                                )
                            })
                            print(f"{Fore.GREEN}Sinal válido encontrado para {ativo}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.YELLOW}Sinal não passou na validação{Style.RESET_ALL}")
                        continue
                else:
                    print(f"{Fore.YELLOW}Score insulficiente {sinal_combinado['score']} para {ativo}, precisa ser maior igual a {self.config.get('analise.min_score_entrada', 0.7)}{Style.RESET_ALL}")
                    continue
            except Exception as e:
                self.logger.error(f"Erro ao analisar {ativo}: {str(e)}")
        
        # Exibe resumo da análise
        print(f"\n{Fore.CYAN}=== Resumo da Análise ==={Style.RESET_ALL}")
        print(f"Ativos analisados: {ativos_analisados}")
        print(f"Sinais válidos encontrados: {len(sinais_validos)}")
        
        if sinais_validos:
            # Ordena sinais por score final
            sinais_validos.sort(key=lambda x: x['score_final'], reverse=True)
            
            # Exibe resultados
            self._exibir_resumo_analise(sinais_validos)
            #self._notificar_resultado(sinais_validos)

            # Registra sinais válidos
            for sinal in sinais_validos:
                    try:
                        dados_registro = self._formatar_sinal_para_registro(sinal)
                        self.registrar_sinal(**dados_registro)
                        await self._notificar_sinal(dados_registro)
                    except Exception as e:
                        print(f"Erro ao processar sinal: {str(e)}")
                        return ""
        else:
            print(f"\n{Fore.YELLOW}Nenhum sinal válido encontrado neste momento{Style.RESET_ALL}")
        
        print(f"\nPróxima análise em 30 segundos...")

    # Antes de chamar registrar_sinal, vamos ajustar o formato dos dados
    def _formatar_sinal_para_registro(self, sinal_dados: Dict) -> Dict:
        """Formata os dados do sinal para registro"""
        return {
            'ativo': sinal_dados['ativo'],
            'direcao': sinal_dados['sinal']['direcao'],
            'momento_entrada': sinal_dados['timing']['momento_ideal'],
            'tempo_expiracao': sinal_dados['sinal']['tempo_expiracao'],
            'score': sinal_dados['sinal']['score'],
            'assertividade': sinal_dados['assertividade'],
            'padroes': sinal_dados['sinal']['sinais'],
            'indicadores': {
                'ml_prob': sinal_dados['sinal']['ml_prob'],
                'padroes_forca': sinal_dados['sinal']['padroes_força'],
                'tendencia': sinal_dados['sinal']['tendencia']
            },
            'ml_prob': sinal_dados['sinal']['ml_prob'],
            'volatilidade': sinal_dados['sinal']['volatilidade']
        }

    def _combinar_analises(self, ativo: str, sinal_ml: Dict, analise_padroes: Dict, dados_mercado: pd.DataFrame) -> Dict:
        """Combina análises ML e técnica"""
        try:
            print(f"\nCombinando análises para {ativo}...")
            
            # Verifica dados de entrada
            if not all([sinal_ml, analise_padroes]):
                print(f"Dados insuficientes para análise completa")
                return None

            if dados_mercado is None or dados_mercado.empty:
                print(f"Sem dados de mercado disponíveis")
                return None

            # Verifica se temos as colunas necessárias
            if 'close' not in dados_mercado.columns:
                print(f"\n{Fore.MAGENTA}Dados de mercado inválidos. Colunas disponíveis: {dados_mercado.columns.tolist()}{Style.RESET_ALL}")
                return None
            
            # Direção predominante
            direcao_ml = sinal_ml.get('direcao')
            direcao_padroes = analise_padroes.get('direcao')
            
            if not all([direcao_ml, direcao_padroes]):
                print(f"\n{Fore.MAGENTA}Direções não definidas{Style.RESET_ALL}")
                return None
            
            # Score base (média ponderada)
            score_ml = sinal_ml.get('probabilidade', 0) * 0.4  # 40% peso
            score_padroes = analise_padroes.get('forca_sinal', 0) * 0.4  # 40% peso
            
            # Análise de tendência
            tendencia = self._analisar_tendencia(dados_mercado)
            score_tendencia = tendencia.get('forca', 0) * 0.2  # 20% peso
            
            print(f"Scores parciais:")
            print(f"ML: {score_ml:.2%}")
            print(f"Padrões: {score_padroes:.2%}")
            print(f"Tendência: {score_tendencia:.2%}")
            
            # Score final
            score_final = score_ml + score_padroes + score_tendencia
            
            # Determina direção final
            if direcao_ml == direcao_padroes:
                direcao_final = direcao_ml
                score_final *= 1.2  # Bônus por concordância
                print(f"Bônus de concordância aplicado")
            else:
                # Se direções diferentes, usa a mais forte
                direcao_final = direcao_ml if score_ml > score_padroes else direcao_padroes
                score_final *= 0.8  # Penalidade por divergência
                print(f"Penalidade por divergência aplicada")
            
            # Ajusta score pela tendência
            if tendencia['direcao'] == direcao_final:
                score_final *= 1.1  # Bônus por seguir tendência
                print(f"Bônus de tendência aplicado")
            
            score_final = min(score_final, 1.0)  # Limita a 1.0
            
            print(f"Score final: {score_final:.2%}")
            print(f"Direção final: {direcao_final}")
            
            return {
                'ativo': ativo,
                'direcao': direcao_final,
                'score': score_final,
                'ml_prob': sinal_ml.get('probabilidade', 0),
                'padroes_força': analise_padroes.get('forca_sinal', 0),
                'tendencia': tendencia['direcao'],
                'sinais': analise_padroes.get('padroes', []),
                'tempo_expiracao': analise_padroes.get('tempo_expiracao', 5),
                'volatilidade': sinal_ml.get('volatilidade', 0)
            }
            
        except Exception as e:
            self.logger.error(f"\n{Fore.RED}Erro ao combinar análises: {str(e)}{Style.RESET_ALL}")
            return None
        
    def _exibir_resumo_analise(self, sinais_validos: List[Dict]):
        """Exibe resumo da análise atual"""
        try:
            print(f"\n{Fore.CYAN}=== Resumo da Análise ==={Style.RESET_ALL}")
            print(f"Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            if not sinais_validos:
                print(f"\n{Fore.YELLOW}Nenhum sinal válido encontrado{Style.RESET_ALL}")
                return
            
            print(f"\n{Fore.GREEN}Sinais encontrados: {len(sinais_validos)}{Style.RESET_ALL}")
            print("-"*80)
            
            # Exibe top 5 sinais
            for i, sinal in enumerate(sinais_validos[:5], 1):
                self._exibir_sinal_detalhado(sinal, ranking=i)
            
            # Estatísticas gerais
            self._exibir_estatisticas_gerais(sinais_validos)
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}Erro ao exibir resumo: {str(e)}{Style.RESET_ALL}")

    def _exibir_sinal_detalhado(self, sinal: Dict, ranking: int = None):
        """Exibe detalhes de um sinal específico"""
        try:
            ativo = sinal['ativo']
            s = sinal['sinal']
            timing = sinal['timing']
            
            # Cabeçalho
            if ranking:
                print(f"\n{Fore.CYAN}#{ranking} - Análise Detalhada: {ativo}{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.CYAN}Análise Detalhada: {ativo}{Style.RESET_ALL}")
            print("-"*80)
            
            # Direção e Timing
            cor_direcao = Fore.GREEN if s['direcao'] == 'CALL' else Fore.RED
            print(f"Direção: {cor_direcao}{s['direcao']}{Style.RESET_ALL}")
            print(f"Entrada: {timing['momento_ideal'].strftime('%H:%M:%S')}")
            print(f"Expiração: {s['tempo_expiracao']} minutos")
            
            # Scores e Probabilidades
            print(f"\n{Fore.YELLOW}Indicadores de Qualidade:{Style.RESET_ALL}")
            print(f"Score Final: {sinal['score_final']:.2%}")
            print(f"Assertividade: {sinal['assertividade']:.1f}%")
            print(f"Probabilidade ML: {s['ml_prob']:.1%}")
            print(f"Força dos Padrões: {s['padroes_força']:.1%}")
            
            # Gestão de Risco
            print(f"\n{Fore.YELLOW}Gestão de Risco:{Style.RESET_ALL}")
            print(f"Valor Sugerido: ${sinal['risco']['valor_risco']:.2f}")
            print(f"Stop Loss: ${sinal['risco']['stop_loss']:.2f}")
            print(f"Take Profit: ${sinal['risco']['take_profit']:.2f}")
            
            # Padrões Detectados
            if s['sinais']:
                print(f"\n{Fore.YELLOW}Padrões Detectados:{Style.RESET_ALL}")
                for padrao in s['sinais']:
                    print(f"• {padrao['nome']} ({padrao['direcao']}) - Força: {padrao['forca']:.1%}")
            
            # Informações Adicionais
            print(f"\n{Fore.YELLOW}Informações Adicionais:{Style.RESET_ALL}")
            print(f"Tendência: {s['tendencia']}")
            print(f"Volatilidade: {s['volatilidade']:.2%}")
            print(f"Taxa de Sucesso no Horário: {timing['taxa_sucesso_horario']:.1%}")
            
            print("-"*80)
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}Erro ao exibir sinal detalhado: {str(e)}{Style.RESET_ALL}")

    def _exibir_estatisticas_gerais(self, sinais_validos: List[Dict]):
        """Exibe estatísticas gerais da análise atual"""
        try:
            print(f"\n{Fore.CYAN}=== Estatísticas Gerais ==={Style.RESET_ALL}")
            print("-"*80)
            
            # Distribuição por direção
            calls = len([s for s in sinais_validos if s['sinal']['direcao'] == 'CALL'])
            puts = len([s for s in sinais_validos if s['sinal']['direcao'] == 'PUT'])
            
            print("Distribuição de Sinais:")
            print(f"CALL: {calls} ({calls/len(sinais_validos)*100:.1f}%)")
            print(f"PUT: {puts} ({puts/len(sinais_validos)*100:.1f}%)")
            
            # Médias
            scores = [s['score_final'] for s in sinais_validos]
            assertividades = [s['assertividade'] for s in sinais_validos]
            
            print(f"\nMédia de Scores: {np.mean(scores):.2%}")
            print(f"Média de Assertividade: {np.mean(assertividades):.1f}%")
            
            # Performance do dia
            stats_dia = self.gestao_risco.get_estatisticas()
            if stats_dia['total_operacoes'] > 0:
                print(f"\nPerformance do Dia:")
                print(f"Operações: {stats_dia['total_operacoes']}")
                print(f"Win Rate: {stats_dia['win_rate']:.1f}%")
                print(f"Resultado: {stats_dia['resultado_dia']:+.2f}%")
            
            print("-"*80)
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}Erro ao exibir estatísticas: {str(e)}{Style.RESET_ALL}")
   
    def _analisar_tendencia(self, dados: pd.DataFrame) -> Dict:
        """Analisa a tendência atual do ativo"""
        try:
            if dados is None or dados.empty:
                return {'direcao': 'NEUTRO', 'forca': 0}
                
            # Certifica que estamos usando as colunas corretas
            if 'close' not in dados.columns:
                print(f"Colunas disponíveis: {dados.columns.tolist()}")
                return {'direcao': 'NEUTRO', 'forca': 0}
                
            # Calcula médias móveis
            close = dados['close']
            ema9 = ta.trend.EMAIndicator(close, window=9).ema_indicator()
            ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
            ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
            
            # Verifica se temos dados suficientes
            if ema50.isna().any():
                return {'direcao': 'NEUTRO', 'forca': 0}
            
            # Verifica alinhamento das médias
            ultimo_ema9 = ema9.iloc[-1]
            ultimo_ema21 = ema21.iloc[-1]
            ultimo_ema50 = ema50.iloc[-1]
            ultimo_preco = close.iloc[-1]
            
            # Determina direção e força
            if ultimo_ema9 > ultimo_ema21 > ultimo_ema50 and ultimo_preco > ultimo_ema9:
                return {'direcao': 'CALL', 'forca': 1.0}
            elif ultimo_ema9 < ultimo_ema21 < ultimo_ema50 and ultimo_preco < ultimo_ema9:
                return {'direcao': 'PUT', 'forca': 1.0}
            elif ultimo_ema9 > ultimo_ema21:
                return {'direcao': 'CALL', 'forca': 0.7}
            elif ultimo_ema9 < ultimo_ema21:
                return {'direcao': 'PUT', 'forca': 0.7}
            else:
                return {'direcao': 'NEUTRO', 'forca': 0.5}
                    
        except Exception as e:
            print(f"{Fore.RED}Erro ao analisar tendência: {str(e)}{Style.RESET_ALL}")
            print(f"Colunas disponíveis: {dados.columns.tolist() if dados is not None else 'None'}")
            return {'direcao': 'NEUTRO', 'forca': 0}

    
    def _validar_sinal(self, ativo: str, sinal: Dict, timing: Dict, assertividade: float) -> bool:
        """Valida se o sinal atende todos os critérios mínimos"""
        try:
            # Critérios mínimos
            min_assertividade = self.config.get('analise.min_assertividade', 65)
            min_score = self.config.get('analise.min_score_entrada', 0.7)
            min_taxa_horario = 0.55
            
            if assertividade < min_assertividade:
                print(f"Assertividade insuficiente: {assertividade:.1f}% (mín: {min_assertividade}%)")
                return False
                
            if sinal['score'] < min_score:
                print(f"Score insuficiente: {sinal['score']:.2f} (mín: {min_score})")
                return False
                
            if timing['taxa_sucesso_horario'] < min_taxa_horario:
                print(f"Taxa de sucesso no horário insuficiente: {timing['taxa_sucesso_horario']:.1%} (mín: {min_taxa_horario:.1%})")
                return False
            
            # Verifica volatilidade
            min_vol = self.config.get('analise.min_volatilidade', 0.001)
            max_vol = self.config.get('analise.max_volatilidade', 0.005)
            
            if not (min_vol <= sinal['volatilidade'] <= max_vol):
                print(f"Volatilidade fora do range: {sinal['volatilidade']:.4f} (range: {min_vol}-{max_vol})")
                return False
            
            print(f"{Fore.GREEN}Todos os critérios atendidos{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao validar sinal: {str(e)}")
            return False
        
    def _calcular_score_final(self, sinal: Dict, timing: Dict, assertividade: float) -> float:
        """Calcula score final considerando todos os fatores"""
        try:
            # Pesos dos componentes
            peso_sinal = 0.4
            peso_timing = 0.3
            peso_assertividade = 0.3
            
            # Normaliza valores
            score_sinal = sinal['score']
            score_timing = timing['taxa_sucesso_horario']
            score_assertividade = assertividade / 100
            
            # Calcula score ponderado
            score_final = (
                score_sinal * peso_sinal +
                score_timing * peso_timing +
                score_assertividade * peso_assertividade
            )
            
            return round(score_final, 4)
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular score final: {str(e)}")
            return 0

    def exibir_resultado(self, ativo, sinal, timing, assertividade, risco):
        """Exibe o resultado da análise de forma formatada"""
        cor = Fore.GREEN if sinal['direcao'] == 'CALL' else Fore.RED
        print(f"\n{Fore.CYAN}Análise para {ativo}:{Style.RESET_ALL}")
        print(f"Direção: {cor}{sinal['direcao']}{Style.RESET_ALL}")
        print(f"Momento ideal de entrada: {timing['momento_ideal'].strftime('%H:%M:%S')}")
        print(f"Tempo de expiração: {self.config.get('trading.tempo_expiracao_padrao')} minutos")
        print(f"Assertividade esperada: {Fore.YELLOW}{assertividade}%{Style.RESET_ALL}")
        print(f"Valor sugerido: {Fore.YELLOW}${risco['valor_risco']}{Style.RESET_ALL}")
        print(f"Sinais detectados: {', '.join(sinal['sinais'])}")
        print("-"*50)

    def registrar_sinal(self, ativo: str, direcao: str, momento_entrada: datetime,
                       tempo_expiracao: int, score: float, assertividade: float,
                       padroes: List, indicadores: Dict, ml_prob: float,
                       volatilidade: float):
        """Registra o sinal no banco de dados"""
        self.db.registrar_sinal(
            ativo=ativo,
            direcao=direcao,
            momento_entrada=momento_entrada,
            tempo_expiracao=tempo_expiracao,
            score=score,
            assertividade=assertividade,
            padroes=padroes,
            indicadores=indicadores,
            ml_prob=ml_prob,
            volatilidade=volatilidade
        )

    async def executar(self):
        """Loop principal do sistema"""
        try:
            print(f"\n{Fore.CYAN}Iniciando sequência de inicialização...{Style.RESET_ALL}")

            # Inicializa modelos e dados históricos
            print(f"\n{Fore.YELLOW}Fase 1: Baixando dados históricos...{Style.RESET_ALL}")
            self.baixar_dados_historicos()

            # Aqui você pode adicionar uma barra de progresso para o download
            
            print(f"\n{Fore.YELLOW}Fase 2: Treinando modelo ML...{Style.RESET_ALL}")
            self.ml_predictor.treinar(self.db.get_dados_treino())
            
            print(f"\n{Fore.GREEN}Sistema pronto! Iniciando primeira análise...{Style.RESET_ALL}")

            # Loop principal assíncrono
            while True:
                await self.analisar_mercado()
                await asyncio.sleep(30)  # Espera 30 segundos
            
            print(f"\n{Fore.GREEN}Sistema em execução contínua. Pressione Ctrl+C para encerrar.{Style.RESET_ALL}")
            print(f"Próxima análise em 30 segundos...")
            
            # Loop principal
            while True:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Sistema encerrado pelo usuário")
        except Exception as e:
            self.logger.error(f"Erro crítico: {str(e)}")

if __name__ == "__main__":
    init()  # Inicializa colorama
    
    sistema = TradingSystem()
    
    try:
        asyncio.run(sistema.executar())
    except KeyboardInterrupt:
        print("Sistema encerrado pelo usuário")
    