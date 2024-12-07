import numpy as np
import pandas as pd
import ta
import yfinance as yf
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Padrao:
    nome: str
    forca: float  # 0 a 1
    direcao: str  # 'CALL' ou 'PUT'
    confiabilidade: float  # Histórico de acertos
    tipo: str  # Categoria do padrão (candlestick, tendência, etc)
    tempo_expiracao: int  # Tempo sugerido para expiração em minutos
    confirmacoes: int  # Novo: número de confirmações técnicas

class AnalisePadroesComplexos:
    def __init__(self, config, logger):
        self.config= config
        self.logger = logger
        # Novos parâmetros de validação
        self.tempo_entre_sinais = self.config.get('trading.controles.min_probabilidade')  # 5 minutos entre sinais do mesmo ativo
        self.max_sinais_hora= self.config.get('trading.controles.max_sinais_hora')      # máximo de sinais por hora (todos os ativos)
        self.min_confirmacoes= self.config.get('trading.controles.min_confirmacoes')     # reduzido para facilitar sinais iniciais
        # Registro de sinais recentes
        self.min_confiabilidade = 0.70
        # Configurações de análise
        self.rsi_config = self.config.get('analise.rsi')
        self.bb_config = self.config.get('analise.bandas_bollinger')
        self.macd_config = self.config.get('analise.macd')
                # Parâmetros otimizados para 1min
        self.parametros = {
            'rsi': {
                'sobrecompra': 75,
                'sobrevenda': 25,
                'neutro_min': 45,
                'neutro_max': 55
            },
            'bandas': {
                'desvio': 2.2,
                'compression_min': 0.1,
                'expansion_max': 0.4
            },
            'volume': {
                'min_ratio': 1.2,  # Volume 20% acima da média
                'periodo_media': 8
            }
        }
        self.cache_analises = {}
  
    def _adicionar_padroes_candlestick(self, padroes: List[Padrao], dados: pd.DataFrame, config: Dict):
        """Identifica padrões de candlestick otimizados para timeframe 1min"""
        try:
            self.logger.debug("Analisando padrões de candlestick")
            if len(dados) < 3:
                return

            ultimo_candle = dados.iloc[-1]
            penultimo_candle = dados.iloc[-2]

            # Indicadores para confirmação
            rsi = ta.momentum.RSIIndicator(dados['Close'], window=5).rsi().iloc[-1]
            stoch = ta.momentum.StochasticOscillator(
                dados['High'], 
                dados['Low'], 
                dados['Close'],
                window=5,
                smooth_window=2
            ).stoch().iloc[-1]

            macd = ta.trend.MACD(
                dados['Close'],
                window_fast=5,
                window_slow=13,
                window_sign=3
            )
            macd_line = macd.macd().iloc[-1]
            signal_line = macd.macd_signal().iloc[-1]

            volume_atual = dados['Volume'].iloc[-1] if 'Volume' in dados.columns else 0
            volume_medio = dados['Volume'].rolling(8).mean().iloc[-1] if 'Volume' in dados.columns else 0

            # 1. Doji
            corpo = abs(ultimo_candle['Open'] - ultimo_candle['Close'])
            range_total = ultimo_candle['High'] - ultimo_candle['Low']

            if corpo <= range_total * 0.1:
                confirmacoes = 0
                if (rsi > 45 and rsi < 55): confirmacoes += 1
                if (stoch > 40 and stoch < 60): confirmacoes += 1
                if abs(macd_line - signal_line) < 0.0001: confirmacoes += 1
                if volume_atual > volume_medio: confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"Doji identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="Doji",
                        forca=0.7 * (confirmacoes/self.min_confirmacoes),
                        direcao="NEUTRO",
                        confiabilidade=0.65,
                        tipo="candlestick",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

            # 2. Martelo
            sombra_inferior = min(ultimo_candle['Open'], ultimo_candle['Close']) - ultimo_candle['Low']
            sombra_superior = ultimo_candle['High'] - max(ultimo_candle['Open'], ultimo_candle['Close'])

            if sombra_inferior > corpo * 2 and sombra_superior < corpo * 0.5:
                confirmacoes = 0
                if rsi < 30: confirmacoes += 1
                if stoch < 20: confirmacoes += 1
                if macd_line > signal_line: confirmacoes += 1
                if volume_atual > volume_medio * 1.2: confirmacoes += 1

                if dados['Close'].iloc[-3:].mean() < ultimo_candle['Close']: confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"Martelo identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="Martelo",
                        forca=0.85 * (confirmacoes/self.min_confirmacoes),
                        direcao="CALL",
                        confiabilidade=0.75,
                        tipo="candlestick",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

            # 3. Estrela Cadente
            if sombra_superior > corpo * 2 and sombra_inferior < corpo * 0.5:
                confirmacoes = 0
                if rsi > 70: confirmacoes += 1
                if stoch > 80: confirmacoes += 1
                if macd_line < signal_line: confirmacoes += 1
                if volume_atual > volume_medio * 1.2: confirmacoes += 1

                if dados['Close'].iloc[-3:].mean() > ultimo_candle['Close']: confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"Estrela Cadente identificada com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="Estrela Cadente",
                        forca=0.85 * (confirmacoes/self.min_confirmacoes),
                        direcao="PUT",
                        confiabilidade=0.75,
                        tipo="candlestick",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

            # 4. Engolfo de Alta
            if (penultimo_candle['Close'] < penultimo_candle['Open'] and
                ultimo_candle['Close'] > ultimo_candle['Open'] and
                ultimo_candle['Open'] < penultimo_candle['Close'] and
                ultimo_candle['Close'] > penultimo_candle['Open']):

                confirmacoes = 0
                if rsi > 40 and rsi < 60: confirmacoes += 1
                if macd_line > signal_line: confirmacoes += 1
                if stoch > 30 and stoch < 70: confirmacoes += 1
                if volume_atual > volume_medio * 1.3: confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"Engolfo de Alta identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="Engolfo de Alta",
                        forca=0.9 * (confirmacoes/self.min_confirmacoes),
                        direcao="CALL",
                        confiabilidade=0.8,
                        tipo="candlestick",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

            # 5. Engolfo de Baixa
            if (penultimo_candle['Close'] > penultimo_candle['Open'] and
                ultimo_candle['Close'] < ultimo_candle['Open'] and
                ultimo_candle['Open'] > penultimo_candle['Close'] and
                ultimo_candle['Close'] < penultimo_candle['Open']):

                confirmacoes = 0
                if rsi > 40 and rsi < 60: confirmacoes += 1
                if macd_line < signal_line: confirmacoes += 1
                if stoch > 30 and stoch < 70: confirmacoes += 1
                if volume_atual > volume_medio * 1.3: confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"Engolfo de Baixa identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="Engolfo de Baixa",
                        forca=0.9 * (confirmacoes/self.min_confirmacoes),
                        direcao="PUT",
                        confiabilidade=0.8,
                        tipo="candlestick",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

        except Exception as e:
            self.logger.error(f"Erro em padrões candlestick: {str(e)}")
            #self.logger.error(f"Stack trace: {traceback.format_exc()}")                 
        
    def _adicionar_padroes_tendencia(self, padroes: List[Padrao], dados: pd.DataFrame, config: Dict):
        """Identifica padrões de tendência otimizados para timeframe 1min"""
        try:
            self.logger.debug("Analisando padrões de tendência")
            close = dados['Close']

            # Médias ajustadas para 1min
            ema3 = ta.trend.EMAIndicator(close, window=3).ema_indicator()
            ema8 = ta.trend.EMAIndicator(close, window=8).ema_indicator()
            ema13 = ta.trend.EMAIndicator(close, window=13).ema_indicator()

            # Indicadores para confirmação
            rsi = ta.momentum.RSIIndicator(close, window=5).rsi()
            stoch = ta.momentum.StochasticOscillator(
                dados['High'],
                dados['Low'],
                dados['Close'],
                window=5,
                smooth_window=2
            )

            macd = ta.trend.MACD(
                close,
                window_fast=5,
                window_slow=13,
                window_sign=3
            )
            macd_line = macd.macd()
            signal_line = macd.macd_signal()
            
            # Volume médio
            volume_atual = dados['Volume'].iloc[-1] if 'Volume' in dados.columns else 0
            volume_medio = dados['Volume'].rolling(8).mean().iloc[-1] if 'Volume' in dados.columns else 0
            
            # 1. Cruzamento EMA Alta
            if (ema3.iloc[-1] > ema8.iloc[-1] and ema3.iloc[-2] <= ema8.iloc[-2]):
                confirmacoes = 0
                if rsi.iloc[-1] > 40: confirmacoes += 1
                if stoch.stoch().iloc[-1] > 30: confirmacoes += 1
                if macd_line.iloc[-1] > signal_line.iloc[-1]: confirmacoes += 1
                if close.iloc[-1] > ema13.iloc[-1]: confirmacoes += 1

                # Volume crescente
                if 'Volume' in dados.columns:
                    volume_atual = dados['Volume'].iloc[-1]
                    volume_medio = dados['Volume'].rolling(8).mean().iloc[-1]
                    if volume_atual > volume_medio * 1.2:
                        confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"Cruzamento alta identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="Cruzamento EMA Alta",
                        forca=0.85 * (confirmacoes/self.min_confirmacoes),
                        direcao="CALL",
                        confiabilidade=0.8,
                        tipo="tendencia",
                        tempo_expiracao=3,
                        confirmacoes=confirmacoes
                    ))

            # 2. Cruzamento EMA Baixa
            if (ema3.iloc[-1] < ema8.iloc[-1] and ema3.iloc[-2] >= ema8.iloc[-2]):
                confirmacoes = 0
                if rsi.iloc[-1] < 60: confirmacoes += 1
                if stoch.stoch().iloc[-1] < 70: confirmacoes += 1
                if macd_line.iloc[-1] < signal_line.iloc[-1]: confirmacoes += 1
                if close.iloc[-1] < ema13.iloc[-1]: confirmacoes += 1

                if 'Volume' in dados.columns:
                    volume_atual = dados['Volume'].iloc[-1]
                    volume_medio = dados['Volume'].rolling(8).mean().iloc[-1]
                    if volume_atual > volume_medio * 1.2:
                        confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"Cruzamento baixa identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="Cruzamento EMA Baixa",
                        forca=0.85 * (confirmacoes/self.min_confirmacoes),
                        direcao="PUT",
                        confiabilidade=0.8,
                        tipo="tendencia",
                        tempo_expiracao=3,
                        confirmacoes=confirmacoes
                    ))

            # 3. Momentum Forte Alta
            if (close.pct_change(3).iloc[-1] > 0.001 and
                macd_line.iloc[-1] > signal_line.iloc[-1] * 1.2):

                confirmacoes = 0
                if rsi.iloc[-1] > 50 and rsi.iloc[-1] < 80: confirmacoes += 1
                if stoch.stoch().iloc[-1] > 40: confirmacoes += 1
                if ema3.iloc[-1] > ema8.iloc[-1]: confirmacoes += 1
                if all(close.iloc[-3:] > ema8.iloc[-3:]): confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"Momentum alta identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="Momentum Forte Alta",
                        forca=0.9 * (confirmacoes/self.min_confirmacoes),
                        direcao="CALL",
                        confiabilidade=0.85,
                        tipo="tendencia",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

            # 4. Momentum Forte Baixa
            if (close.pct_change(3).iloc[-1] < -0.001 and
                macd_line.iloc[-1] < signal_line.iloc[-1] * 0.8):

                confirmacoes = 0
                if rsi.iloc[-1] < 50 and rsi.iloc[-1] > 20: confirmacoes += 1
                if stoch.stoch().iloc[-1] < 60: confirmacoes += 1
                if ema3.iloc[-1] < ema8.iloc[-1]: confirmacoes += 1
                if all(close.iloc[-3:] < ema8.iloc[-3:]): confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"Momentum baixa identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="Momentum Forte Baixa",
                        forca=0.9 * (confirmacoes/self.min_confirmacoes),
                        direcao="PUT",
                        confiabilidade=0.85,
                        tipo="tendencia",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

            # 5. Reversão de Tendência Alta
            if (close.iloc[-1] < ema8.iloc[-1] and
                rsi.iloc[-1] < 30 and
                stoch.stoch().iloc[-1] < 20):

                confirmacoes = 0
                if macd_line.iloc[-1] > macd_line.iloc[-2]: confirmacoes += 1
                if volume_atual > volume_medio * 1.5: confirmacoes += 1
                if close.iloc[-1] > close.iloc[-2]: confirmacoes += 1
                if stoch.stoch().iloc[-1] > stoch.stoch().iloc[-2]: confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"Reversão alta identificada com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="Reversão Alta",
                        forca=0.8 * (confirmacoes/self.min_confirmacoes),
                        direcao="CALL",
                        confiabilidade=0.75,
                        tipo="tendencia",
                        tempo_expiracao=3,
                        confirmacoes=confirmacoes
                    ))

            # 6. Reversão de Tendência Baixa
            if (close.iloc[-1] > ema8.iloc[-1] and
                rsi.iloc[-1] > 70 and
                stoch.stoch().iloc[-1] > 80):

                confirmacoes = 0
                if macd_line.iloc[-1] < macd_line.iloc[-2]: confirmacoes += 1
                if volume_atual > volume_medio * 1.5: confirmacoes += 1
                if close.iloc[-1] < close.iloc[-2]: confirmacoes += 1
                if stoch.stoch().iloc[-1] < stoch.stoch().iloc[-2]: confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"Reversão baixa identificada com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="Reversão Baixa",
                        forca=0.8 * (confirmacoes/self.min_confirmacoes),
                        direcao="PUT",
                        confiabilidade=0.75,
                        tipo="tendencia",
                        tempo_expiracao=3,
                        confirmacoes=confirmacoes
                    ))

        except Exception as e:
            self.logger.error(f"Erro em padrões tendência: {str(e)}")
        
    def _adicionar_padroes_momentum(self, padroes: List[Padrao], dados: pd.DataFrame, config: Dict):
        """Identifica padrões de momentum otimizados para 1min"""
        try:
            self.logger.debug("Analisando padrões de momentum")

            # RSI rápido
            rsi = ta.momentum.RSIIndicator(dados['Close'], window=5).rsi()

            # Estocástico rápido
            stoch = ta.momentum.StochasticOscillator(
                dados['High'],
                dados['Low'],
                dados['Close'],
                window=5,
                smooth_window=2
            )
            k = stoch.stoch()
            d = stoch.stoch_signal()

            # MACD rápido
            macd = ta.trend.MACD(
                dados['Close'],
                window_fast=5,
                window_slow=13,
                window_sign=3
            )
            macd_line = macd.macd()
            signal_line = macd.macd_signal()

            # Volume médio
            volume_atual = dados['Volume'].iloc[-1] if 'Volume' in dados.columns else 0
            volume_medio = dados['Volume'].rolling(8).mean().iloc[-1] if 'Volume' in dados.columns else 0

            # 1. RSI Sobrevendido
            if rsi.iloc[-1] < 25:
                confirmacoes = 0
                if rsi.iloc[-2] < 25: confirmacoes += 1
                if k.iloc[-1] < 20: confirmacoes += 1
                if macd_line.iloc[-1] > macd_line.iloc[-2]: confirmacoes += 1
                if volume_atual > volume_medio * 1.2: confirmacoes += 1

                if dados['Close'].iloc[-1] > dados['Close'].iloc[-2]: confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"RSI sobrevendido identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="RSI Sobrevendido",
                        forca=0.9 * (confirmacoes/self.min_confirmacoes),
                        direcao="CALL",
                        confiabilidade=0.85,
                        tipo="momentum",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

            # 2. RSI Sobrecomprado
            if rsi.iloc[-1] > 75:
                confirmacoes = 0
                if rsi.iloc[-2] > 75: confirmacoes += 1
                if k.iloc[-1] > 80: confirmacoes += 1
                if macd_line.iloc[-1] < macd_line.iloc[-2]: confirmacoes += 1
                if volume_atual > volume_medio * 1.2: confirmacoes += 1

                if dados['Close'].iloc[-1] < dados['Close'].iloc[-2]: confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"RSI sobrecomprado identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="RSI Sobrecomprado",
                        forca=0.9 * (confirmacoes/self.min_confirmacoes),
                        direcao="PUT",
                        confiabilidade=0.85,
                        tipo="momentum",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

            # 3. Estocástico Sobrevendido com Divergência
            if k.iloc[-1] < 20 and d.iloc[-1] < 20:
                confirmacoes = 0
                if k.iloc[-1] > k.iloc[-2]: confirmacoes += 1  # %K virando pra cima
                if k.iloc[-1] > d.iloc[-1]: confirmacoes += 1  # Cruzamento %K > %D
                if rsi.iloc[-1] < 40: confirmacoes += 1
                if macd_line.iloc[-1] > macd_line.iloc[-2]: confirmacoes += 1
                if volume_atual > volume_medio: confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"Estocástico sobrevendido identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="Estocástico Sobrevendido",
                        forca=0.85 * (confirmacoes/self.min_confirmacoes),
                        direcao="CALL",
                        confiabilidade=0.8,
                        tipo="momentum",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

            # 4. Estocástico Sobrecomprado com Divergência
            if k.iloc[-1] > 80 and d.iloc[-1] > 80:
                confirmacoes = 0
                if k.iloc[-1] < k.iloc[-2]: confirmacoes += 1  # %K virando pra baixo
                if k.iloc[-1] < d.iloc[-1]: confirmacoes += 1  # Cruzamento %K < %D
                if rsi.iloc[-1] > 60: confirmacoes += 1
                if macd_line.iloc[-1] < macd_line.iloc[-2]: confirmacoes += 1
                if volume_atual > volume_medio: confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"Estocástico sobrecomprado identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="Estocástico Sobrecomprado",
                        forca=0.85 * (confirmacoes/self.min_confirmacoes),
                        direcao="PUT",
                        confiabilidade=0.8,
                        tipo="momentum",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

            # 5. Divergência MACD Alta
            if (macd_line.iloc[-1] > signal_line.iloc[-1] and 
                macd_line.iloc[-2] <= signal_line.iloc[-2]):

                confirmacoes = 0
                if macd_line.iloc[-1] > macd_line.iloc[-2]: confirmacoes += 1
                if rsi.iloc[-1] > 40 and rsi.iloc[-1] < 70: confirmacoes += 1
                if k.iloc[-1] > 20 and k.iloc[-1] < 80: confirmacoes += 1
                if volume_atual > volume_medio * 1.2: confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"MACD bullish identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="MACD Bullish",
                        forca=0.8 * (confirmacoes/self.min_confirmacoes),
                        direcao="CALL",
                        confiabilidade=0.75,
                        tipo="momentum",
                        tempo_expiracao=3,
                        confirmacoes=confirmacoes
                    ))

            # 6. Divergência MACD Baixa
            if (macd_line.iloc[-1] < signal_line.iloc[-1] and 
                macd_line.iloc[-2] >= signal_line.iloc[-2]):

                confirmacoes = 0
                if macd_line.iloc[-1] < macd_line.iloc[-2]: confirmacoes += 1
                if rsi.iloc[-1] < 60 and rsi.iloc[-1] > 30: confirmacoes += 1
                if k.iloc[-1] < 80 and k.iloc[-1] > 20: confirmacoes += 1
                if volume_atual > volume_medio * 1.2: confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"MACD bearish identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="MACD Bearish",
                        forca=0.8 * (confirmacoes/self.min_confirmacoes),
                        direcao="PUT",
                        confiabilidade=0.75,
                        tipo="momentum",
                        tempo_expiracao=3,
                        confirmacoes=confirmacoes
                    ))

        except Exception as e:
            self.logger.error(f"Erro em padrões momentum: {str(e)}")
            #self.logger.error(f"Stack trace: {traceback.format_exc()}")

    def _adicionar_padroes_volatilidade(self, padroes: List[Padrao], dados: pd.DataFrame, config: Dict):
        """Identifica padrões de volatilidade otimizados para 1min"""
        try:
            self.logger.debug("Analisando padrões de volatilidade")

            # Bollinger Bands rápidas
            bb = ta.volatility.BollingerBands(
                dados['Close'],
                window=8,  # Reduzido para 1min
                window_dev=2.2
            )
            bb_superior = bb.bollinger_hband()
            bb_inferior = bb.bollinger_lband()
            bb_media = bb.bollinger_mavg()
            bb_width = bb.bollinger_wband()

            # ATR rápido
            atr = ta.volatility.AverageTrueRange(
                dados['High'],
                dados['Low'],
                dados['Close'],
                window=5  # Reduzido para 1min
            ).average_true_range()

            # Indicadores auxiliares
            rsi = ta.momentum.RSIIndicator(dados['Close'], window=5).rsi()
            volume_atual = dados['Volume'].iloc[-1] if 'Volume' in dados.columns else 0
            volume_medio = dados['Volume'].rolling(8).mean().iloc[-1] if 'Volume' in dados.columns else 0

            # 1. Squeeze e Expansão
            bb_range = (bb_superior - bb_inferior) / bb_media
            bb_range_anterior = bb_range.shift(1)

            # Detecta squeeze (compressão de volatilidade)
            if bb_range.iloc[-1] < bb_range.iloc[-3:].mean() * 0.8:
                confirmacoes = 0
                if atr.iloc[-1] < atr.iloc[-3:].mean(): confirmacoes += 1
                if volume_atual > volume_medio * 1.2: confirmacoes += 1
                if abs(dados['Close'].iloc[-1] - bb_media.iloc[-1]) < atr.iloc[-1]: confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"Squeeze identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="Volatility Squeeze",
                        forca=0.8 * (confirmacoes/self.min_confirmacoes),
                        direcao="NEUTRO",
                        confiabilidade=0.7,
                        tipo="volatilidade",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

            # 2. Rompimento Inferior
            if dados['Close'].iloc[-1] < bb_inferior.iloc[-1]:
                confirmacoes = 0
                if rsi.iloc[-1] < 30: confirmacoes += 1
                if volume_atual > volume_medio * 1.3: confirmacoes += 1
                if dados['Close'].iloc[-2] > bb_inferior.iloc[-2]: confirmacoes += 1
                if atr.iloc[-1] > atr.iloc[-3:].mean(): confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"Rompimento inferior identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="BB Oversold",
                        forca=0.85 * (confirmacoes/self.min_confirmacoes),
                        direcao="CALL",
                        confiabilidade=0.8,
                        tipo="volatilidade",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

            # 3. Rompimento Superior
            if dados['Close'].iloc[-1] > bb_superior.iloc[-1]:
                confirmacoes = 0
                if rsi.iloc[-1] > 70: confirmacoes += 1
                if volume_atual > volume_medio * 1.3: confirmacoes += 1
                if dados['Close'].iloc[-2] < bb_superior.iloc[-2]: confirmacoes += 1
                if atr.iloc[-1] > atr.iloc[-3:].mean(): confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    self.logger.info(f"Rompimento superior identificado com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="BB Overbought",
                        forca=0.85 * (confirmacoes/self.min_confirmacoes),
                        direcao="PUT",
                        confiabilidade=0.8,
                        tipo="volatilidade",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

            # 4. Volatilidade Alta
            if atr.iloc[-1] > atr.iloc[-5:].mean() * 1.5:
                confirmacoes = 0
                if bb_width.iloc[-1] > bb_width.iloc[-5:].mean() * 1.3: confirmacoes += 1
                if volume_atual > volume_medio * 1.5: confirmacoes += 1

                tendencia = 'CALL' if dados['Close'].iloc[-3:].mean() > dados['Close'].iloc[-6:-3].mean() else 'PUT'

                if confirmacoes >= 2:
                    self.logger.info(f"Alta volatilidade identificada com {confirmacoes} confirmações")
                    padroes.append(Padrao(
                        nome="Alta Volatilidade",
                        forca=0.7 * (confirmacoes/self.min_confirmacoes),
                        direcao=tendencia,
                        confiabilidade=0.65,
                        tipo="volatilidade",
                        tempo_expiracao=1,  # Reduzido devido alta volatilidade
                        confirmacoes=confirmacoes
                    ))

        except Exception as e:
            self.logger.error(f"Erro em padrões volatilidade: {str(e)}")
            #self.logger.error(f"Stack trace: {traceback.format_exc()}")

    def _consolidar_padroes(self, padroes: List[Padrao]) -> List[Padrao]:
        """Consolida e filtra padrões identificados otimizado para 1min"""
        try:
            self.logger.debug("Iniciando consolidação de padrões")
            if not padroes:
                return []

            # Agrupa por direção
            calls = [p for p in padroes if p.direcao == "CALL"]
            puts = [p for p in padroes if p.direcao == "PUT"]

            # Calcula forças médias
            forca_calls = sum(p.forca * p.confiabilidade for p in calls) / len(calls) if calls else 0
            forca_puts = sum(p.forca * p.confiabilidade for p in puts) / len(puts) if puts else 0

            # Filtros de consistência
            padroes_filtrados = []
            for padrao in padroes:
                # Aumenta força baseado em confirmações
                if padrao.direcao == "CALL":
                    padrao.forca *= (1 + 0.15 * len(calls))  # +15% por confirmação
                    if forca_puts > 0.3:  # Se há força oposta significativa
                        padrao.forca *= 0.8  # Reduz força em 20%
                elif padrao.direcao == "PUT":
                    padrao.forca *= (1 + 0.15 * len(puts))
                    if forca_calls > 0.3:
                        padrao.forca *= 0.8

                # Ajusta tempo de expiração baseado na força
                if padrao.forca > 0.85:
                    padrao.tempo_expiracao = min(padrao.tempo_expiracao, 2)
                elif padrao.forca < 0.7:
                    padrao.tempo_expiracao = min(padrao.tempo_expiracao + 1, 5)

                # Filtra padrões fracos
                if padrao.forca >= 0.6 and padrao.confirmacoes >= self.min_confirmacoes:
                    if padrao.tipo == "volatilidade" and padrao.forca < 0.75:
                        continue  # Mais rigoroso com padrões de volatilidade
                    padroes_filtrados.append(padrao)

            # Ordena por força e confiabilidade
            padroes_filtrados.sort(
                key=lambda x: (x.forca * x.confiabilidade, x.confirmacoes),
                reverse=True
            )

            # Limita número de padrões por direção
            calls_filtrados = [p for p in padroes_filtrados if p.direcao == "CALL"][:3]
            puts_filtrados = [p for p in padroes_filtrados if p.direcao == "PUT"][:3]
            neutros_filtrados = [p for p in padroes_filtrados if p.direcao == "NEUTRO"][:1]

            resultado_final = calls_filtrados + puts_filtrados + neutros_filtrados

            self.logger.info(f"Padrões consolidados: {len(resultado_final)} de {len(padroes)} originais")
            for p in resultado_final:
                self.logger.debug(f"Padrão: {p.nome}, Força: {p.forca:.2f}, Confirmações: {p.confirmacoes}")

            return resultado_final

        except Exception as e:
            self.logger.error(f"Erro na consolidação de padrões: {str(e)}")
            #self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return []

    async def analisar(self, dados: pd.DataFrame, ativo: str) -> Dict:
        """Análise técnica principal otimizada para 1min"""
        try:
            self.logger.debug(f"Iniciando análise para {ativo}")

            # Adiciona validação de dados
            if not self._validar_dados(dados):
                self.logger.warning("Dados inválidos para análise")
                return None


            # Adiciona indicadores técnicos
            indicadores = self._adicionar_indicadores_tecnico(dados)

            if not dados.empty:
                # Identifica padrões
                padroes = []
                self._adicionar_padroes_candlestick(padroes, dados, {})
                self._adicionar_padroes_tendencia(padroes, dados, {})
                self._adicionar_padroes_momentum(padroes, dados, {})
                self._adicionar_padroes_volatilidade(padroes, dados, {})
                # Novas análises
                self._adicionar_padroes_tres_velas(padroes, dados)
                self._adicionar_analise_volume(padroes, dados)
                self._adicionar_analise_niveis(padroes, dados)
                self._adicionar_price_action(padroes, dados)

                # Consolida e filtra
                padroes_consolidados = self._consolidar_padroes(padroes)
                if not padroes_consolidados:
                    self.logger.info("Nenhum padrão significativo identificado")
                    return None

                # Determina direção predominante e força
                        # Melhor análise de direção
                direcao = self._determinar_direcao_avancada(
                    padroes_consolidados, 
                    indicadores,
                    dados
                )
                forca_sinal = self._calcular_forca(padroes_consolidados)

                # Calcula volatilidade normalizada
                volatilidade = dados['Close'].pct_change().rolling(8).std() * np.sqrt(252)
                volatilidade_norm = (volatilidade - volatilidade.rolling(21).mean()) / volatilidade.rolling(21).std()

                if abs(float(volatilidade_norm.iloc[-1])) > 2:
                    self.logger.warning("Volatilidade muito alta, sinal descartado")
                    return None

                resultado = {
                    'ativo': ativo,
                    'timestamp': dados.index[-1],
                    'direcao': direcao,
                    'forca_sinal': forca_sinal,
                    'padroes': [self._formatar_padrao(p) for p in padroes_consolidados],
                    'num_padroes': len(padroes_consolidados),
                    'volatilidade': float(volatilidade.iloc[-1]),
                    'volatilidade_norm': float(volatilidade_norm.iloc[-1]),
                    'price_action': self._analisar_price_action(dados),
                    'indicadores': indicadores
                }

                self.logger.info(f"Análise concluída para {ativo}: {resultado['direcao']} (força: {resultado['forca_sinal']:.2f})")
                return resultado

            return None

        except Exception as e:
            self.logger.error(f"Erro na análise: {str(e)}")
            #self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return None

    def _determinar_direcao(self, padroes: List[Padrao]) -> str:
        """Determina direção predominante com peso por tipo de padrão"""
        try:
            if not padroes:
                return "NEUTRO"

            # Pesos por tipo de padrão
            pesos = {
                'tendencia': 1.2,
                'momentum': 1.0,
                'candlestick': 0.8,
                'volatilidade': 0.7
            }

            # Calcula força ponderada por direção
            calls = sum(
                p.forca * p.confiabilidade * pesos.get(p.tipo, 1.0)
                for p in padroes if p.direcao == "CALL"
            )

            puts = sum(
                p.forca * p.confiabilidade * pesos.get(p.tipo, 1.0)
                for p in padroes if p.direcao == "PUT"
            )

            # Aplica threshold mínimo de diferença
            diferenca = abs(calls - puts)
            if diferenca < 0.3:  # Requer diferença mínima de 30%
                return "NEUTRO"

            return "CALL" if calls > puts else "PUT"

        except Exception as e:
            self.logger.error(f"Erro ao determinar direção: {str(e)}")
            return "NEUTRO"

    def _calcular_forca(self, padroes: List[Padrao]) -> float:
        """Calcula força do sinal com pesos dinâmicos"""
        try:
            if not padroes:
                return 0.0

            # Pesos base por número de confirmações
            pesos_confirmacao = {
                4: 1.0,   # Mínimo
                5: 1.1,   # Bom
                6: 1.2,   # Muito bom
                7: 1.3    # Excelente
            }

            forca_total = 0
            peso_total = 0

            for padrao in padroes:
                # Peso base pelo tipo
                peso_tipo = {
                    'tendencia': 1.2,
                    'momentum': 1.0,
                    'candlestick': 0.8,
                    'volatilidade': 0.7
                }.get(padrao.tipo, 1.0)

                # Ajusta peso pelas confirmações
                peso_conf = pesos_confirmacao.get(padrao.confirmacoes, 1.0)

                # Peso final
                peso_final = peso_tipo * peso_conf

                forca_total += padrao.forca * padrao.confiabilidade * peso_final
                peso_total += peso_final

            forca_media = forca_total / peso_total if peso_total > 0 else 0

            # Limita entre 0 e 1
            return min(1.0, max(0.0, forca_media))

        except Exception as e:
            self.logger.error(f"Erro ao calcular força: {str(e)}")
            return 0.0
  
    async def _identificar_padroes_completos(self, dados: pd.DataFrame, ativo: str) -> List[Padrao]:
        """Identificação otimizada de padrões"""
        try:
            padroes = []
            
            # Indicadores base
            indicadores = await self._calcular_indicadores_base(dados)
            
            # Análise de padrões
            self._adicionar_padroes_candlestick(padroes, dados, indicadores)
            self._adicionar_padroes_tendencia(padroes, dados, indicadores)
            self._adicionar_padroes_momentum(padroes, dados, indicadores)
            self._adicionar_padroes_reversao(padroes, dados, indicadores)  # Nova linha

            # Filtragem e consolidação
            padroes = [p for p in padroes if p.confirmacoes >= self.min_confirmacoes]
            return self._consolidar_padroes(padroes)

        except Exception as e:
            self.logger.error(f"Erro ao identificar padrões: {str(e)}")
            return []

    def _adicionar_padroes_reversao(self, padroes: List[Padrao], dados: pd.DataFrame, config: Dict):
        """Identifica padrões de reversão rápida com confirmações técnicas"""
        try:
            if len(dados) < 5:  # Mínimo necessário para análise
                return

            # Últimos candles para análise
            ultimos_candles = dados.tail(5)

            # Calcula indicadores para confirmação
            rsi = ta.momentum.RSIIndicator(dados['Close']).rsi().iloc[-1]
            stoch = ta.momentum.StochasticOscillator(
                dados['High'], dados['Low'], dados['Close']
            ).stoch().iloc[-1]

            bb = ta.volatility.BollingerBands(dados['Close'])
            bb_inferior = bb.bollinger_lband().iloc[-1]
            bb_superior = bb.bollinger_hband().iloc[-1]

            volume_medio = dados['Volume'].rolling(5).mean().iloc[-1]
            volume_atual = dados['Volume'].iloc[-1]

            # Detecta reversão de alta
            if (ultimos_candles['Close'].pct_change().iloc[-1] > 0.001 and
                all(ultimos_candles['Close'].pct_change().iloc[-3:-1] < -0.0005)):

                confirmacoes = 0

                # Confirmações técnicas para alta
                if volume_atual > volume_medio * 1.2:  # Volume 20% acima da média
                    confirmacoes += 1

                if dados['Close'].iloc[-1] > dados['Open'].iloc[-1]:  # Candle de alta
                    confirmacoes += 1

                if rsi < 40:  # Sobrevendido
                    confirmacoes += 1

                if stoch < 30:  # Sobrevendido
                    confirmacoes += 1

                if dados['Close'].iloc[-1] < bb_inferior:  # Preço abaixo da banda inferior
                    confirmacoes += 1

                # Verifica força da reversão
                variacao_preco = abs(dados['Close'].iloc[-1] - dados['Low'].iloc[-1])
                if variacao_preco > dados['Close'].iloc[-1] * 0.002:  # Movimento significativo
                    confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    padroes.append(Padrao(
                        nome="Reversão de Alta",
                        forca=0.85,
                        direcao="CALL",
                        confiabilidade=0.8,
                        tipo="reversao",
                        tempo_expiracao=5,  # Tempo curto para reversões
                        confirmacoes=confirmacoes
                    ))

            # Detecta reversão de baixa
            if (ultimos_candles['Close'].pct_change().iloc[-1] < -0.001 and
                all(ultimos_candles['Close'].pct_change().iloc[-3:-1] > 0.0005)):

                confirmacoes = 0

                # Confirmações técnicas para baixa
                if volume_atual > volume_medio * 1.2:
                    confirmacoes += 1

                if dados['Close'].iloc[-1] < dados['Open'].iloc[-1]:  # Candle de baixa
                    confirmacoes += 1

                if rsi > 60:  # Sobrecomprado
                    confirmacoes += 1

                if stoch > 70:  # Sobrecomprado
                    confirmacoes += 1

                if dados['Close'].iloc[-1] > bb_superior:  # Preço acima da banda superior
                    confirmacoes += 1

                # Verifica força da reversão
                variacao_preco = abs(dados['Close'].iloc[-1] - dados['High'].iloc[-1])
                if variacao_preco > dados['Close'].iloc[-1] * 0.002:
                    confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    padroes.append(Padrao(
                        nome="Reversão de Baixa",
                        forca=0.85,
                        direcao="PUT",
                        confiabilidade=0.8,
                        tipo="reversao",
                        tempo_expiracao=5,
                        confirmacoes=confirmacoes
                    ))

        except Exception as e:
            self.logger.error(f"Erro em padrões reversão: {str(e)}")

    def _formatar_padrao(self, padrao: Padrao) -> Dict:
        """Formata detalhes do padrão para análise"""
        try:
            self.logger.debug(f"Formatando padrão: {padrao.nome}")
            
            return {
                "nome": padrao.nome,
                "forca": round(padrao.forca, 3),
                "direcao": padrao.direcao,
                "confiabilidade": round(padrao.confiabilidade, 3),
                "tipo": padrao.tipo,
                "tempo_expiracao": min(padrao.tempo_expiracao, 5),  # Limita em 5min
                "confirmacoes": padrao.confirmacoes
            }
        except Exception as e:
            self.logger.error(f"Erro ao formatar padrão: {str(e)}")
            return {}
    
    def _validar_dados(self, dados: pd.DataFrame) -> bool:
        """Valida dados de entrada"""
        try:
            self.logger.debug("Validando dados de entrada")

            if dados is None or dados.empty:
                self.logger.warning("Dados vazios")
                return False

            # Obtém as colunas atuais e converte para lista
            colunas_atuais = list(dados.columns)
            self.logger.debug(f"Colunas disponíveis: {colunas_atuais}")

            # Verifica colunas (case insensitive)
            colunas_requeridas = ['open', 'high', 'low', 'close']
            colunas_presentes = [col for col in colunas_atuais 
                                if col.lower() in colunas_requeridas]

            if len(colunas_presentes) < len(colunas_requeridas):
                self.logger.warning(f"Colunas ausentes. Requeridas: {colunas_requeridas}")
                self.logger.warning(f"Presentes: {colunas_presentes}")
                return False

            if len(dados) < 30:
                self.logger.warning(f"Dados insuficientes: {len(dados)} registros")
                return False

            if dados[colunas_presentes].isnull().any().any():
                self.logger.warning("Dados contêm valores nulos")
                return False

            if not isinstance(dados.index, pd.DatetimeIndex):
                self.logger.warning("Índice não é timestamp")
                return False

            self.logger.info("Dados validados com sucesso")
            return True

        except Exception as e:
            self.logger.error(f"Erro na validação: {str(e)}")
            #self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False
    
    def _calcular_volatilidade(self, dados: pd.DataFrame) -> float:
        """Calcula volatilidade normalizada"""
        try:
            if len(dados) < 8:
                return 0
                
            retornos = dados['close'].pct_change()
            volatilidade = retornos.rolling(8).std() * np.sqrt(252)
            volatilidade_norm = (
                volatilidade - volatilidade.rolling(21).mean()
            ) / volatilidade.rolling(21).std()
            
            return float(volatilidade_norm.iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular volatilidade: {str(e)}")
            return 0



    async def _calcular_indicadores_base(self, dados: pd.DataFrame) -> Dict:
        """Calcula indicadores técnicos base"""
        return {
            'rsi': ta.momentum.RSIIndicator(dados['Close'], 
                                          window=self.rsi_config['periodo']).rsi(),
            'bb': ta.volatility.BollingerBands(dados['Close'], 
                                             window=self.bb_config['periodo']),
            'macd': ta.trend.MACD(dados['Close'],
                                window_fast=self.macd_config['rapida'],
                                window_slow=self.macd_config['lenta'],
                                window_sign=self.macd_config['sinal'])
        }
        
    def _adicionar_analise_niveis(self, padroes: List[Padrao], dados: pd.DataFrame):
        """Análise de suporte e resistência"""
        try:
            
            # No início da função, após o try:
            if dados is None or dados.empty or len(dados) < 2:
                self.logger.warning("Dados insuficientes para análise de níveis")
                return

            if not all(col in dados.columns for col in ['High', 'Low', 'Close']):
                self.logger.warning("Colunas necessárias ausentes para análise de níveis")
                return
            
            # Calcula todos os indicadores necessários no início
            rsi = ta.momentum.RSIIndicator(dados['Close'], window=14).rsi()
            ema8 = ta.trend.EMAIndicator(dados['Close'], window=8).ema_indicator()
            volume = dados['Volume'] if 'Volume' in dados.columns else None
            volume_media = volume.rolling(5).mean() if volume is not None else None

            # Preço atual
            price = dados['Close'].iloc[-1]

            # Calcula níveis de Fibonacci
            high = dados['High'].max()
            low = dados['Low'].min()
            diff = high - low

            fib_levels = {
                0.236: low + 0.236 * diff,
                0.382: low + 0.382 * diff,
                0.618: low + 0.618 * diff
            }

            # Verifica proximidade com níveis Fibonacci
            for level, value in fib_levels.items():
                if abs(price - value) / price < 0.0005:  # 0.05% de proximidade
                    direcao = "CALL" if price > value else "PUT"
                    confirmacoes = 0

                    # RSI
                    if direcao == "CALL" and rsi.iloc[-1] < 30:
                        confirmacoes += 1
                    if direcao == "PUT" and rsi.iloc[-1] > 70:
                        confirmacoes += 1

                    # Volume
                    if volume is not None and volume.iloc[-1] > volume_media.iloc[-1]:
                        confirmacoes += 1

                    # Tendência
                    if direcao == "CALL" and price > ema8.iloc[-1]:
                        confirmacoes += 1
                    if direcao == "PUT" and price < ema8.iloc[-1]:
                        confirmacoes += 1

                    if confirmacoes >= self.min_confirmacoes:
                        padroes.append(Padrao(
                            nome=f"Nível Fibonacci {level}",
                            forca=0.85,
                            direcao=direcao,
                            confiabilidade=0.8,
                            tipo="niveis",
                            tempo_expiracao=2,
                            confirmacoes=confirmacoes
                        ))

            # Calcula Pivot Points
            pivot = (dados['High'].iloc[-1] + dados['Low'].iloc[-1] + dados['Close'].iloc[-1]) / 3
            r1 = 2 * pivot - dados['Low'].iloc[-1]
            s1 = 2 * pivot - dados['High'].iloc[-1]

            # Verifica proximidade com pivot points
            for level, value in [('R1', r1), ('S1', s1)]:
                if abs(price - value) / price < 0.0005:
                    direcao = "PUT" if level.startswith('R') else "CALL"
                    confirmacoes = 0

                    # RSI
                    if direcao == "CALL" and rsi.iloc[-1] < 30:
                        confirmacoes += 1
                    if direcao == "PUT" and rsi.iloc[-1] > 70:
                        confirmacoes += 1

                    # Volume
                    if volume is not None and volume.iloc[-1] > volume_media.iloc[-1]:
                        confirmacoes += 1

                    # Tendência
                    if direcao == "CALL" and price > ema8.iloc[-1]:
                        confirmacoes += 1
                    if direcao == "PUT" and price < ema8.iloc[-1]:
                        confirmacoes += 1

                    if confirmacoes >= self.min_confirmacoes:
                        padroes.append(Padrao(
                            nome=f"Pivot {level}",
                            forca=0.8,
                            direcao=direcao,
                            confiabilidade=0.75,
                            tipo="niveis",
                            tempo_expiracao=2,
                            confirmacoes=confirmacoes
                        ))

        except Exception as e:
            self.logger.error(f"Erro na análise de níveis: {str(e)}")
            import traceback
            self.logger.error(f"Stack trace: {traceback.format_exc()}")


    def _adicionar_padroes_tres_velas(self, padroes: List[Padrao], dados: pd.DataFrame):
        """Identifica padrões de três velas"""
        try:
            if len(dados) < 3:
                return
    
            # Calcula indicadores necessários
            rsi = ta.momentum.RSIIndicator(dados['Close'], window=14).rsi()
            volume = dados['Volume'] if 'Volume' in dados.columns else None
            volume_media = volume.rolling(5).mean() if volume is not None else None
    
            ultimas_velas = dados.tail(3)
            
            # Three Line Strike
            if (ultimas_velas['Close'].iloc[0] < ultimas_velas['Open'].iloc[0] and
                ultimas_velas['Close'].iloc[1] < ultimas_velas['Open'].iloc[1] and
                ultimas_velas['Close'].iloc[2] > ultimas_velas['Open'].iloc[2]):
                
                confirmacoes = 0
                if volume is not None and volume.iloc[-1] > volume_media.iloc[-1]:
                    confirmacoes += 1
                if dados['Close'].iloc[-1] > dados['Close'].rolling(8).mean().iloc[-1]:
                    confirmacoes += 1
                if rsi.iloc[-1] < 30:  # Condição de sobrevenda
                    confirmacoes += 1
                    
                if confirmacoes >= self.min_confirmacoes:
                    padroes.append(Padrao(
                        nome="Three Line Strike",
                        forca=0.85,
                        direcao="CALL",
                        confiabilidade=0.8,
                        tipo="candlestick",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))
    
            # Morning Star
            if (ultimas_velas['Close'].iloc[0] < ultimas_velas['Open'].iloc[0] and
                abs(ultimas_velas['Close'].iloc[1] - ultimas_velas['Open'].iloc[1]) < 
                abs(ultimas_velas['High'].iloc[1] - ultimas_velas['Low'].iloc[1]) * 0.3 and
                ultimas_velas['Close'].iloc[2] > ultimas_velas['Open'].iloc[2]):
                
                confirmacoes = 0
                if rsi.iloc[-1] < 30:
                    confirmacoes += 1
                if volume is not None and volume.iloc[-1] > volume_media.iloc[-1]:
                    confirmacoes += 1
                if dados['Close'].iloc[-1] > dados['Close'].rolling(8).mean().iloc[-1]:
                    confirmacoes += 1
                    
                if confirmacoes >= self.min_confirmacoes:
                    padroes.append(Padrao(
                        nome="Morning Star",
                        forca=0.9,
                        direcao="CALL",
                        confiabilidade=0.85,
                        tipo="candlestick",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))
    
        except Exception as e:
            self.logger.error(f"Erro em padrões três velas: {str(e)}")
            import traceback
            self.logger.error(f"Stack trace: {traceback.format_exc()}")


       
    def _adicionar_analise_volume(self, padroes: List[Padrao], dados: pd.DataFrame):
        """Análise avançada de volume"""
        try:
            volume = dados['Volume']
            close = dados['Close']

            # Volume Spread Analysis (VSA)
            volume_ma = volume.rolling(5).mean()
            volume_std = volume.rolling(5).std()

            # Volume Climax
            if (volume.iloc[-1] > volume_ma.iloc[-1] + 2 * volume_std.iloc[-1] and
                close.pct_change().iloc[-1] > 0):

                confirmacoes = 0
                if close.iloc[-1] > close.rolling(8).mean().iloc[-1]: 
                    confirmacoes += 1
                if volume.iloc[-2] > volume_ma.iloc[-2]:
                    confirmacoes += 1

                if confirmacoes >= self.min_confirmacoes:
                    padroes.append(Padrao(
                        nome="Volume Climax",
                        forca=0.85,
                        direcao="CALL" if close.pct_change().iloc[-1] > 0 else "PUT",
                        confiabilidade=0.8,
                        tipo="volume",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

        except Exception as e:
            self.logger.error(f"Erro na análise de volume: {str(e)}")
            
            
    def _adicionar_indicadores_tecnico(self, dados: pd.DataFrame) -> Dict:
        # Média móvel adaptativa
        ema_rapida = ta.trend.EMAIndicator(dados['Close'], window=5).ema_indicator()
        ema_media = ta.trend.EMAIndicator(dados['Close'], window=13).ema_indicator()

        # CCI - Commodity Channel Index
        cci = ta.trend.CCIIndicator(dados['High'], dados['Low'], dados['Close'], window=20)

        # Force Index
        force_index = (dados['Close'] - dados['Close'].shift(1)) * dados['Volume']

        return {
            'ema_cross': ema_rapida[-1] > ema_media[-1],
            'cci': cci.cci()[-1],
            'force_index': force_index[-1]
        }
        
        
    def _analisar_price_action(self, dados: pd.DataFrame) -> Dict:
        try:
            closes = dados['Close']
            highs = dados['High']
            lows = dados['Low']

            # Suporte e resistência 
            suporte = min(lows[-20:])
            resistencia = max(highs[-20:])

            # Topos e fundos
            topos = []
            fundos = []

            for i in range(2, len(dados)-2):
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
                   highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    topos.append(highs[i])

                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
                   lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    fundos.append(lows[i])

            return {
                'suporte': suporte,
                'resistencia': resistencia,
                'tendencia': 'ALTA' if len(topos) > len(fundos) else 'BAIXA',
                'forca_tendencia': abs(len(topos) - len(fundos)) / 20
            }

        except Exception as e:
            self.logger.error(f"Erro análise price action: {str(e)}")
            return {}
    def _determinar_direcao_avancada(self, padroes: List[Padrao], 
                                    indicadores: Dict,
                                    dados: pd.DataFrame) -> str:
        try:
            # Pesos para cada componente
            pesos = {
                'price_action': 0.3,
                'indicadores': 0.3,
                'padroes': 0.4
            }

            # Análise de price action
            pa = self._analisar_price_action(dados)
            score_pa = 1 if pa['tendencia'] == 'ALTA' else -1
            score_pa *= pa['forca_tendencia']

            # Score dos indicadores
            score_ind = (
                (1 if indicadores['ema_cross'] else -1) * 0.4 +
                (indicadores['cci'] / 100) * 0.3 +
                (1 if indicadores['force_index'] > 0 else -1) * 0.3
            )

            # Score dos padrões
            score_padroes = sum(
                p.forca * (1 if p.direcao == 'CALL' else -1)
                for p in padroes
            ) / len(padroes)

            score_final = (
                score_pa * pesos['price_action'] +
                score_ind * pesos['indicadores'] +
                score_padroes * pesos['padroes']
            )

            return 'CALL' if score_final > 0 else 'PUT'

        except Exception as e:
            self.logger.error(f"Erro direção avançada: {str(e)}")
            return 'NEUTRO'
        
        
    def _adicionar_price_action(self, padroes: List[Padrao], dados: pd.DataFrame):
        """Adiciona análise de price action aos padrões"""
        try:
            closes = dados['Close']
            highs = dados['High']
            lows = dados['Low']
    
            # Tendência direcional
            direcao = 'NEUTRO'
            if len(closes) >= 3:
                if closes.iloc[-1] > closes.iloc[-2] > closes.iloc[-3]:
                    direcao = 'CALL'
                elif closes.iloc[-1] < closes.iloc[-2] < closes.iloc[-3]:
                    direcao = 'PUT'
    
            confirmacoes = 0
    
            # Verifica suporte/resistência
            suporte = min(lows[-20:])
            resistencia = max(highs[-20:])
            preco_atual = closes.iloc[-1]
    
            # Verifica proximidade dos níveis
            near_support = abs(preco_atual - suporte) / preco_atual < 0.001
            near_resistance = abs(preco_atual - resistencia) / preco_atual < 0.001
    
            if (near_support and direcao == 'CALL') or (near_resistance and direcao == 'PUT'):
                confirmacoes += 1
    
            # Verifica fechamento dos candles
            if len(closes) >= 3:
                body_size = abs(closes.iloc[-1] - closes.iloc[-2])
                avg_body = abs(closes - closes.shift()).mean()
                
                if body_size > avg_body * 1.2:  # Candle forte
                    confirmacoes += 1
    
            # Adiciona padrão se houver confirmações suficientes
            if confirmacoes >= 2:
                padroes.append(Padrao(
                    nome="Price Action",
                    forca=0.8,
                    direcao=direcao,
                    confiabilidade=0.75,
                    tipo="price_action",
                    tempo_expiracao=3,
                    confirmacoes=confirmacoes
                ))
    
        except Exception as e:
            self.logger.error(f"Erro em price action: {str(e)}")