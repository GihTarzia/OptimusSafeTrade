import numpy as np
import pandas as pd
import ta
from datetime import datetime
import yfinance as yf
from typing import Dict, List, Union
from dataclasses import dataclass

@dataclass
class Padrao:
    nome: str
    forca: float  # 0 a 1
    direcao: str  # 'CALL' ou 'PUT'
    confiabilidade: float  # Histórico de acertos

class AnalisePadroesComplexos:
    def __init__(self):
        self.padroes_historico = {}
        self.min_confiabilidade = 0.6
        self.periodos_analise = {
            'curto': 14,
            'medio': 28,
            'longo': 56
        }

    def identificar_padroes(self, dados: pd.DataFrame) -> List[Padrao]:
        """Identifica todos os padrões presentes nos dados"""
        padroes = []
        
        try:
            # 1. Padrões de Candlestick
            self._adicionar_padroes_candlestick(padroes, dados)
            
            # 2. Padrões de Tendência
            self._adicionar_padroes_tendencia(padroes, dados)
            
            # 3. Padrões de Momentum
            self._adicionar_padroes_momentum(padroes, dados)
            
            # 4. Padrões de Volatilidade
            self._adicionar_padroes_volatilidade(padroes, dados)
            
            # 5. Padrões de Suporte/Resistência
            self._adicionar_padroes_sr(padroes, dados)
            
            # Filtra padrões por confiabilidade
            padroes = [p for p in padroes if p.confiabilidade >= self.min_confiabilidade]
            
            return padroes
            
        except Exception as e:
            print(f"Erro ao identificar padrões: {str(e)}")
            return []

    def _adicionar_padroes_candlestick(self, padroes: List[Padrao], dados: pd.DataFrame):
        """Identifica padrões de candlestick"""
        try:
            # Doji (corpo pequeno)
            ultimo_candle = dados.iloc[-1]
            corpo = abs(ultimo_candle['Open'] - ultimo_candle['Close'])
            range_total = ultimo_candle['High'] - ultimo_candle['Low']
            
            if corpo <= range_total * 0.1:
                padroes.append(Padrao(
                    nome="Doji",
                    forca=0.7,
                    direcao="NEUTRO",
                    confiabilidade=0.65
                ))
            
            # Martelo
            sombra_inferior = min(ultimo_candle['Open'], ultimo_candle['Close']) - ultimo_candle['Low']
            sombra_superior = ultimo_candle['High'] - max(ultimo_candle['Open'], ultimo_candle['Close'])
            
            if sombra_inferior > corpo * 2 and sombra_superior < corpo:
                padroes.append(Padrao(
                    nome="Martelo",
                    forca=0.8,
                    direcao="CALL",
                    confiabilidade=0.75
                ))
            
            # Estrela Cadente
            if sombra_superior > corpo * 2 and sombra_inferior < corpo:
                padroes.append(Padrao(
                    nome="Estrela Cadente",
                    forca=0.8,
                    direcao="PUT",
                    confiabilidade=0.75
                ))
                
        except Exception as e:
            print(f"Erro em padrões candlestick: {str(e)}")

    def _adicionar_padroes_tendencia(self, padroes: List[Padrao], dados: pd.DataFrame):
        """Identifica padrões de tendência"""
        try:
            close = dados['Close']
            
            # Médias Móveis
            ma_curta = ta.trend.SMAIndicator(close, self.periodos_analise['curto']).sma_indicator()
            ma_media = ta.trend.SMAIndicator(close, self.periodos_analise['medio']).sma_indicator()
            ma_longa = ta.trend.SMAIndicator(close, self.periodos_analise['longo']).sma_indicator()
            
            # Cruzamentos de Médias
            if (ma_curta.iloc[-1] > ma_media.iloc[-1] and 
                ma_curta.iloc[-2] <= ma_media.iloc[-2]):
                padroes.append(Padrao(
                    nome="Cruzamento MA - Golden Cross",
                    forca=0.85,
                    direcao="CALL",
                    confiabilidade=0.8
                ))
            elif (ma_curta.iloc[-1] < ma_media.iloc[-1] and 
                  ma_curta.iloc[-2] >= ma_media.iloc[-2]):
                padroes.append(Padrao(
                    nome="Cruzamento MA - Death Cross",
                    forca=0.85,
                    direcao="PUT",
                    confiabilidade=0.8
                ))
                
        except Exception as e:
            print(f"Erro em padrões tendência: {str(e)}")

    def _adicionar_padroes_momentum(self, padroes: List[Padrao], dados: pd.DataFrame):
        """Identifica padrões de momentum"""
        try:
            # RSI
            rsi = ta.momentum.RSIIndicator(dados['Close']).rsi()
            if rsi.iloc[-1] < 30:
                padroes.append(Padrao(
                    nome="RSI Sobrevendido",
                    forca=0.9,
                    direcao="CALL",
                    confiabilidade=0.85
                ))
            elif rsi.iloc[-1] > 70:
                padroes.append(Padrao(
                    nome="RSI Sobrecomprado",
                    forca=0.9,
                    direcao="PUT",
                    confiabilidade=0.85
                ))
            
            # MACD
            macd = ta.trend.MACD(dados['Close'])
            macd_line = macd.macd()
            signal_line = macd.macd_signal()
            
            if (macd_line.iloc[-1] > signal_line.iloc[-1] and 
                macd_line.iloc[-2] <= signal_line.iloc[-2]):
                padroes.append(Padrao(
                    nome="MACD Cruzamento Alta",
                    forca=0.75,
                    direcao="CALL",
                    confiabilidade=0.7
                ))
            elif (macd_line.iloc[-1] < signal_line.iloc[-1] and 
                  macd_line.iloc[-2] >= signal_line.iloc[-2]):
                padroes.append(Padrao(
                    nome="MACD Cruzamento Baixa",
                    forca=0.75,
                    direcao="PUT",
                    confiabilidade=0.7
                ))
                
        except Exception as e:
            print(f"Erro em padrões momentum: {str(e)}")

    def _adicionar_padroes_volatilidade(self, padroes: List[Padrao], dados: pd.DataFrame):
        """Identifica padrões de volatilidade"""
        try:
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(dados['Close'])
            
            if dados['Close'].iloc[-1] < bb.bollinger_lband().iloc[-1]:
                padroes.append(Padrao(
                    nome="BB Oversold",
                    forca=0.8,
                    direcao="CALL",
                    confiabilidade=0.75
                ))
            elif dados['Close'].iloc[-1] > bb.bollinger_hband().iloc[-1]:
                padroes.append(Padrao(
                    nome="BB Overbought",
                    forca=0.8,
                    direcao="PUT",
                    confiabilidade=0.75
                ))
            
            # ATR
            atr = ta.volatility.AverageTrueRange(
                dados['High'], dados['Low'], dados['Close']
            ).average_true_range()
            
            atr_medio = atr.rolling(window=20).mean()
            if atr.iloc[-1] > atr_medio.iloc[-1] * 1.5:
                padroes.append(Padrao(
                    nome="Alta Volatilidade",
                    forca=0.6,
                    direcao="NEUTRO",
                    confiabilidade=0.65
                ))
                
        except Exception as e:
            print(f"Erro em padrões volatilidade: {str(e)}")

    def _adicionar_padroes_sr(self, padroes: List[Padrao], close_prices):
        """Identifica níveis de suporte e resistência"""
        try:
            # Pivô Points
            pivots = self._calcular_pivot_points(close_prices)
            ultimo_preco = close_prices[-1]
            
            for nivel, valor in pivots.items():
                if abs(ultimo_preco - valor) / valor < 0.001:  # 0.1% de proximidade
                    if ultimo_preco > valor:
                        padroes.append(Padrao(
                            nome=f"Teste Resistência {nivel}",
                            forca=0.7,
                            direcao="PUT",
                            confiabilidade=0.7
                        ))
                    else:
                        padroes.append(Padrao(
                            nome=f"Teste Suporte {nivel}",
                            forca=0.7,
                            direcao="CALL",
                            confiabilidade=0.7
                        ))
        except Exception as e:
            print(f"Erro ao calcular S/R: {str(e)}")

    def _calcular_pivot_points(self, close_prices):
        """Calcula os níveis de Pivot Point"""
        high = np.max(close_prices[-20:])
        low = np.min(close_prices[-20:])
        close = close_prices[-1]
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        return {
            'P': pivot,
            'R1': r1,
            'R2': r2,
            'S1': s1,
            'S2': s2
        }

    def calcular_forca_sinais(self, padroes: List[Padrao]) -> float:
        """Calcula a força combinada dos sinais"""
        if not padroes:
            return 0
            
        forca_total = 0
        peso_total = 0
        
        for padrao in padroes:
            peso = padrao.forca * padrao.confiabilidade
            forca_total += peso
            peso_total += 1
            
        return forca_total / peso_total if peso_total > 0 else 0

    def analisar(self, ativo: str, periodo: str = '1d', intervalo: str = '5m') -> Dict:
        """Realiza análise completa de um ativo"""
        try:
            # Baixa dados
            df = yf.download(ativo, period=periodo, interval=intervalo, progress=False)
            if df.empty:
                return None
            
            # Identifica padrões
            padroes = self.identificar_padroes(df)
            
            # Calcula força combinada
            forca_sinais = self.calcular_forca_sinais(padroes)
            
            # Determina direção predominante
            calls = len([p for p in padroes if p.direcao == 'CALL'])
            puts = len([p for p in padroes if p.direcao == 'PUT'])
            
            direcao = 'CALL' if calls > puts else 'PUT' if puts > calls else 'NEUTRO'
            
            return {
                'ativo': ativo,
                'timestamp': datetime.now(),
                'direcao': direcao,
                'forca_sinal': forca_sinais,
                'padroes': [{'nome': p.nome, 'direcao': p.direcao, 'forca': p.forca} for p in padroes],
                'num_padroes': len(padroes),
                'confiabilidade_media': np.mean([p.confiabilidade for p in padroes]) if padroes else 0
            }
            
        except Exception as e:
            print(f"Erro na análise de padrões: {str(e)}")
            return None