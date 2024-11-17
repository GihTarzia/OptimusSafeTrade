import numpy as np
import pandas as pd
import ta
from datetime import datetime
import yfinance as yf
from typing import Dict, List, Union
from dataclasses import dataclass
from colorama import Fore, Style

@dataclass
class Padrao:
    nome: str
    forca: float  # 0 a 1
    direcao: str  # 'CALL' ou 'PUT'
    confiabilidade: float  # Histórico de acertos
    tipo: str  # Categoria do padrão (candlestick, tendência, etc)
    tempo_expiracao: int  # Tempo sugerido para expiração em minutos

class AnalisePadroesComplexos:
    def __init__(self):
        self.padroes_historico = {}
        self.min_confiabilidade = 0.5
        self.periodos_analise = {
            'curto': 14,
            'medio': 28,
            'longo': 56
        }
        
        # Configurações específicas por tipo de ativo
        self.config_ativos = {
            'forex': {
                'volatilidade_min': 0.0001,
                'volume_min': 100,
                'tempo_padrao': 5
            },
            'indices': {
                'volatilidade_min': 0.001,
                'volume_min': 1000,
                'tempo_padrao': 15
            },
            'commodities': {
                'volatilidade_min': 0.002,
                'volume_min': 500,
                'tempo_padrao': 10
            }
        }

    def _get_config_ativo(self, ativo: str) -> Dict:
        """Retorna configurações específicas para o tipo de ativo"""
        if any(par in ativo for par in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD']):
            return {
                'volatilidade_min': 0.00005,
                'volume_min': 0,  # Forex não precisa verificar volume
                'tempo_padrao': 5
            }
        elif any(indice in ativo for indice in ['^GSPC', '^DJI', '^IXIC']):
            return {
                'volatilidade_min': 0.001,
                'volume_min': 1000,
                'tempo_padrao': 15
            }
        else:
            return {
                'volatilidade_min': 0.002,
                'volume_min': 500,
                'tempo_padrao': 10
            }

    def identificar_padroes(self, dados: pd.DataFrame, ativo: str) -> List[Padrao]:
        """Identifica todos os padrões presentes nos dados"""
        padroes = []
        
        try:
            config = self._get_config_ativo(ativo)
            
            # Verifica condições mínimas
            volatilidade = dados['Close'].pct_change().std()
            volume_medio = dados['Volume'].mean() if 'Volume' in dados.columns else 0
            
            if volatilidade < config['volatilidade_min']:
                print(f"{Fore.YELLOW}Volatilidade muito baixa para {ativo}{Style.RESET_ALL}")
                return []
                
            if volume_medio < config['volume_min'] and 'Volume' in dados.columns:
                print(f"{Fore.YELLOW}Volume muito baixo para {ativo}{Style.RESET_ALL}")
                return []
            
            # 1. Padrões de Candlestick
            self._adicionar_padroes_candlestick(padroes, dados, config)
            
            # 2. Padrões de Tendência
            self._adicionar_padroes_tendencia(padroes, dados, config)
            
            # 3. Padrões de Momentum
            self._adicionar_padroes_momentum(padroes, dados, config)
            
            # 4. Padrões de Volatilidade
            self._adicionar_padroes_volatilidade(padroes, dados, config)
            
            # 5. Padrões de Suporte/Resistência
            self._adicionar_padroes_sr(padroes, dados, config)
            
            # 6. Padrões Harmônicos
            self._adicionar_padroes_harmonicos(padroes, dados, config)
            
            # Filtra padrões por confiabilidade
            padroes = [p for p in padroes if p.confiabilidade >= self.min_confiabilidade]
            
            # Agrupa padrões similares e ajusta força
            padroes = self._consolidar_padroes(padroes)
            
            return padroes
            
        except Exception as e:
            print(f"{Fore.RED}Erro ao identificar padrões: {str(e)}{Style.RESET_ALL}")
            return []
    def _adicionar_padroes_candlestick(self, padroes: List[Padrao], dados: pd.DataFrame, config: Dict):
        """Identifica padrões de candlestick"""
        try:
            ultimo_candle = dados.iloc[-1]
            penultimo_candle = dados.iloc[-2]
            
            # Doji
            corpo = abs(ultimo_candle['Open'] - ultimo_candle['Close'])
            range_total = ultimo_candle['High'] - ultimo_candle['Low']
            
            if corpo <= range_total * 0.1:
                padroes.append(Padrao(
                    nome="Doji",
                    forca=0.7,
                    direcao="NEUTRO",
                    confiabilidade=0.65,
                    tipo="candlestick",
                    tempo_expiracao=config['tempo_padrao']
                ))
            
            # Martelo
            sombra_inferior = min(ultimo_candle['Open'], ultimo_candle['Close']) - ultimo_candle['Low']
            sombra_superior = ultimo_candle['High'] - max(ultimo_candle['Open'], ultimo_candle['Close'])
            
            if sombra_inferior > corpo * 2 and sombra_superior < corpo:
                padroes.append(Padrao(
                    nome="Martelo",
                    forca=0.8,
                    direcao="CALL",
                    confiabilidade=0.75,
                    tipo="candlestick",
                    tempo_expiracao=config['tempo_padrao']
                ))
            
            # Estrela Cadente
            if sombra_superior > corpo * 2 and sombra_inferior < corpo:
                padroes.append(Padrao(
                    nome="Estrela Cadente",
                    forca=0.8,
                    direcao="PUT",
                    confiabilidade=0.75,
                    tipo="candlestick",
                    tempo_expiracao=config['tempo_padrao']
                ))
            
            # Engolfo de Alta
            if (penultimo_candle['Close'] < penultimo_candle['Open'] and  # Candle anterior vermelho
                ultimo_candle['Close'] > ultimo_candle['Open'] and        # Candle atual verde
                ultimo_candle['Open'] < penultimo_candle['Close'] and     # Abre abaixo do fechamento anterior
                ultimo_candle['Close'] > penultimo_candle['Open']):       # Fecha acima da abertura anterior
                
                padroes.append(Padrao(
                    nome="Engolfo de Alta",
                    forca=0.85,
                    direcao="CALL",
                    confiabilidade=0.8,
                    tipo="candlestick",
                    tempo_expiracao=config['tempo_padrao']
                ))
            
            # Engolfo de Baixa
            if (penultimo_candle['Close'] > penultimo_candle['Open'] and  # Candle anterior verde
                ultimo_candle['Close'] < ultimo_candle['Open'] and        # Candle atual vermelho
                ultimo_candle['Open'] > penultimo_candle['Close'] and     # Abre acima do fechamento anterior
                ultimo_candle['Close'] < penultimo_candle['Open']):       # Fecha abaixo da abertura anterior
                
                padroes.append(Padrao(
                    nome="Engolfo de Baixa",
                    forca=0.85,
                    direcao="PUT",
                    confiabilidade=0.8,
                    tipo="candlestick",
                    tempo_expiracao=config['tempo_padrao']
                ))
                
        except Exception as e:
            print(f"Erro em padrões candlestick: {str(e)}")

    def _adicionar_padroes_tendencia(self, padroes: List[Padrao], dados: pd.DataFrame, config: Dict):
        """Identifica padrões de tendência"""
        try:
            close = dados['Close']
            
            # Médias Móveis
            ema9 = ta.trend.EMAIndicator(close, window=9).ema_indicator()
            ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
            ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
            
            # Cruzamentos de Médias
            if (ema9.iloc[-1] > ema21.iloc[-1] and 
                ema9.iloc[-2] <= ema21.iloc[-2]):
                padroes.append(Padrao(
                    nome="Cruzamento EMA - Golden Cross",
                    forca=0.85,
                    direcao="CALL",
                    confiabilidade=0.8,
                    tipo="tendencia",
                    tempo_expiracao=config['tempo_padrao'] * 2
                ))
            elif (ema9.iloc[-1] < ema21.iloc[-1] and 
                  ema9.iloc[-2] >= ema21.iloc[-2]):
                padroes.append(Padrao(
                    nome="Cruzamento EMA - Death Cross",
                    forca=0.85,
                    direcao="PUT",
                    confiabilidade=0.8,
                    tipo="tendencia",
                    tempo_expiracao=config['tempo_padrao'] * 2
                ))
            
            # Tendência forte
            if (ema9.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1] and
                close.iloc[-1] > ema9.iloc[-1]):
                padroes.append(Padrao(
                    nome="Tendência Forte de Alta",
                    forca=1.0,
                    direcao="CALL",
                    confiabilidade=0.85,
                    tipo="tendencia",
                    tempo_expiracao=config['tempo_padrao'] * 3
                ))
            elif (ema9.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1] and
                  close.iloc[-1] < ema9.iloc[-1]):
                padroes.append(Padrao(
                    nome="Tendência Forte de Baixa",
                    forca=0.9,
                    direcao="PUT",
                    confiabilidade=0.85,
                    tipo="tendencia",
                    tempo_expiracao=config['tempo_padrao'] * 3
                ))
                
        except Exception as e:
            print(f"Erro em padrões tendência: {str(e)}")
        
    def _adicionar_padroes_momentum(self, padroes: List[Padrao], dados: pd.DataFrame, config: Dict):
        """Identifica padrões de momentum"""
        try:
            # RSI
            rsi = ta.momentum.RSIIndicator(dados['Close']).rsi()
            
            if rsi.iloc[-1] < 30:
                padroes.append(Padrao(
                    nome="RSI Sobrevendido",
                    forca=0.9,
                    direcao="CALL",
                    confiabilidade=0.85,
                    tipo="momentum",
                    tempo_expiracao=config['tempo_padrao']
                ))
            elif rsi.iloc[-1] > 70:
                padroes.append(Padrao(
                    nome="RSI Sobrecomprado",
                    forca=0.9,
                    direcao="PUT",
                    confiabilidade=0.85,
                    tipo="momentum",
                    tempo_expiracao=config['tempo_padrao']
                ))
            
            # Estocástico
            stoch = ta.momentum.StochasticOscillator(
                dados['High'], dados['Low'], dados['Close']
            )
            k = stoch.stoch()
            d = stoch.stoch_signal()
            
            if k.iloc[-1] < 20 and d.iloc[-1] < 20:
                padroes.append(Padrao(
                    nome="Estocástico Sobrevendido",
                    forca=0.85,
                    direcao="CALL",
                    confiabilidade=0.8,
                    tipo="momentum",
                    tempo_expiracao=config['tempo_padrao']
                ))
            elif k.iloc[-1] > 80 and d.iloc[-1] > 80:
                padroes.append(Padrao(
                    nome="Estocástico Sobrecomprado",
                    forca=0.85,
                    direcao="PUT",
                    confiabilidade=0.8,
                    tipo="momentum",
                    tempo_expiracao=config['tempo_padrao']
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
                    confiabilidade=0.7,
                    tipo="momentum",
                    tempo_expiracao=config['tempo_padrao'] * 2
                ))
            elif (macd_line.iloc[-1] < signal_line.iloc[-1] and 
                  macd_line.iloc[-2] >= signal_line.iloc[-2]):
                padroes.append(Padrao(
                    nome="MACD Cruzamento Baixa",
                    forca=0.75,
                    direcao="PUT",
                    confiabilidade=0.7,
                    tipo="momentum",
                    tempo_expiracao=config['tempo_padrao'] * 2
                ))
                
        except Exception as e:
            print(f"Erro em padrões momentum: {str(e)}")

    def _adicionar_padroes_volatilidade(self, padroes: List[Padrao], dados: pd.DataFrame, config: Dict):
        """Identifica padrões de volatilidade"""
        try:
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(dados['Close'])
            
            if dados['Close'].iloc[-1] < bb.bollinger_lband().iloc[-1]:
                padroes.append(Padrao(
                    nome="BB Oversold",
                    forca=0.8,
                    direcao="CALL",
                    confiabilidade=0.75,
                    tipo="volatilidade",
                    tempo_expiracao=config['tempo_padrao']
                ))
            elif dados['Close'].iloc[-1] > bb.bollinger_hband().iloc[-1]:
                padroes.append(Padrao(
                    nome="BB Overbought",
                    forca=0.8,
                    direcao="PUT",
                    confiabilidade=0.75,
                    tipo="volatilidade",
                    tempo_expiracao=config['tempo_padrao']
                ))
            
            # ATR para volatilidade alta
            atr = ta.volatility.AverageTrueRange(
                dados['High'], dados['Low'], dados['Close']
            ).average_true_range()
            
            atr_medio = atr.rolling(window=20).mean()
            if atr.iloc[-1] > atr_medio.iloc[-1] * 1.5:
                padroes.append(Padrao(
                    nome="Alta Volatilidade",
                    forca=0.6,
                    direcao="NEUTRO",
                    confiabilidade=0.65,
                    tipo="volatilidade",
                    tempo_expiracao=config['tempo_padrao'] * 2
                ))
                
        except Exception as e:
            print(f"Erro em padrões volatilidade: {str(e)}")
    def _adicionar_padroes_sr(self, padroes: List[Padrao], dados: pd.DataFrame, config: Dict):
        """Identifica níveis de suporte e resistência"""
        try:
            # Calcula Pivot Points
            pivots = self._calcular_pivot_points(dados)
            ultimo_preco = dados['Close'].iloc[-1]
            
            # Verifica proximidade com níveis
            for nivel, valor in pivots.items():
                if abs(ultimo_preco - valor) / valor < 0.001:  # 0.1% de proximidade
                    if ultimo_preco > valor:
                        padroes.append(Padrao(
                            nome=f"Teste Resistência {nivel}",
                            forca=0.7,
                            direcao="PUT",
                            confiabilidade=0.7,
                            tipo="suporte_resistencia",
                            tempo_expiracao=config['tempo_padrao']
                        ))
                    else:
                        padroes.append(Padrao(
                            nome=f"Teste Suporte {nivel}",
                            forca=0.7,
                            direcao="CALL",
                            confiabilidade=0.7,
                            tipo="suporte_resistencia",
                            tempo_expiracao=config['tempo_padrao']
                        ))
                        
        except Exception as e:
            print(f"Erro em padrões S/R: {str(e)}")

    def _adicionar_padroes_harmonicos(self, padroes: List[Padrao], dados: pd.DataFrame, config: Dict):
        """Identifica padrões harmônicos (Gartley, Butterfly, etc)"""
        try:
            # Encontra pontos de swing (máximos e mínimos locais)
            highs = dados['High'].rolling(window=5, center=True).max()
            lows = dados['Low'].rolling(window=5, center=True).min()
            
            # Verifica padrão Gartley
            if self._verificar_gartley(highs, lows):
                padroes.append(Padrao(
                    nome="Padrão Gartley",
                    forca=0.9,
                    direcao="CALL" if dados['Close'].iloc[-1] > dados['Open'].iloc[-1] else "PUT",
                    confiabilidade=0.85,
                    tipo="harmonico",
                    tempo_expiracao=config['tempo_padrao'] * 2
                ))
                
        except Exception as e:
            print(f"Erro em padrões harmônicos: {str(e)}")

    def _calcular_pivot_points(self, dados: pd.DataFrame) -> Dict:
        """Calcula os níveis de Pivot Point"""
        high = dados['High'].iloc[-20:].max()
        low = dados['Low'].iloc[-20:].min()
        close = dados['Close'].iloc[-1]
        
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

    def _verificar_gartley(self, highs: pd.Series, lows: pd.Series) -> bool:
        """Verifica se existe um padrão Gartley"""
        try:
            # Simplificação do padrão Gartley
            # Em uma implementação real, isso seria muito mais complexo
            return False  # Por enquanto, retorna False
        except Exception as e:
            print(f"Erro ao verificar Gartley: {str(e)}")
            return False

    def _consolidar_padroes(self, padroes: List[Padrao]) -> List[Padrao]:
        """Agrupa padrões similares e ajusta suas forças"""
        if not padroes:
            return []
            
        # Agrupa por direção
        calls = [p for p in padroes if p.direcao == "CALL"]
        puts = [p for p in padroes if p.direcao == "PUT"]
        neutros = [p for p in padroes if p.direcao == "NEUTRO"]
        
        # Calcula força média por direção
        forca_calls = sum(p.forca * p.confiabilidade for p in calls) / len(calls) if calls else 0
        forca_puts = sum(p.forca * p.confiabilidade for p in puts) / len(puts) if puts else 0
        
        # Ajusta forças baseado em confirmações múltiplas
        for padrao in padroes:
            if padrao.direcao == "CALL":
                padrao.forca *= (1 + 0.1 * len(calls))  # +10% por confirmação
            elif padrao.direcao == "PUT":
                padrao.forca *= (1 + 0.1 * len(puts))   # +10% por confirmação
            
            # Limita força máxima a 1
            padrao.forca = min(padrao.forca, 1.0)
        
        return padroes

    def analisar(self, ativo: str, periodo: str = '1d', intervalo: str = '5m') -> Dict:
        """Realiza análise completa de um ativo"""
        try:
            # Baixa dados
            df = yf.download(ativo, period=periodo, interval=intervalo, progress=False)
            if df.empty:
                return None
            
            # Identifica padrões
            padroes = self.identificar_padroes(df, ativo)
            
            # Calcula força combinada
            forca_sinais = sum(p.forca * p.confiabilidade for p in padroes) / len(padroes) if padroes else 0
            
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
                'confiabilidade_media': np.mean([p.confiabilidade for p in padroes]) if padroes else 0,
                'tempo_expiracao': max([p.tempo_expiracao for p in padroes]) if padroes else 5
            }
            
        except Exception as e:
            print(f"Erro na análise de padrões: {str(e)}")
            return None