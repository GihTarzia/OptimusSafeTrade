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
        self.min_confiabilidade = 0.60
           # Configurações de análise
        self.rsi_config = self.config.get('analise.rsi')
        self.bb_config = self.config.get('analise.bandas_bollinger')
        self.macd_config = self.config.get('analise.macd')
        self.cache_analises = {}

    def _calcular_volatilidade(self, ativo: str) -> float:
        """Calcula volatilidade atual do ativo"""
        try:
            dados = yf.download(
                ativo,
                period="1d",
                interval="1m",
                progress=False
            )
            if dados is not None and not dados.empty:
                # Calcula volatilidade usando retornos dos últimos candles
                volatilidade = dados['Close'].pct_change().tail(20).std() * np.sqrt(252)
                self.logger.info(f"Volatilidade calculada para {ativo}: {volatilidade:.4f}")
                return volatilidade
            
            self.logger.warning(f"Sem dados para calcular volatilidade de {ativo}")
            return 0
        except:
            return 0
  
        
    def _adicionar_padroes_candlestick(self, padroes: List[Padrao], dados: pd.DataFrame, config: Dict):
        """Identifica padrões de candlestick com confirmações"""
        try:
            if len(dados) < 2:  # Mínimo necessário para análise
                return
            ultimo_candle = dados.iloc[-1]
            penultimo_candle = dados.iloc[-2]
            
            # Calcula confirmações técnicas
            rsi = ta.momentum.RSIIndicator(dados['Close']).rsi().iloc[-1]
            stoch = ta.momentum.StochasticOscillator(
                dados['High'], dados['Low'], dados['Close']
            ).stoch().iloc[-1]
            macd = ta.trend.MACD(dados['Close'])
            macd_line = macd.macd().iloc[-1]
            signal_line = macd.macd_signal().iloc[-1]
            
            # Doji
            corpo = abs(ultimo_candle['Open'] - ultimo_candle['Close'])
            range_total = ultimo_candle['High'] - ultimo_candle['Low']
            
            if corpo <= range_total * 0.1:
                # Confirmações para Doji
                confirmacoes = 0
                if (rsi > 45 and rsi < 55):  # RSI neutro
                    confirmacoes += 1
                if (stoch > 40 and stoch < 60):  # Estocástico neutro
                    confirmacoes += 1
                if abs(macd_line - signal_line) < 0.0001:  # MACD próximo da linha de sinal
                    confirmacoes += 1
                
                if confirmacoes >= 2:
                    padroes.append(Padrao(
                        nome="Doji",
                        forca=0.7,
                        direcao="NEUTRO",
                        confiabilidade=0.65,
                        tipo="candlestick",
                        tempo_expiracao=self.config.get('ativos.config.tempo_padrao'),
                        confirmacoes=confirmacoes
                    ))
            
            # Martelo
            sombra_inferior = min(ultimo_candle['Open'], ultimo_candle['Close']) - ultimo_candle['Low']
            sombra_superior = ultimo_candle['High'] - max(ultimo_candle['Open'], ultimo_candle['Close'])
            
            if sombra_inferior > corpo * 2 and sombra_superior < corpo:
                confirmacoes = 0
                if rsi < 40:  # Sobrevendido
                    confirmacoes += 1
                if stoch < 30:  # Sobrevendido
                    confirmacoes += 1
                if macd_line > signal_line:  # MACD virando positivo
                    confirmacoes += 1
                                
                # Confirmação de volume
                if 'Volume' in dados.columns:
                    volume_atual = dados['Volume'].iloc[-1]
                    volume_medio = dados['Volume'].rolling(20).mean().iloc[-1]
                    if volume_atual > volume_medio * 1.2:  # Volume 20% acima da média
                        confirmacoes += 1
                        
                if confirmacoes >= self.min_confirmacoes:
                    padroes.append(Padrao(
                        nome="Martelo",
                        forca=0.8,
                        direcao="CALL",
                        confiabilidade=0.75,
                        tipo="candlestick",
                        tempo_expiracao=self.config.get('ativos.config.tempo_padrao'),
                        confirmacoes=confirmacoes
                    ))
            
            # Estrela Cadente
            if sombra_superior > corpo * 2 and sombra_inferior < corpo:
                confirmacoes = 0
                if rsi > 60:  # Sobrecomprado
                    confirmacoes += 1
                if stoch > 70:  # Sobrecomprado
                    confirmacoes += 1
                if macd_line < signal_line:  # MACD virando negativo
                    confirmacoes += 1
                
                # Confirmação de volume
                if 'Volume' in dados.columns:
                    volume_atual = dados['Volume'].iloc[-1]
                    volume_medio = dados['Volume'].rolling(20).mean().iloc[-1]
                    if volume_atual > volume_medio * 1.2:
                        confirmacoes += 1
                        
                if confirmacoes >= self.min_confirmacoes:       
                    padroes.append(Padrao(
                        nome="Estrela Cadente",
                        forca=0.8,
                        direcao="PUT",
                        confiabilidade=0.75,
                        tipo="candlestick",
                        tempo_expiracao=self.config.get('ativos.config.tempo_padrao'),
                        confirmacoes=confirmacoes
                    ))
            
            # Engolfo de Alta
            if (penultimo_candle['Close'] < penultimo_candle['Open'] and  # Candle anterior vermelho
                ultimo_candle['Close'] > ultimo_candle['Open'] and        # Candle atual verde
                ultimo_candle['Open'] < penultimo_candle['Close'] and     # Abre abaixo do fechamento anterior
                ultimo_candle['Close'] > penultimo_candle['Open']):       # Fecha acima da abertura anterior
                
                confirmacoes = 0
                
                # Confirmações técnicas
                if rsi > 40 and rsi < 60:  # RSI em zona neutra
                    confirmacoes += 1
                if macd_line > signal_line:  # MACD positivo
                    confirmacoes += 1
                if stoch > 30 and stoch < 70:  # Estocástico em zona neutra
                    confirmacoes += 1
                
                # Confirmação de tendência
                ema20 = ta.trend.EMAIndicator(dados['Close'], window=20).ema_indicator()
                if dados['Close'].iloc[-1] > ema20.iloc[-1]:
                    confirmacoes += 1
                
                if confirmacoes >= self.min_confirmacoes:
                    padroes.append(Padrao(
                        nome="Engolfo de Alta",
                        forca=0.85,
                        direcao="CALL",
                        confiabilidade=0.8,
                        tipo="candlestick",
                        tempo_expiracao=self.config.get('ativos.config.tempo_padrao'),
                        confirmacoes=confirmacoes
                    ))
        
            # Engolfo de Baixa
            if (penultimo_candle['Close'] > penultimo_candle['Open'] and  # Candle anterior verde
                ultimo_candle['Close'] < ultimo_candle['Open'] and        # Candle atual vermelho
                ultimo_candle['Open'] > penultimo_candle['Close'] and     # Abre acima do fechamento anterior
                ultimo_candle['Close'] < penultimo_candle['Open']):       # Fecha abaixo da abertura anterior
                
                confirmacoes = 0
                
                # Confirmações técnicas
                if rsi > 40 and rsi < 60:  # RSI em zona neutra
                    confirmacoes += 1
                if macd_line < signal_line:  # MACD negativo
                    confirmacoes += 1
                if stoch > 30 and stoch < 70:  # Estocástico em zona neutra
                    confirmacoes += 1
                
                # Confirmação de tendência
                ema20 = ta.trend.EMAIndicator(dados['Close'], window=20).ema_indicator()
                if dados['Close'].iloc[-1] < ema20.iloc[-1]:
                    confirmacoes += 1
                
                if confirmacoes >= self.min_confirmacoes:
                    padroes.append(Padrao(
                        nome="Engolfo de Baixa",
                        forca=0.85,
                        direcao="PUT",
                        confiabilidade=0.8,
                        tipo="candlestick",
                        tempo_expiracao=self.config.get('ativos.config.tempo_padrao'),
                        confirmacoes=confirmacoes
                    ))
                
        except Exception as e:
            self.logger.error(f"Erro em padrões candlestick: {str(e)}")
            
        
    def _adicionar_padroes_tendencia(self, padroes: List[Padrao], dados: pd.DataFrame, config: Dict):
        """Identifica padrões de tendência"""
        try:
            confirmacoes = 0
            close = dados['Close']
            
            # Médias Móveis
            ema9 = ta.trend.EMAIndicator(close, window=5).ema_indicator()
            ema21 = ta.trend.EMAIndicator(close, window=9).ema_indicator()
            ema50 = ta.trend.EMAIndicator(close, window=13).ema_indicator()
            
            # RSI e Estocástico para confirmação
            rsi = ta.momentum.RSIIndicator(close).rsi()
            stoch = ta.momentum.StochasticOscillator(
                dados['High'], dados['Low'], dados['Close']
            )
            
            # MACD para confirmação adicional
            macd = ta.trend.MACD(close)
            macd_line = macd.macd()
            signal_line = macd.macd_signal()
            
            # Cruzamentos de Médias
            if (ema9.iloc[-1] > ema21.iloc[-1] and 
                ema9.iloc[-2] <= ema21.iloc[-2]):
                confirmacoes = 0
                
                # Confirmações para CALL
                if rsi.iloc[-1] > 40:  # RSI em zona favorável
                    confirmacoes += 1
                if stoch.stoch().iloc[-1] > 30:  # Estocástico saindo de sobrevendido
                    confirmacoes += 1
                if macd_line.iloc[-1] > signal_line.iloc[-1]:  # MACD positivo
                    confirmacoes += 1
                if close.iloc[-1] > ema50.iloc[-1]:  # Preço acima da EMA 50
                    confirmacoes += 1
                    
                # Volume crescente
                if 'Volume' in dados.columns:
                    volume_crescente = dados['Volume'].iloc[-1] > dados['Volume'].rolling(5).mean().iloc[-1]
                    if volume_crescente:
                        confirmacoes += 1
                
                if confirmacoes >= self.min_confirmacoes:
                    padroes.append(Padrao(
                        nome="Golden Cross EMA",
                        forca=0.85,
                        direcao="CALL",
                        confiabilidade=0.8,
                        tipo="tendencia",
                        tempo_expiracao=self.config.get('ativos.config.tempo_padrao') * 2,
                        confirmacoes=confirmacoes
                    ))

            elif (ema9.iloc[-1] < ema21.iloc[-1] and 
                  ema9.iloc[-2] >= ema21.iloc[-2]):
              
                confirmacoes = 0
                
                # Confirmações para PUT
                if rsi.iloc[-1] < 60:  # RSI em zona favorável
                    confirmacoes += 1
                if stoch.stoch().iloc[-1] < 70:  # Estocástico saindo de sobrecomprado
                    confirmacoes += 1
                if macd_line.iloc[-1] < signal_line.iloc[-1]:  # MACD negativo
                    confirmacoes += 1
                if close.iloc[-1] < ema50.iloc[-1]:  # Preço abaixo da EMA 50
                    confirmacoes += 1
                    
                # Volume crescente
                if 'Volume' in dados.columns:
                    volume_crescente = dados['Volume'].iloc[-1] > dados['Volume'].rolling(5).mean().iloc[-1]
                    if volume_crescente:
                        confirmacoes += 1
                
                if confirmacoes >= self.min_confirmacoes:
                    padroes.append(Padrao(
                        nome="Death Cross EMA",
                        forca=0.85,
                        direcao="PUT",
                        confiabilidade=0.8,
                        tipo="tendencia",
                        tempo_expiracao= self.config.get('ativos.config.tempo_padrao') * 2,
                        confirmacoes=confirmacoes
                    ))
                
        except Exception as e:
            self.logger.error(f"Erro em padrões tendência: {str(e)}")


    def _adicionar_padroes_momentum(self, padroes: List[Padrao], dados: pd.DataFrame, config: Dict):
       """Identifica padrões de momentum"""
       try:
           # RSI
           rsi = ta.momentum.RSIIndicator(dados['Close']).rsi()
           
           if rsi.iloc[-1] < 25:
               # Conta confirmações para RSI sobrevendido
               confirmacoes = 0
               if rsi.iloc[-2] < 25: confirmacoes += 1  # RSI mantendo-se sobrevendido
               if dados['Close'].iloc[-1] > dados['Close'].iloc[-2]: confirmacoes += 1  # Preço subindo
               if dados['Volume'].iloc[-1] > dados['Volume'].mean(): confirmacoes += 1  # Volume acima da média

               padroes.append(Padrao(
                   nome="RSI Sobrevendido",
                   forca=0.9,
                   direcao="CALL",
                   confiabilidade=0.85,
                   tipo="momentum",
                   tempo_expiracao=self.config.get('ativos.config.tempo_padrao'),
                   confirmacoes=confirmacoes
               ))
               
           elif rsi.iloc[-1] > 75:
               # Conta confirmações para RSI sobrecomprado
               confirmacoes = 0
               if rsi.iloc[-2] > 75: confirmacoes += 1  # RSI mantendo-se sobrecomprado
               if dados['Close'].iloc[-1] < dados['Close'].iloc[-2]: confirmacoes += 1  # Preço caindo
               if dados['Volume'].iloc[-1] > dados['Volume'].mean(): confirmacoes += 1  # Volume acima da média

               padroes.append(Padrao(
                   nome="RSI Sobrecomprado",
                   forca=0.9,
                   direcao="PUT",
                   confiabilidade=0.85,
                   tipo="momentum",
                   tempo_expiracao=self.config.get('ativos.config.tempo_padrao'),
                   confirmacoes=confirmacoes
               ))
           
           # Estocástico
           stoch = ta.momentum.StochasticOscillator(
                dados['High'], dados['Low'], dados['Close'],
                window=5,  # Reduzido de 14
                smooth_window=2  # Reduzido de 3
           )
           k = stoch.stoch()
           d = stoch.stoch_signal()
           
           if k.iloc[-1] < 20 and d.iloc[-1] < 20:
               # Confirmações para estocástico sobrevendido
               confirmacoes = 0
               if k.iloc[-2] < 20: confirmacoes += 1  # Mantendo-se sobrevendido
               if k.iloc[-1] > d.iloc[-1]: confirmacoes += 1  # Linha K cruzando acima do D
               if rsi.iloc[-1] < 40: confirmacoes += 1  # RSI também baixo

               padroes.append(Padrao(
                   nome="Estocástico Sobrevendido",
                   forca=0.85,
                   direcao="CALL",
                   confiabilidade=0.8,
                   tipo="momentum",
                   tempo_expiracao=self.config.get('ativos.config.tempo_padrao'),
                   confirmacoes=confirmacoes
               ))
               
           elif k.iloc[-1] > 80 and d.iloc[-1] > 80:
               # Confirmações para estocástico sobrecomprado
               confirmacoes = 0
               if k.iloc[-2] > 80: confirmacoes += 1  # Mantendo-se sobrecomprado
               if k.iloc[-1] < d.iloc[-1]: confirmacoes += 1  # Linha K cruzando abaixo do D
               if rsi.iloc[-1] > 60: confirmacoes += 1  # RSI também alto

               padroes.append(Padrao(
                   nome="Estocástico Sobrecomprado",
                   forca=0.85,
                   direcao="PUT",
                   confiabilidade=0.8,
                   tipo="momentum",
                   tempo_expiracao=self.config.get('ativos.config.tempo_padrao'),
                   confirmacoes=confirmacoes
               ))
           
           # MACD
           macd = ta.trend.MACD(
                    dados['Close'],
                    window_fast=6,   # Reduzido de 12
                    window_slow=13,  # Reduzido de 26
                    window_sign=4    # Reduzido de 9
                )
           macd_line = macd.macd()
           signal_line = macd.macd_signal()
           
           if (macd_line.iloc[-1] > signal_line.iloc[-1] and 
               macd_line.iloc[-2] <= signal_line.iloc[-2]):
               # Confirmações para cruzamento MACD alta
               confirmacoes = 0
               if macd_line.iloc[-1] > 0: confirmacoes += 1  # MACD positivo
               if dados['Close'].iloc[-1] > dados['Close'].iloc[-2]: confirmacoes += 1  # Preço subindo
               if rsi.iloc[-1] > 40 and rsi.iloc[-1] < 70: confirmacoes += 1  # RSI em zona neutra

               padroes.append(Padrao(
                   nome="MACD Cruzamento Alta",
                   forca=0.75,
                   direcao="CALL",
                   confiabilidade=0.7,
                   tipo="momentum",
                   tempo_expiracao=self.config.get('ativos.config.tempo_padrao') * 2,
                   confirmacoes=confirmacoes
               ))
               
           elif (macd_line.iloc[-1] < signal_line.iloc[-1] and 
                 macd_line.iloc[-2] >= signal_line.iloc[-2]):
               # Confirmações para cruzamento MACD baixa
               confirmacoes = 0
               if macd_line.iloc[-1] < 0: confirmacoes += 1  # MACD negativo
               if dados['Close'].iloc[-1] < dados['Close'].iloc[-2]: confirmacoes += 1  # Preço caindo
               if rsi.iloc[-1] > 30 and rsi.iloc[-1] < 60: confirmacoes += 1  # RSI em zona neutra

               padroes.append(Padrao(
                   nome="MACD Cruzamento Baixa",
                   forca=0.75,
                   direcao="PUT",
                   confiabilidade=0.7,
                   tipo="momentum",
                   tempo_expiracao=self.config.get('ativos.config.tempo_padrao') * 2,
                   confirmacoes=confirmacoes
               ))
                   
       except Exception as e:
           self.logger.error(f"Erro em padrões momentum: {str(e)}")       

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
                    tempo_expiracao=self.config.get('ativos.config.tempo_padrao')
                ))
            elif dados['Close'].iloc[-1] > bb.bollinger_hband().iloc[-1]:
                padroes.append(Padrao(
                    nome="BB Overbought",
                    forca=0.8,
                    direcao="PUT",
                    confiabilidade=0.75,
                    tipo="volatilidade",
                    tempo_expiracao=self.config.get('ativos.config.tempo_padrao')
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
                    tempo_expiracao=self.config.get('ativos.config.tempo_padrao') * 2
                ))
                
        except Exception as e:
            self.logger.error(f"Erro em padrões volatilidade: {str(e)}")

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

    async def analisar(self, dados: pd.DataFrame, ativo: str) -> Dict:
        """Realiza análise técnica com cache"""
        try:
            self.logger.debug(f"Iniciando análise para {ativo}")
            
            #if not self._validar_dados(dados):
            #    return None

            # Verifica cache
            cache_key = f"{ativo}_{dados.index[-1]}"
            if cache_key in self.cache_analises:
                return self.cache_analises[cache_key]

            # Identifica padrões
            padroes = await self._identificar_padroes_completos(dados, ativo)
            if not padroes:
                return None

            # Consolida resultados
            resultado = {
                'ativo': ativo,
                'timestamp': dados.index[-1],
                'direcao': self._determinar_direcao(padroes),
                'forca_sinal': self._calcular_forca(padroes),
                'padroes': [self._formatar_padrao(p) for p in padroes],
                'num_padroes': len(padroes)
            }

            # Atualiza cache
            self.cache_analises[cache_key] = resultado
            return resultado

        except Exception as e:
            self.logger.error(f"2Erro na análise de {ativo}: {str(e)}")
            return None
        
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

    def _determinar_direcao(self, padroes: List[Padrao]) -> str:
        """Determina a direção predominante dos padrões"""
        calls = sum(p.forca * p.confiabilidade for p in padroes if p.direcao == "CALL")
        puts = sum(p.forca * p.confiabilidade for p in padroes if p.direcao == "PUT")

        if calls > puts:
            return "CALL"
        elif puts > calls:
            return "PUT"
        else:
            return "NEUTRO"

    def _calcular_forca(self, padroes: List[Padrao]) -> float:
        """Calcula a força do sinal com base nos padrões"""
        forca_total = sum(p.forca * p.confiabilidade for p in padroes)
        return forca_total / len(padroes) if padroes else 0.0

    def _formatar_padrao(self, padrao: Padrao) -> Dict:
        """Formata os detalhes de um padrão"""
        return {
            "nome": padrao.nome,
            "forca": padrao.forca,
            "direcao": padrao.direcao,
            "confiabilidade": padrao.confiabilidade,
            "tipo": padrao.tipo,
            "tempo_expiracao": padrao.tempo_expiracao,
            "confirmacoes": padrao.confirmacoes
        }

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
