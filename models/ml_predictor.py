import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import yfinance as yf
from datetime import datetime, timedelta
import ta
from colorama import Fore, Style
import warnings
warnings.filterwarnings('ignore')

class MLPredictor:
    def __init__(self):
        self.models = {}  # Dicionário para armazenar modelos por ativo
        self.scalers = {}  # Dicionário para armazenar scalers por ativo
        self.min_probabilidade = 0.60
        self.min_accuracy = 0.52  # Mínimo de acurácia aceitável
        
        # Parâmetros otimizados por tipo de ativo
        self.parametros_modelo = {
            'forex': {
                'max_depth': 6,
                'learning_rate': 0.03,
                'n_estimators': 300,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'random_state': 42
            },
            'indices': {
                'max_depth': 4,
                'learning_rate': 0.05,
                'n_estimators': 400,
                'min_child_weight': 5,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'objective': 'binary:logistic',
                'random_state': 42
            },
            'commodities': {
                'max_depth': 5,
                'learning_rate': 0.04,
                'n_estimators': 350,
                'min_child_weight': 4,
                'subsample': 0.75,
                'colsample_bytree': 0.75,
                'objective': 'binary:logistic',
                'random_state': 42
            }
        }

    def _get_modelo_params(self, ativo):
        """Retorna parâmetros específicos para cada tipo de ativo"""
        if any(par in ativo for par in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD']):
            return self.parametros_modelo['forex']
        elif any(indice in ativo for indice in ['^GSPC', '^DJI', '^IXIC']):
            return self.parametros_modelo['indices']
        else:
            return self.parametros_modelo['commodities']

    def criar_features(self, df):
        """Cria features para o modelo ML"""
        try:
            features = pd.DataFrame()
            
            if len(df) < 50:  # Verifica se há dados suficientes
                print(f"Dados insuficientes: {len(df)} registros (mínimo: 50)")
                return None
            
            print("Calculando indicadores técnicos...")
            
            # RSI
            features['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
            features['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            features['bb_high'] = bollinger.bollinger_hband()
            features['bb_low'] = bollinger.bollinger_lband()
            features['bb_position'] = bollinger.bollinger_pband()
            
            # Médias móveis exponenciais
            features['ema_9'] = ta.trend.EMAIndicator(df['Close'], window=9).ema_indicator()
            features['ema_21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
            
            # Ichimoku Cloud
            ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
            features['ichimoku_a'] = ichimoku.ichimoku_a()
            features['ichimoku_b'] = ichimoku.ichimoku_b()
            
            # Estocástico
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            features['stoch_k'] = stoch.stoch()
            features['stoch_d'] = stoch.stoch_signal()
            
            # ADX
            adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
            features['adx'] = adx.adx()
            features['adx_pos'] = adx.adx_pos()
            features['adx_neg'] = adx.adx_neg()
            
            # Volatilidade
            atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'])
            features['volatility'] = atr.average_true_range()
            
            # Momentum
            features['momentum'] = ta.momentum.ROCIndicator(df['Close']).roc()
            
            # Volume
            if 'Volume' in df.columns and df['Volume'].sum() > 0:
                features['volume_change'] = df['Volume'].pct_change()
                features['volume_ema'] = ta.trend.EMAIndicator(df['Volume'], window=20).ema_indicator()
            
            # Fibonacci Retracement levels
            max_price = df['High'].rolling(window=20).max()
            min_price = df['Low'].rolling(window=20).min()
            diff = max_price - min_price
            features['fib_236'] = (df['Close'] - (min_price + diff * 0.236)) / df['Close']
            features['fib_382'] = (df['Close'] - (min_price + diff * 0.382)) / df['Close']
            features['fib_618'] = (df['Close'] - (min_price + diff * 0.618)) / df['Close']
            
            # Tendências
            for periodo in [5, 10, 20]:
                sma = ta.trend.SMAIndicator(df['Close'], window=periodo)
                features[f'trend_{periodo}'] = (df['Close'] - sma.sma_indicator()) / df['Close']
            
            # Remove valores infinitos e NaN
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(0)
            
            print(f"Features criadas: {features.shape[1]} indicadores")
            return features
            
        except Exception as e:
            print(f"Erro ao criar features: {str(e)}")
            return None

    def atualizar_modelo(self, ativo: str, novos_dados: pd.DataFrame):
        """Atualiza o modelo com novos dados mantendo o aprendizado"""
        try:
            print(f"\nAtualizando modelo para {ativo}...")
            
            if ativo not in self.models:
                print(f"Modelo não existe para {ativo} - criando novo modelo")
                return self.treinar(novos_dados)
    
            # Prepara novos dados
            features = self.criar_features(novos_dados)
            if features is None:
                print("Erro ao criar features para atualização")
                return False
    
            # Remove registros com dados faltantes
            valid_idx = ~features.isnull().any(axis=1)
            features = features[valid_idx]
            
            # Cria labels (1 para alta, 0 para baixa)
            retornos_futuros = novos_dados['close'].pct_change().shift(-1)
            labels = (retornos_futuros > 0).astype(int)
            labels = labels[valid_idx]
            
            if len(features) < 50:  # Mínimo de dados para atualização
                print("Dados insuficientes para atualização")
                return False
    
            # Normaliza features usando o scaler existente
            features_scaled = self.scalers[ativo].transform(features)
            
            # Atualiza o modelo (partial_fit)
            modelo_atual = self.models[ativo]['model']
            
            # XGBoost não suporta partial_fit, então vamos retreinar com um conjunto maior
            modelo_atual.fit(
                features_scaled,
                labels,
                xgb_model=modelo_atual.get_booster(),  # Usa o modelo existente como base
                eval_set=[(features_scaled, labels)],
                eval_metric='logloss',
                early_stopping_rounds=20,
                verbose=False
            )
    
            # Avalia performance após atualização
            predicoes = modelo_atual.predict(features_scaled)
            accuracy_atual = (predicoes == labels).mean()
            
            print(f"Modelo atualizado - Acurácia atual: {accuracy_atual:.2%}")
            
            # Atualiza métricas do modelo
            self.models[ativo].update({
                'accuracy': accuracy_atual,
                'ultima_atualizacao': datetime.now()
            })
    
            return True
    
        except Exception as e:
            print(f"Erro ao atualizar modelo: {str(e)}")
            return False

    def treinar(self, dados_historicos):
        """Treina o modelo de ML"""
        try:
            print("\nIniciando treinamento do modelo ML...")

            if dados_historicos is None or dados_historicos.empty:
                print("Sem dados históricos para treino")
                return False
            
            print(f"Total de dados recebidos: {len(dados_historicos)} registros")

            # Para cada ativo
            for ativo in dados_historicos['ativo'].unique():
                try:
                    print(f"\nTreinando modelo para {ativo}")
                    dados_ativo = dados_historicos[dados_historicos['ativo'] == ativo].copy()

                    if len(dados_ativo) < 50:  # Reduzido de 100
                        print(f"Dados insuficientes para {ativo}: {len(dados_ativo)} registros")
                        continue

                    # Organiza os dados
                    dados_ativo = dados_ativo.sort_values('timestamp')

                    # Cria features
                    features = self.criar_features(dados_ativo)
                    if features is None:
                        print(f"Erro ao criar features para {ativo}")
                        continue

                    # Remove registros com dados faltantes
                    valid_idx = ~features.isnull().any(axis=1)
                    features = features[valid_idx]

                    # Cria labels
                    retornos_futuros = dados_ativo['Close'].pct_change().shift(-1)
                    labels = (retornos_futuros > 0).astype(int)
                    labels = labels[valid_idx]

                    if len(features) < 50:
                        print(f"Features insuficientes para {ativo}")
                        continue
                    
                    # Divide dados
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, labels, test_size=0.2, random_state=42
                    )

                    # Cria scaler
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Treina modelo
                    params = {
                        'max_depth': 3,
                        'learning_rate': 0.1,
                        'n_estimators': 100,
                        'subsample': 0.8,
                        'objective': 'binary:logistic',
                        'random_state': 42
                    }

                    model = XGBClassifier(**params)
                    model.fit(X_train_scaled, y_train)

                    # Avalia e salva sempre
                    y_pred = model.predict(X_test_scaled)
                    accuracy = (y_pred == y_test).mean()

                    print(f"Modelo treinado para {ativo} - Acurácia: {accuracy:.2%}")

                    # Salva modelo
                    self.models[ativo] = {
                        'model': model,
                        'features': features.columns.tolist(),
                        'accuracy': accuracy,
                        'params': params
                    }
                    self.scalers[ativo] = scaler

                except Exception as e:
                    print(f"Erro ao treinar modelo para {ativo}: {str(e)}")
                    continue
                
            return len(self.models) > 0

        except Exception as e:
            print(f"Erro durante o treinamento: {str(e)}")
            return False

    def prever(self, dados, ativo):
        """Faz previsões para um ativo específico"""
        try:
            if ativo not in self.models:
                print(f"Modelo não encontrado para {ativo}")
                return None

            # Prepara features
            features = self.criar_features(dados)
            if features is None or features.empty:
                print(f"Erro ao criar features para {ativo}")
                return None

            try:
                # Seleciona apenas as features usadas no treino
                features = features[self.models[ativo]['features']]
            except KeyError as e:
                print(f"Erro nas features do modelo para {ativo}: {str(e)}")
                return None

            try:
                # Normaliza
                features_scaled = self.scalers[ativo].transform(features)
            except Exception as e:
                print(f"Erro ao normalizar features para {ativo}: {str(e)}")
                return None

            try:
                # Faz previsão
                model = self.models[ativo]['model']
                probabilidades = model.predict_proba(features_scaled)

                # Retorna última previsão
                ultima_prob = probabilidades[-1]

                if max(ultima_prob) >= self.min_probabilidade:
                    direcao = np.argmax(ultima_prob)
                    return {
                        'direcao': 'CALL' if direcao == 1 else 'PUT',
                        'probabilidade': float(max(ultima_prob)),
                        'score': float(max(ultima_prob)) * self.models[ativo]['accuracy']
                    }
                else:
                    return {
                        'direcao': 'NEUTRO',
                        'probabilidade': float(max(ultima_prob)),
                        'score': 0
                    }

            except Exception as e:
                print(f"Erro ao fazer previsão para {ativo}: {str(e)}")
                return None

        except Exception as e:
            print(f"Erro na previsão para {ativo}: {str(e)}")
            return None

    def analisar(self, ativo, periodo='7d', intervalo='5m'):
        """Realiza análise completa de um ativo"""
        try:
            # Baixa dados mais recentes
            df = yf.download(ativo, period=periodo, interval=intervalo, progress=False)
            if df.empty:
                return None
            
            # Faz previsão
            previsao = self.prever(df, ativo)
            if previsao is None:
                return None
            
            # Adiciona informações adicionais
            previsao.update({
                'ativo': ativo,
                'timestamp': datetime.now(),
                'preco_atual': float(df['Close'].iloc[-1]),
                'volume': float(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0,
                'volatilidade': float(df['Close'].pct_change().std())
            })
            
            return previsao
            
        except Exception as e:
            print(f"Erro na análise de {ativo}: {str(e)}")
            return None