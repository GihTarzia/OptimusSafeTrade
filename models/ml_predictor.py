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
    def __init__(self, logger):
        self.model = None
        self.logger = logger
        self.scaler = StandardScaler()
        self.min_probabilidade = 0.65  # Probabilidade mínima para considerar um sinal
        self.janela_analise = 20  # Períodos para análise
        self.features_importance = {}

    def criar_features(self, df):
        """Cria features para o modelo de ML"""
        try:
            features = pd.DataFrame()
            
            print("Calculando indicadores técnicos...")
            
            # RSI
            features['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            features['bb_high'] = bollinger.bollinger_hband()
            features['bb_low'] = bollinger.bollinger_lband()
            features['bb_position'] = bollinger.bollinger_pband()
            
            # Estocástico
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            features['stoch_k'] = stoch.stoch()
            features['stoch_d'] = stoch.stoch_signal()
            
            # ADX
            adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
            features['adx'] = adx.adx()
            
            # Volatilidade
            atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'])
            features['volatility'] = atr.average_true_range()
            
            # Momentum
            features['momentum'] = ta.momentum.ROCIndicator(df['Close']).roc()
            
            # Volume
            if 'Volume' in df.columns and df['Volume'].sum() > 0:
                features['volume_change'] = df['Volume'].pct_change()
            
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
 
    def treinar(self, dados_historicos):
        """Treina o modelo de ML"""
        try:
            print("\nIniciando treinamento do modelo ML...")

            if dados_historicos.empty:
                print("Sem dados históricos suficientes para treino")
                return False

            print(f"Total de dados recebidos: {len(dados_historicos)} registros")

            # Dicionário para armazenar modelos por ativo
            self.models = {}

            # Treina um modelo para cada ativo
            for ativo in dados_historicos['ativo'].unique():
                print(f"\n{Fore.CYAN}Treinando modelo para {ativo}{Style.RESET_ALL}")
                dados_ativo = dados_historicos[dados_historicos['ativo'] == ativo].copy()

                if len(dados_ativo) < 100:
                    print(f"Dados insuficientes para {ativo} ({len(dados_ativo)} registros)")
                    continue

                # Organiza os dados por timestamp
                dados_ativo = dados_ativo.sort_values('timestamp')

                # Cria features
                print("Criando features...")
                features = self.criar_features(dados_ativo)
                if features is None:
                    print(f"Erro ao criar features para {ativo}")
                    continue

                # Remove registros com dados faltantes
                valid_idx = ~features.isnull().any(axis=1)
                features = features[valid_idx]

                # Cria labels (1 para alta, 0 para baixa)
                retornos_futuros = dados_ativo['Close'].pct_change().shift(-1)
                labels = (retornos_futuros > 0).astype(int)
                labels = labels[valid_idx]

                print(f"Features criadas: {len(features)} registros")

                # Divide dados em treino e teste
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.2, random_state=42
                )

                # Normaliza features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)

                print("Treinando modelo...")
                # Treina modelo
                model = XGBClassifier(
                    max_depth=5,
                    learning_rate=0.05,
                    n_estimators=200,
                    objective='binary:logistic',
                    random_state=42
                )

                model.fit(
                    X_train_scaled,
                    y_train,
                    eval_set=[(X_test_scaled, y_test)],
                    eval_metric='logloss',
                    early_stopping_rounds=20,
                    verbose=False
                )

                # Avalia modelo
                y_pred = model.predict(X_test_scaled)
                accuracy = (y_pred == y_test).mean()

                print(f"{Fore.GREEN}Modelo treinado para {ativo}")
                print(f"Acurácia: {accuracy:.2%}{Style.RESET_ALL}")

                # Salva modelo
                self.models[ativo] = {
                    'model': model,
                    'scaler': self.scaler,
                    'accuracy': accuracy,
                    'features': features.columns.tolist()
                }

            print(f"\n{Fore.GREEN}Treinamento concluído!")
            print(f"Modelos treinados: {len(self.models)}{Style.RESET_ALL}")
            return True

        except Exception as e:
            print(f"{Fore.RED}Erro durante o treinamento: {str(e)}{Style.RESET_ALL}")
            return False

    def prever(self, dados):
        """Faz previsões com o modelo treinado"""
        try:
            if self.model is None:
                return None
                
            # Prepara features
            features = self.criar_features(dados)
            if features is None or features.empty:
                return None
                
            # Normaliza
            features_scaled = self.scaler.transform(features)
            
            # Faz previsão
            probabilidades = self.model.predict_proba(features_scaled)
            
            # Retorna última previsão
            ultima_prob = probabilidades[-1]
            
            # Determina direção apenas se probabilidade for alta o suficiente
            if max(ultima_prob) >= self.min_probabilidade:
                direcao = np.argmax(ultima_prob)
                return {
                    'direcao': 'CALL' if direcao == 1 else 'PUT',
                    'probabilidade': float(max(ultima_prob)),
                    'confianca': self.calcular_confianca(features.iloc[-1])
                }
            else:
                return {'direcao': 'NEUTRO', 'probabilidade': 0, 'confianca': 0}
                
        except Exception as e:
            print(f"Erro na previsão: {str(e)}")
            return None

    def calcular_confianca(self, features_atuais):
        """Calcula nível de confiança baseado na importância das features"""
        confianca = 0
        for feature, valor in features_atuais.items():
            if feature in self.features_importance:
                # Normaliza o valor da feature e multiplica pela sua importância
                valor_norm = abs(float(valor)) / 100  # Normalização simples
                confianca += valor_norm * self.features_importance[feature]
        return min(float(confianca), 1.0)

    def analisar(self, ativo, periodo='1d', intervalo='5m'):
        """Realiza análise completa de um ativo"""
        try:
            # Baixa dados mais recentes
            df = yf.download(ativo, period=periodo, interval=intervalo, progress=False)
            if df.empty:
                return None
            
            # Faz previsão
            previsao = self.prever(df)
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
            print(f"Erro na análise: {str(e)}")
            return None