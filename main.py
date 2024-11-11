import os
import sys
import schedule
import time
from datetime import datetime, timedelta
from colorama import init, Fore, Style
from pathlib import Path
from tqdm import tqdm
import yfinance as yf

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
        
        # Inicializa componentes principais com barra de progresso
        componentes = [
            ("Logger", self.logger),
            ("Banco de Dados", self.db),
            ("Configurações", self.config),
            ("ML Predictor", MLPredictor(self.logger)),
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

    def calcular_timing_entrada(self, ativo, sinal):
        """Calcula o melhor momento para entrada baseado nos padrões históricos"""
        agora = datetime.now()
        dados_mercado = self.db.get_dados_mercado(ativo, limit=1000)
        
        # Analisa padrões de tempo mais favoráveis
        horarios_sucesso = self.db.get_horarios_sucesso(ativo)
        
        # Verifica se estamos próximos de um horário de sucesso
        melhor_horario = None
        maior_taxa_sucesso = 0
        
        for horario, taxa in horarios_sucesso.items():
            horario_dt = datetime.strptime(horario, "%H:%M").time()
            if taxa > maior_taxa_sucesso:
                maior_taxa_sucesso = taxa
                melhor_horario = horario_dt
        
        # Calcula tempo de espera ideal
        if melhor_horario:
            horario_atual = agora.time()
            if horario_atual < melhor_horario:
                tempo_espera = datetime.combine(agora.date(), melhor_horario) - datetime.combine(agora.date(), horario_atual)
            else:
                tempo_espera = timedelta(minutes=1)
        else:
            tempo_espera = timedelta(minutes=1)
        
        momento_entrada = agora + tempo_espera
        
        return {
            'momento_ideal': momento_entrada,
            'tempo_espera': tempo_espera,
            'taxa_sucesso_horario': maior_taxa_sucesso if maior_taxa_sucesso else 0
        }

    def calcular_assertividade(self, ativo, sinal):
        """Calcula a probabilidade de sucesso do sinal baseado em múltiplos fatores"""
        # Previsão do modelo ML
        prob_ml = self.ml_predictor.prever(self.db.get_dados_recentes(ativo))
        
        # Força dos padrões técnicos
        forca_padroes = self.analise_padroes.calcular_forca_sinais(sinal['sinais'])
        
        # Histórico recente
        assertividade_historica = self.db.get_assertividade_recente(ativo, sinal['direcao'])
        
        # Condições de mercado
        volatilidade_favoravel = sinal['volatilidade'] > 0.001 and sinal['volatilidade'] < 0.005
        
        # Combina todos os fatores
        peso_ml = 0.4
        peso_padroes = 0.3
        peso_historico = 0.2
        peso_mercado = 0.1
        
        assertividade = (
            prob_ml['probabilidade'] * peso_ml +
            forca_padroes * peso_padroes +
            assertividade_historica * peso_historico +
            (1 if volatilidade_favoravel else 0) * peso_mercado
        )
        
        return round(assertividade * 100, 2)  # Retorna porcentagem

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
                    print("Colunas disponíveis:", dados.columns.tolist())
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

    def analisar_mercado(self):
        """Análise principal do mercado"""
        self.logger.info(f"\n{Fore.CYAN}=== Nova Análise ==={Style.RESET_ALL}")
        self.logger.info(f"Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        
        ativos = self.config.get_ativos_ativos()
        for ativo in tqdm(ativos, desc="Analisando ativos"):
            try:
                # Análise técnica e ML
                sinal = self.ml_predictor.analisar(ativo)
                padroes = self.analise_padroes.identificar_padroes(
                    self.db.get_dados_mercado(ativo, limite=100)
                )
                
                if sinal and padroes:
                    # Combina sinais
                    sinal['sinais'] = [p.nome for p in padroes]
                    sinal['forca_sinal'] = self.analise_padroes.calcular_forca_sinais(padroes)
                    
                    if sinal['forca_sinal'] >= self.config.get('analise.min_score_entrada'):
                        # Calcula timing e assertividade
                        timing = self.calcular_timing_entrada(ativo, sinal)
                        assertividade = self.calcular_assertividade(ativo, sinal)
                        
                        # Calcula risco
                        risco = self.gestao_risco.calcular_risco_operacao(
                            ativo,
                            sinal['forca_sinal'],
                            assertividade
                        )
                        
                        if risco:
                            # Formata saída
                            self.exibir_resultado(ativo, sinal, timing, assertividade, risco)
                            
                            # Registra sinal
                            self.registrar_sinal(ativo, sinal, timing, assertividade, risco)
                
            except Exception as e:
                self.logger.error(f"Erro ao analisar {ativo}: {str(e)}")

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

    def registrar_sinal(self, ativo, sinal, timing, assertividade, risco):
        """Registra o sinal no banco de dados"""
        self.db.registrar_sinal(
            ativo=ativo,
            direcao=sinal['direcao'],
            momento_entrada=timing['momento_ideal'],
            tempo_expiracao=self.config.get('trading.tempo_expiracao_padrao'),
            score=sinal['forca_sinal'],
            assertividade=assertividade,
            padroes=sinal['sinais'],
            indicadores=sinal.get('indicadores', {}),
            ml_prob=sinal.get('probabilidade', 0),
            volatilidade=sinal.get('volatilidade', 0)
        )

    def executar(self):
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
            
            # Faz primeira análise imediatamente
            self.analisar_mercado()
            
            # Agenda análises regulares
            schedule.every(1).minutes.do(self.analisar_mercado)
            
            print(f"\n{Fore.GREEN}Sistema em execução contínua. Pressione Ctrl+C para encerrar.{Style.RESET_ALL}")
            print(f"Próxima análise em 1 minuto...")
            
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
    sistema.executar()