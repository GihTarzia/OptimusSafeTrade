import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import pandas as pd

@dataclass
class Operacao:
    ativo: str
    tipo: str  # 'CALL' ou 'PUT'
    entrada: float
    saida: float
    resultado: float
    timestamp: datetime
    score_entrada: float
    assertividade_prevista: float

class GestaoRiscoAdaptativo:
    def __init__(self, saldo_inicial: float, risco_inicial: float = 0.02):
        self.saldo_inicial = saldo_inicial
        self.saldo_atual = saldo_inicial
        self.risco_inicial = risco_inicial
        self.operacoes = []
        self.meta_diaria = 0.05  # 5% ao dia
        self.stop_diario = -0.1  # -10% ao dia
        self.martingale_ativo = False
        self.multiplicador_martingale = 2.0
        
        # Configurações adaptativas
        self.win_rate_minimo = 0.55
        self.fator_kelly = 0.5  # Fração de Kelly conservadora
        self.drawdown_maximo = 0.15  # 15% de drawdown máximo
        
        # Métricas de desempenho
        self.metricas = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'drawdown_atual': 0.0,
            'drawdown_maximo': 0.0,
            'resultado_dia': 0.0,
            'sequencia_atual': 0,
            'maior_sequencia_loss': 0
        }
        
        # Controle de horários
        self.horarios_ruins = set()
        self.melhores_horarios = {}

    def calcular_risco_operacao(self, ativo: str, score_entrada: float, 
                              assertividade: float) -> Dict:
        """Calcula o risco ideal para a próxima operação"""
        try:
            # Valor base de risco
            risco_base = self.saldo_atual * self.risco_inicial
            
            # Ajustes baseados em desempenho
            if self.metricas['win_rate'] < self.win_rate_minimo:
                risco_base *= 0.5
            
            # Ajuste pelo Score de entrada
            multiplicador_score = min(max(score_entrada, 0.5), 1.5)
            risco_base *= multiplicador_score
            
            # Ajuste pela assertividade prevista
            multiplicador_assertividade = assertividade / 100  # Converte para decimal
            risco_base *= multiplicador_assertividade
            
            # Ajuste por drawdown
            if self.metricas['drawdown_atual'] > self.drawdown_maximo * 0.5:
                risco_base *= 0.5
            
            # Ajuste por horário
            hora_atual = datetime.now().hour
            if hora_atual in self.horarios_ruins:
                risco_base *= 0.25
            
            # Ajuste por sequência de perdas
            if self.metricas['sequencia_atual'] < 0:
                risco_base *= 0.7 ** abs(self.metricas['sequencia_atual'])
            
            # Cálculo final usando Fração de Kelly
            win_rate = self.metricas['win_rate'] or 0.5
            kelly = win_rate - (1 - win_rate)
            risco_kelly = kelly * self.fator_kelly * risco_base
            
            # Limites de segurança
            risco_maximo = self.saldo_atual * 0.05  # Máximo 5% do saldo
            risco_final = min(max(risco_kelly, 1), risco_maximo)
            
            return {
                'valor_risco': round(risco_final, 2),
                'percentual_saldo': round((risco_final / self.saldo_atual) * 100, 2),
                'stop_loss': round(risco_final * 1.5, 2),
                'take_profit': round(risco_final * 2, 2)
            }
            
        except Exception as e:
            print(f"Erro ao calcular risco: {str(e)}")
            return None

    def registrar_operacao(self, operacao: Operacao) -> None:
        """Registra uma operação e atualiza métricas"""
        try:
            self.operacoes.append(operacao)
            self.saldo_atual += operacao.resultado
            
            # Atualiza métricas
            self._atualizar_metricas()
            
            # Atualiza horários
            self._atualizar_analise_horarios()
            
        except Exception as e:
            print(f"Erro ao registrar operação: {str(e)}")

    def _atualizar_metricas(self) -> None:
        """Atualiza todas as métricas de desempenho"""
        if not self.operacoes:
            return
            
        # Últimas operações (últimas 50)
        ops_recentes = self.operacoes[-50:]
        
        # Win Rate
        wins = len([op for op in ops_recentes if op.resultado > 0])
        self.metricas['win_rate'] = wins / len(ops_recentes)
        
        # Profit Factor
        ganhos = sum([op.resultado for op in ops_recentes if op.resultado > 0])
        perdas = abs(sum([op.resultado for op in ops_recentes if op.resultado < 0]))
        self.metricas['profit_factor'] = ganhos / perdas if perdas > 0 else float('inf')
        
        # Drawdown
        saldo_maximo = self.saldo_inicial
        drawdown_atual = 0
        
        for op in self.operacoes:
            saldo_apos_op = saldo_maximo + op.resultado
            if saldo_apos_op > saldo_maximo:
                saldo_maximo = saldo_apos_op
            else:
                drawdown = (saldo_maximo - saldo_apos_op) / saldo_maximo
                drawdown_atual = max(drawdown_atual, drawdown)
                
        self.metricas['drawdown_atual'] = drawdown_atual
        self.metricas['drawdown_maximo'] = max(
            self.metricas['drawdown_maximo'],
            drawdown_atual
        )
        
        # Sequência atual
        sequencia = 0
        for op in reversed(self.operacoes):
            if (sequencia >= 0 and op.resultado > 0) or (sequencia <= 0 and op.resultado < 0):
                sequencia = sequencia + 1 if op.resultado > 0 else sequencia - 1
            else:
                break
        self.metricas['sequencia_atual'] = sequencia

    def _atualizar_analise_horarios(self) -> None:
        """Analisa e atualiza horários bons e ruins"""
        df_ops = pd.DataFrame([{
            'hora': op.timestamp.hour,
            'resultado': op.resultado,
            'assertividade': op.assertividade_prevista
        } for op in self.operacoes])
        
        if not df_ops.empty:
            # Análise por hora
            analise_hora = df_ops.groupby('hora').agg({
                'resultado': ['count', 'mean', 'sum'],
                'assertividade': 'mean'
            })
            
            # Identifica horários ruins
            horarios_ruins = analise_hora[
                (analise_hora['resultado']['mean'] < 0) & 
                (analise_hora['resultado']['count'] >= 5)
            ].index
            
            self.horarios_ruins = set(horarios_ruins)
            
            # Identifica melhores horários
            melhores = analise_hora[
                (analise_hora['resultado']['mean'] > 0) & 
                (analise_hora['resultado']['count'] >= 5)
            ].sort_values(('resultado', 'mean'), ascending=False)
            
            self.melhores_horarios = {
                hora: {
                    'win_rate': float(stats['resultado']['mean']),
                    'total_ops': int(stats['resultado']['count'])
                }
                for hora, stats in melhores.iterrows()
            }

    def verificar_stop_diario(self) -> bool:
        """Verifica se atingiu stop diário"""
        ops_hoje = [op for op in self.operacoes 
                   if op.timestamp.date() == datetime.now().date()]
        
        if not ops_hoje:
            return False
            
        resultado_dia = sum(op.resultado for op in ops_hoje)
        resultado_percentual = resultado_dia / self.saldo_inicial
        
        return resultado_percentual <= self.stop_diario

    def verificar_meta_diaria(self) -> bool:
        """Verifica se atingiu meta diária"""
        ops_hoje = [op for op in self.operacoes 
                   if op.timestamp.date() == datetime.now().date()]
        
        if not ops_hoje:
            return False
            
        resultado_dia = sum(op.resultado for op in ops_hoje)
        resultado_percentual = resultado_dia / self.saldo_inicial
        
        return resultado_percentual >= self.meta_diaria

    def get_estatisticas(self) -> Dict:
        """Retorna estatísticas completas"""
        return {
            'saldo_inicial': self.saldo_inicial,
            'saldo_atual': self.saldo_atual,
            'lucro_total': self.saldo_atual - self.saldo_inicial,
            'lucro_percentual': ((self.saldo_atual / self.saldo_inicial) - 1) * 100,
            'total_operacoes': len(self.operacoes),
            'metricas': self.metricas,
            'horarios_ruins': list(self.horarios_ruins),
            'melhores_horarios': self.melhores_horarios
        }