import asyncio
from telegram import Bot
from typing import Dict, List
from datetime import datetime
from colorama import Fore, Style

class Notificador:
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token
        self.chat_id = chat_id
        self.bot = Bot(token) if token else None
        self.mensagens_enviadas = []

    async def enviar_mensagem(self, mensagem: str) -> bool:
        """Envia mensagem para o Telegram"""
        try:
            if not self.bot or not self.chat_id:
                print(f"{Fore.YELLOW}Telegram não configurado{Style.RESET_ALL}")
                return False

            await self.bot.send_message(
                chat_id=self.chat_id,
                text=mensagem,
                parse_mode='Markdown'
            )
            
            self.mensagens_enviadas.append({
                'timestamp': datetime.now(),
                'mensagem': mensagem
            })
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Erro ao enviar mensagem Telegram: {str(e)}{Style.RESET_ALL}")
            return False

    def formatar_sinal(self, sinal: Dict) -> str:
        """Formata sinal para mensagem do Telegram"""
        try:
            s = sinal['sinal']
            timing = sinal['timing']
            
            # Emojis para direção
            emoji_direcao = "🟢" if s['direcao'] == 'CALL' else "🔴"
            
            mensagem = [
                f"{emoji_direcao} *SINAL {s['direcao']}*",
                f"",
                f"🎯 *Ativo:* {sinal['ativo']}",
                f"⏰ *Entrada:* {timing['momento_ideal'].strftime('%H:%M:%S')}",
                f"⌛️ *Expiração:* {s['tempo_expiracao']} minutos",
                f"",
                f"📊 *Qualidade do Sinal:*",
                f"• Score: {sinal['score_final']:.2%}",
                f"• Assertividade: {sinal['assertividade']:.1f}%",
                f"• Tendência: {s['tendencia']}",
                f"",
                f"💰 *Gestão:*",
                f"• Valor: ${sinal['risco']['valor_risco']:.2f}",
                f"• Stop Loss: ${sinal['risco']['stop_loss']:.2f}",
                f"• Take Profit: ${sinal['risco']['take_profit']:.2f}",
                f"",
                f"⚠️ *Padrões Detectados:*"
            ]
            
            # Adiciona padrões detectados
            for padrao in s['sinais'][:3]:  # Limita a 3 padrões
                mensagem.append(f"• {padrao['nome']}")
            
            mensagem.extend([
                f"",
                f"🔄 *Acompanhamento:*",
                f"• Volatilidade: {s['volatilidade']:.2%}",
                f"• Taxa Sucesso Horário: {timing['taxa_sucesso_horario']:.1%}"
            ])
            
            return "\n".join(mensagem)
            
        except Exception as e:
            print(f"Erro ao formatar sinal: {str(e)}")
            return ""

    def formatar_resultado(self, operacao: Dict) -> str:
        """Formata resultado de operação para Telegram"""
        try:
            emoji = "✅" if operacao['resultado'] == 'WIN' else "❌"
            
            mensagem = [
                f"{emoji} *RESULTADO {operacao['ativo']}*",
                f"",
                f"📈 *Operação:* {operacao['direcao']}",
                f"💵 *Resultado:* {'Gain' if operacao['resultado'] == 'WIN' else 'Loss'}",
                f"💰 *Lucro:* ${operacao['lucro']:.2f}",
                f"",
                f"📊 *Estatísticas do Dia:*",
                f"• Win Rate: {operacao['win_rate_dia']:.1f}%",
                f"• Resultado: {operacao['resultado_dia']:+.2f}%"
            ]
            
            return "\n".join(mensagem)
            
        except Exception as e:
            print(f"Erro ao formatar resultado: {str(e)}")
            return ""