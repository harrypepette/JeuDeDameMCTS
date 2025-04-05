from Jeu import Jeu
from ChatInterface import ChatInterface
import sys

class TerminalToChat:
    """
    Redirige les messages du terminal vers l'interface de chat.
    """
    def __init__(self, chat_interface):
        self.chat_interface = chat_interface

    def write(self, message):
        if message.strip():  # Ignorer les lignes vides
            self.chat_interface.add_message(message)

    def flush(self):
        pass  # Nécessaire pour rediriger correctement les flux

def main():
    # Créer l'interface de chat
    chat = ChatInterface()

    # Rediriger les messages du terminal vers le chat
    sys.stdout = TerminalToChat(chat)

    # Lancer le jeu dans un thread séparé pour ne pas bloquer l'interface
    from threading import Thread
    def run_game():
        jeu = Jeu()
        jeu.run()

    game_thread = Thread(target=run_game)
    game_thread.start()

    # Lancer l'interface de chat
    chat.run()

if __name__ == "__main__":
    main()