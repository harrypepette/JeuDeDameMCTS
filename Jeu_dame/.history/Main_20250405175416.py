from Jeu import Jeu
from ChatInterface import ChatInterface
import sys
from threading import Thread

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

class GameManager:
    def __init__(self):
        self.game_thread = None
        self.jeu = None

    def start_game(self):
        """
        Démarre une nouvelle instance du jeu.
        """
        self.jeu = Jeu()
        self.game_thread = Thread(target=self.jeu.run)
        self.game_thread.start()

    def restart_game(self):
        """
        Redémarre le jeu.
        """
        if self.jeu:
            self.jeu.stop()  # Ajoutez une méthode stop() dans la classe Jeu pour arrêter proprement le jeu
        self.start_game()

def main():
    # Créer le gestionnaire de jeu
    game_manager = GameManager()

    # Créer l'interface de chat avec le callback pour redémarrer
    chat = ChatInterface(restart_callback=game_manager.restart_game)

    # Rediriger les messages du terminal vers le chat
    sys.stdout = TerminalToChat(chat)

    # Démarrer le jeu
    game_manager.start_game()

    # Lancer l'interface de chat
    chat.run()

if __name__ == "__main__":
    main()