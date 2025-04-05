from Jeu import Jeu
from ChatInterface import ChatInterface
import sys
import random

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

class Instance:
    def __init__(self, positions):
        """
        Initialise une instance du jeu avec une liste de positions de pièces.
        :param positions: Liste de tuples représentant les positions des pièces (x, y).
        """
        self.positions = positions

    def run_to_the_end(self):
        """
        Démarre le jeu à partir de la liste de positions, exécutant des mouvements aléatoires jusqu'à ce qu'un joueur gagne.
        """
        jeu = Jeu()
        jeu.plateau.initialiser_positions(self.positions)  # Méthode à définir pour initialiser le plateau
        while not jeu.partie_terminee:
            joueur = jeu.joueurs[jeu.joueur_actuel]
            mouvements_possibles = jeu.plateau.get_mouvements_possibles(joueur)  # Méthode à définir pour obtenir les mouvements possibles
            if mouvements_possibles:
                mouvement = random.choice(mouvements_possibles)
                jeu.effectuer_mouvement(mouvement)  # Méthode à définir pour effectuer un mouvement
            else:
                print(f"Le joueur {joueur.couleur} n'a plus de mouvements possibles.")
                jeu.partie_terminee = True

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