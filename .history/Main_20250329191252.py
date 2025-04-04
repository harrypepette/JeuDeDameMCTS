from Plateau import Plateau
from Joueur import Joueur
from Jeu import Jeu
from Interface import Interface

class ConsoleInterface(Interface):
    """
    Implémentation concrète de l'interface pour une interaction via la console.
    """

    def afficher_plateau(self, plateau):
        plateau.afficher()

    def demander_mouvement(self):
        try:
            case_depart = input("Entrez la case de départ (x, y) : ")
            case_arrivee = input("Entrez la case d'arrivée (x, y) : ")
            return tuple(map(int, case_depart.split(','))), tuple(map(int, case_arrivee.split(',')))
        except ValueError:
            print("Entrée invalide. Veuillez entrer des coordonnées sous la forme x, y.")
            return self.demander_mouvement()

    def afficher_message(self, message):
        print(message)

def main():
    """
    Point d'entrée principal pour lancer le jeu de dames.
    """
    # Initialisation des joueurs
    joueur1 = Joueur("Joueur 1", "noir")
    joueur2 = Joueur("Joueur 2", "blanc")

    # Initialisation du plateau
    plateau = Plateau()

    # Initialisation de l'interface
    interface = ConsoleInterface()

    # Initialisation du jeu
    jeu = Jeu(interface, plateau, [joueur1, joueur2])

    # Lancer le jeu
    jeu.lancer()

if __name__ == "__main__":
    main()