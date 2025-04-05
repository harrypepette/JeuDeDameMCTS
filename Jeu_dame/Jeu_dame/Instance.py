from Jeu import Jeu
import random

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
        jeu.plateau.initialiser_positions(self.positions)  # Méthode à définir pour initialiser le plateau avec les positions
        while not jeu.partie_terminee:
            joueur = jeu.joueurs[jeu.joueur_actuel]
            mouvements_possibles = jeu.plateau.get_mouvements_possibles(joueur.couleur)  # Méthode à définir pour obtenir les mouvements possibles
            if mouvements_possibles:
                mouvement = random.choice(mouvements_possibles)  # Choisir un mouvement aléatoire
                jeu.effectuer_mouvement(mouvement)  # Méthode à définir pour effectuer le mouvement
            else:
                jeu.partie_terminee = True  # Si aucun mouvement n'est possible, la partie est terminée
        print(f"Partie terminée! Le joueur {joueur.couleur} a gagné.")  # Annonce du gagnant