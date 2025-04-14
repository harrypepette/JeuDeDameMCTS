import random
from Plateau import Plateau
from Joueur import Joueur
from Mouvement import Mouvement  # Import the Mouvement class or function
from Pion import Pion  # Import the Pion class or function
class Instance:
    def __init__(self, positions_initiales):
        """
        Initialise une instance du jeu avec des positions spécifiques.
        :param positions_initiales: Liste des positions des pions (format : [(couleur, (x, y)), ...]).
        """
        self.plateau = Plateau()
        self.plateau.cases = [[None for _ in range(8)] for _ in range(8)]  # Réinitialiser le plateau
        for couleur, position in positions_initiales:
            x, y = position
            if couleur == "blanc":
                self.plateau.cases[y][x] = Pion("blanc", position)
            elif couleur == "noir":
                self.plateau.cases[y][x] = Pion("noir", position)
        self.joueurs = [Joueur("blanc"), Joueur("noir")]
        self.joueur_actuel = 0

    def run_to_the_end(self):
        """
        Lance le jeu avec des mouvements aléatoires jusqu'à ce qu'un joueur gagne.
        """
        while True:
            joueur_couleur = self.joueurs[self.joueur_actuel].couleur
            pions = [
                pion for ligne in self.plateau.cases for pion in ligne
                if pion and pion.couleur == joueur_couleur
            ]

            # Trouver tous les mouvements possibles pour les pions du joueur actuel
            mouvements_possibles = []
            for pion in pions:
                mouvements = self.plateau.mouvements_possibles(pion, forcer_manger=True)
                if not mouvements:
                    mouvements = self.plateau.mouvements_possibles(pion, forcer_manger=False)
                for mouvement in mouvements:
                    mouvements_possibles.append((pion, mouvement))

            if not mouvements_possibles:
                # Si aucun mouvement n'est possible, l'autre joueur gagne
                gagnant = "noir" if joueur_couleur == "blanc" else "blanc"
                print(f"Le joueur {gagnant} a gagné !")
                break

            # Choisir un mouvement aléatoire
            pion, mouvement = random.choice(mouvements_possibles)
            self.plateau.deplacer_pion(Mouvement(pion.position, mouvement))

            # Vérifier si la partie est terminée
            est_terminee, gagnant = self.plateau.partie_terminee()
            if est_terminee:
                print(f"Le joueur {gagnant} a gagné !")
                break

            # Passer au joueur suivant
            self.joueur_actuel = 1 - self.joueur_actuel