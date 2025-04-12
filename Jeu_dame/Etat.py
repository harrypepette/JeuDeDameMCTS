import copy
from Mouvement import Mouvement
from Plateau import Plateau
from Joueur import Joueur

class Etat:
    def __init__(self, plateau=None, joueurs=None, joueur_actuel=0):
        """
        Initialise un état du jeu de dames.
        :param plateau: Le plateau de jeu
        :param joueurs: Liste des joueurs
        :param joueur_actuel: Indice du joueur actuel
        """
        self.plateau = plateau if plateau else Plateau()
        self.joueurs = joueurs if joueurs else [Joueur("blanc"), Joueur("noir")]
        self.joueur_actuel = joueur_actuel
    
    @classmethod
    def depuis_jeu(cls, jeu):
        """
        Crée un état à partir d'une instance de jeu.
        :param jeu: Instance de la classe Jeu
        :return: Une nouvelle instance d'Etat
        """
       # Créer une copie du plateau sans les éléments Pygame
        plateau_copie = jeu.plateau.copier()
        joueurs_copie = jeu.joueurs[:]  # Copie superficielle de la liste
        return cls(
            plateau=plateau_copie,
            joueurs=joueurs_copie,
            joueur_actuel=jeu.joueur_actuel
    )
    def copier(self):
        """
        Crée une copie profonde de l'état.
        :return: Une nouvelle instance d'Etat
        """
        return Etat(
            plateau=self.plateau.copier(),  # À implémenter dans Plateau
            joueurs=self.joueurs,  # Pas besoin de copier si immuable
            joueur_actuel=self.joueur_actuel
        )
    
    def appliquer_mouvement(self, mouvement):
        """
        Applique un mouvement à l'état et retourne le nouvel état.
        :param mouvement: Le mouvement à appliquer
        :return: Un nouvel état après l'application du mouvement
        """
        nouvel_etat = self.copier()
        nouvel_etat.plateau.deplacer_pion(mouvement)
        
        # Vérifier si une capture supplémentaire est possible
        pion_apres_mouvement = nouvel_etat.plateau.get_pion(*mouvement.arrivee)
        if pion_apres_mouvement and nouvel_etat.plateau.peut_encore_manger(pion_apres_mouvement):
            # Ne pas changer de joueur, car le même joueur continue
            pass
        else:
            # Passer au joueur suivant
            nouvel_etat.joueur_actuel = 1 - nouvel_etat.joueur_actuel
        
        return nouvel_etat
    
    def est_terminal(self):
        """
        Vérifie si l'état est terminal (fin de partie).
        :return: Un tuple (est_terminee, gagnant)
        """
        return self.plateau.partie_terminee()
    
    def obtenir_mouvements_legaux(self):
        """
        Obtient tous les mouvements légaux pour le joueur actuel.
        :return: Liste des mouvements légaux [(pion, position_arrivee), ...]
        """
        joueur_couleur = self.joueurs[self.joueur_actuel].couleur
        pions = [
            pion for ligne in self.plateau.cases for pion in ligne
            if pion and pion.couleur == joueur_couleur
        ]
        
        mouvements_possibles = []
        # Vérifier si des captures sont possibles
        captures_possibles = False
        for pion in pions:
            cases_capture = self.plateau.cases_fin_manger(pion)
            if cases_capture:
                captures_possibles = True
                for case in cases_capture:
                    mouvements_possibles.append((pion, case))
        
        # Si aucune capture n'est possible, ajouter les mouvements simples
        if not captures_possibles:
            for pion in pions:
                mouvements = self.plateau.mouvements_possibles(pion, forcer_manger=False)
                for mouvement in mouvements:
                    mouvements_possibles.append((pion, mouvement))
        
        return mouvements_possibles