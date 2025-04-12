import random
import math
import copy
from Mouvement import Mouvement

class Noeud:
    def __init__(self, etat=None, parent=None, mouvement=None):
        """
        Initialise un nœud de l'arbre MCTS.
        :param etat: Instance du jeu (plateau, joueurs, etc.)
        :param parent: Nœud parent
        :param mouvement: Mouvement qui a mené à cet état
        """
        self.etat = etat
        self.parent = parent
        self.mouvement = mouvement
        self.enfants = []
        self.visites = 0
        self.victoires = 0
        self.mouvements_non_explores = None

    def est_entierement_explore(self):
        """
        Vérifie si tous les mouvements possibles ont été explorés.
        """
        return self.mouvements_non_explores is not None and len(self.mouvements_non_explores) == 0

    def est_terminal(self):
        """
        Vérifie si le nœud est un état terminal (fin de partie).
        """
        est_termine, _ = self.etat.plateau.partie_terminee()
        return est_termine

    def UCT(self, constante_exploration):
        """
        Calcule la valeur UCT du nœud (Upper Confidence Bound 1).
        :param constante_exploration: Constante pour ajuster l'exploration.
        :return: Valeur UCT
        """
        if self.visites == 0:
            return float('inf')
        return (self.victoires / self.visites) + constante_exploration * math.sqrt(math.log(self.parent.visites) / self.visites)
