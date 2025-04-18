from copy import deepcopy
import math

class Node:
    def __init__(self, plateau, mouvement=None, parent=None):
        # Assurez-vous que tous les membres sont picklables
        # Le plateau doit être copié sans références à pygame
        self.plateau = plateau  # État du plateau après le mouvement
        self.mouvement = mouvement  # Mouvement qui a conduit à cet état
        self.parent = parent  # Référence au nœud parent (None pour les nœuds créés en parallèle)
        self.enfants = []  # Liste des nœuds enfants
        self.visites = 0  # Nombre de visites du nœud
        self.victoires = 0  # Nombre de victoires simulées
        self.mouvements_non_explores = None  # Mouvements possibles non encore explorés

    def est_feuille(self):
        """
        Vérifie si le nœud est une feuille (pas d'enfants).
        """
        return len(self.enfants) == 0

    def est_completement_explore(self):
        """
        Vérifie si tous les mouvements possibles ont été explorés.
        """
        return len(self.mouvements_non_explores) == 0

    def calculer_uct(self, c=1.414):
        """
        Calcule la valeur UCT pour guider la sélection.
        :param c: Constante d'exploration (par défaut √2).
        """
        if self.visites == 0:
            return float('inf')  # Prioriser les nœuds non visités
        parent_visites = self.parent.visites if self.parent else 1
        exploitation = self.victoires / self.visites
        exploration = c * math.sqrt(math.log(parent_visites) / self.visites)
        return exploitation + exploration