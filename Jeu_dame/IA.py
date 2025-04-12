from MCTS import MCTS


class IA:
    def __init__(self, couleur, iterations=1000, exploration=0.7):
        """
        Initialise l'IA avec l'algorithme MCTS.
        :param couleur: Couleur des pions de l'IA ('blanc' ou 'noir')
        :param iterations: Nombre d'itérations pour l'algorithme MCTS
        """
        self.couleur = couleur
        self.mcts = MCTS(iterations=iterations, constante_exploration=exploration)
    
    def choisir_mouvement(self, etat):
        """
        Choisit le meilleur mouvement pour l'état actuel du jeu.
        :param etat: Instance du jeu (plateau, joueurs, etc.)
        :return: Le meilleur mouvement trouvé (Mouvement)
        """
        return self.mcts.rechercher_meilleur_mouvement(etat)