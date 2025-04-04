from Pion import Pion

class Dame(Pion):
    """
    Classe représentant une dame dans le jeu de dames.
    Hérite de la classe Pion.
    """

    def __init__(self, couleur):
        """
        Initialise une dame.
        :param couleur: Couleur de la dame ('noir' ou 'blanc').
        """
        super().__init__(couleur, est_dame=True)

    def __str__(self):
        """
        Retourne une représentation textuelle de la dame.
        """
        return "D"