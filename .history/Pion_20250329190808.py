class Pion:
    """
    Classe représentant un pion dans le jeu de dames.
    """

    def __init__(self, couleur, est_dame=False):
        """
        Initialise un pion.
        :param couleur: Couleur du pion ('noir' ou 'blanc').
        :param est_dame: Indique si le pion est une dame (par défaut False).
        """
        self.couleur = couleur
        self.est_dame = est_dame

    def promouvoir(self):
        """
        Transforme le pion en dame.
        """
        self.est_dame = True

    def __str__(self):
        """
        Retourne une représentation textuelle du pion.
        """
        return "D" if self.est_dame else "P"