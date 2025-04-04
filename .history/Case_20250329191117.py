class Case:
    """
    Classe représentant une case du plateau de jeu de dames.
    """

    def __init__(self, x, y, couleur):
        """
        Initialise une case du plateau.
        :param x: Coordonnée x de la case.
        :param y: Coordonnée y de la case.
        :param couleur: Couleur de la case ('noir' ou 'blanc').
        """
        self.x = x
        self.y = y
        self.couleur = couleur
        self.pion = None  # Contient un objet Pion ou None si la case est vide

    def est_vide(self):
        """
        Retourne True si la case est vide, sinon False.
        """
        return self.pion is None

    def placer_pion(self, pion):
        """
        Place un pion sur la case.
        :param pion: Instance de la classe Pion à placer sur la case.
        """
        self.pion = pion

    def retirer_pion(self):
        """
        Retire le pion de la case.
        """
        self.pion = None

    def __str__(self):
        """
        Retourne une représentation textuelle de la case.
        """
        if self.pion:
            return str(self.pion)
        return " " if self.couleur == "blanc" else "■"