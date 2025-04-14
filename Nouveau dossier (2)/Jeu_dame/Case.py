class Case:
    def __init__(self, x, y, couleur):
        """
        Initialise une case.
        :param x: Coordonnée x de la case.
        :param y: Coordonnée y de la case.
        :param couleur: Couleur de la case ('blanc' ou 'noir').
        """
        self.x = x
        self.y = y
        self.couleur = couleur
        self.pion = None
