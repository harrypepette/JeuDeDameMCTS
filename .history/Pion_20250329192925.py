class Pion:
    def __init__(self, couleur, position):
        """
        Initialise un pion.
        :param couleur: La couleur du pion ('blanc' ou 'noir').
        :param position: La position initiale du pion (tuple (x, y)).
        """
        self.couleur = couleur
        self.position = position
        self.est_dame = False

    def promouvoir(self):
        """
        Promeut le pion en dame.
        """
        self.est_dame = True
