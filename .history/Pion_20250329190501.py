class Pion:
    def __init__(self, couleur, est_dame=False):
        """
        Initialise un pion.
        :param couleur: Couleur du pion ('noir' ou 'blanc').
        :param est_dame: Indique si le pion est une dame.
        """
        self.couleur = couleur
        self.est_dame = est_dame

    def promouvoir(self):
        """Transforme le pion en dame."""
        self.est_dame = True