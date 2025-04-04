class Dame(Pion):
    def __init__(self, couleur):
        """
        Initialise une dame.
        :param couleur: Couleur de la dame ('noir' ou 'blanc').
        """
        super().__init__(couleur, est_dame=True)