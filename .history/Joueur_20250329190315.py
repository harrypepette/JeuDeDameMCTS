class Joueur:
    def __init__(self, nom, couleur):
        """
        Initialise un joueur.
        :param nom: Nom du joueur.
        :param couleur: Couleur des pions du joueur ('noir' ou 'blanc').
        """
        self.nom = nom
        self.couleur = couleur
        self.pions = []  # Liste des pions du joueur

    def ajouter_pion(self, pion):
        """Ajoute un pion à la liste des pions du joueur."""
        self.pions.append(pion)

    def retirer_pion(self, pion):
        """Retire un pion de la liste des pions du joueur."""
        self.pions.remove(pion)