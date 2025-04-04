class Joueur:
    """
    Classe représentant un joueur dans le jeu de dames.
    """

    def __init__(self, nom, couleur):
        """
        Initialise un joueur.
        :param nom: Nom du joueur.
        :param couleur: Couleur des pions du joueur ('noir' ou 'blanc').
        """
        self.nom = nom
        self.couleur = couleur
        self.pions = []  # Liste des pions appartenant au joueur

    def ajouter_pion(self, pion):
        """
        Ajoute un pion à la liste des pions du joueur.
        :param pion: Instance de la classe Pion.
        """
        self.pions.append(pion)

    def retirer_pion(self, pion):
        """
        Retire un pion de la liste des pions du joueur.
        :param pion: Instance de la classe Pion.
        """
        if pion in self.pions:
            self.pions.remove(pion)

    def a_perdu(self):
        """
        Vérifie si le joueur a perdu (plus de pions).
        :return: True si le joueur a perdu, sinon False.
        """
        return len(self.pions) == 0

    def __str__(self):
        """
        Retourne une représentation textuelle du joueur.
        """
        return f"{self.nom} ({self.couleur})"