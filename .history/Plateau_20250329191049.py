from Case import Case

class Plateau:
    """
    Classe représentant le plateau de jeu de dames.
    """

    def __init__(self, taille=8):
        """
        Initialise le plateau de jeu.
        :param taille: Taille du plateau (par défaut 8x8).
        """
        self.taille = taille
        self.cases = [[Case(x, y, 'noir' if (x + y) % 2 == 1 else 'blanc') for y in range(taille)] for x in range(taille)]

    def get_case(self, x, y):
        """
        Retourne la case aux coordonnées spécifiées.
        :param x: Coordonnée x de la case.
        :param y: Coordonnée y de la case.
        :return: Instance de la classe Case.
        :raises IndexError: Si les coordonnées sont hors du plateau.
        """
        if 0 <= x < self.taille and 0 <= y < self.taille:
            return self.cases[x][y]
        else:
            raise IndexError("Coordonnées hors du plateau.")

    def afficher(self):
        """
        Affiche le plateau dans la console.
        """
        for ligne in self.cases:
            print(" ".join(["X" if not case.est_vide() else "O" for case in ligne]))