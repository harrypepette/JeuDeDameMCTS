class Plateau:
    def __init__(self, taille=8):
        """
        Initialise le plateau de jeu.
        :param taille: Taille du plateau (par défaut 8x8).
        """
        self.taille = taille
        self.cases = [[Case(x, y, 'noir' if (x + y) % 2 == 1 else 'blanc') for y in range(taille)] for x in range(taille)]

    def afficher(self):
        """Affiche le plateau dans la console."""
        for ligne in self.cases:
            print(" ".join(["X" if not case.est_vide() else "O" for case in ligne]))