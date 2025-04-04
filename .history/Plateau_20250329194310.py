import arcade
from Pion import Pion

class Plateau:
    def __init__(self):
        self.cases = [[None for _ in range(8)] for _ in range(8)]
        self.initialiser_pions()

    def initialiser_pions(self):
        """
        Place les pions sur le plateau au début du jeu.
        """
        for y in range(3):
            for x in range(8):
                if (x + y) % 2 == 1:
                    self.cases[y][x] = Pion("noir", (x, y))
        for y in range(5, 8):
            for x in range(8):
                if (x + y) % 2 == 1:
                    self.cases[y][x] = Pion("blanc", (x, y))

    def afficher(self):
        """
        Affiche le plateau.
        """
        for x in range(8):
            for y in range(8):
                couleur = arcade.color.WHITE if (x + y) % 2 == 0 else arcade.color.BLACK
                left = x * 100
                right = left + 100
                bottom = y * 100
                top = bottom + 100
                arcade.draw_lrtb_rectangle_filled(left, right, top, bottom, couleur)
                pion = self.cases[y][x]
                if pion:
                    pion.afficher()

    def get_pion(self, x, y):
        """
        Retourne le pion à une position donnée.
        """
        return self.cases[y][x]

    def deplacer_pion(self, mouvement):
        """
        Déplace un pion sur le plateau.
        """
        pion = self.get_pion(*mouvement.depart)
        self.cases[mouvement.depart[1]][mouvement.depart[0]] = None
        self.cases[mouvement.arrivee[1]][mouvement.arrivee[0]] = pion
        pion.position = mouvement.arrivee
