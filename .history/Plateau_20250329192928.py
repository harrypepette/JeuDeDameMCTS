import arcade

class Plateau:
    def __init__(self):
        self.cases = [[None for _ in range(8)] for _ in range(8)]

    def afficher(self):
        """
        Affiche le plateau.
        """
        for x in range(8):
            for y in range(8):
                couleur = arcade.color.WHITE if (x + y) % 2 == 0 else arcade.color.BLACK
                arcade.draw_rectangle_filled(50 + x * 100, 50 + y * 100, 100, 100, couleur)
