import arcade

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

    def afficher(self):
        """
        Affiche le pion sur le plateau.
        """
        couleur = arcade.color.WHITE if self.couleur == "blanc" else arcade.color.BLACK
        x, y = self.position
        arcade.draw_circle_filled(50 + x * 100, 50 + y * 100, 40, couleur)
