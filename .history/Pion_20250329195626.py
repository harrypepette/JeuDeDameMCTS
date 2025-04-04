import pygame

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

    def afficher(self, screen):
        """
        Affiche le pion sur le plateau.
        """
        couleur = (255, 255, 255) if self.couleur == "blanc" else (0, 0, 0)
        x, y = self.position
        pygame.draw.circle(screen, couleur, (50 + x * 100, 50 + y * 100), 40)
