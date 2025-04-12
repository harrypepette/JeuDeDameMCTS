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

    def mouvement_valide_direction(self, dx, dy, est_capture):
        """
        Vérifie si le mouvement respecte les règles de direction pour un pion.
        :param dx: Déplacement en x.
        :param dy: Déplacement en y.
        :param est_capture: True si le mouvement est une capture, False sinon.
        :return: True si le mouvement est valide, False sinon.
        """
        if self.est_dame or est_capture:
            return True  # Les dames ou les captures peuvent reculer
        if self.couleur == "blanc":
            return dy < 0  # Les pions blancs avancent vers le haut
        elif self.couleur == "noir":
            return dy > 0  # Les pions noirs avancent vers le bas
        return False
    def copier(self):
        return Pion(self.couleur, self.position[:])