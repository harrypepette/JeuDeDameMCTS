from Pion import Pion
import pygame


class Dame(Pion):
    def __init__(self, couleur, position):
        """
        Initialise une dame.
        :param couleur: La couleur de la dame ('blanc' ou 'noir').
        :param position: La position initiale de la dame (tuple (x, y)).
        """
        super().__init__(couleur, position)
        self.est_dame = True

    def afficher(self, screen):
        """
        Affiche la dame sur le plateau.
        """
        couleur = (255, 255, 255) if self.couleur == "blanc" else (0, 0, 0)
        x, y = self.position
        pygame.draw.circle(screen, couleur, (50 + x * 100, 50 + y * 100), 40)
        pygame.draw.circle(screen, (255, 215, 0), (50 + x * 100, 50 + y * 100), 30)  # Cercle doré pour indiquer une dame

    def mouvements_possibles(self, plateau):
        """
        Retourne une liste des mouvements possibles pour une dame.
        :param plateau: Le plateau de jeu.
        :return: Liste des positions disponibles (x, y).
        """
        mouvements = []
        directions = [(-1, -1), (1, -1), (-1, 1), (1, 1)]  # Diagonales
        for dx, dy in directions:
            x, y = self.position
            while True:
                x += dx
                y += dy
                if 0 <= x < 8 and 0 <= y < 8:
                    pion = plateau.get_pion(x, y)
                    if pion is None:
                        mouvements.append((x, y))  # Case vide, déplacement possible
                    else:
                        if pion.couleur != self.couleur:
                            mouvements.append((x, y))  # Capture possible
                        break  # Arrêter si un pion bloque le chemin
                else:
                    break
        return mouvements
