import pygame
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

    def afficher(self, screen):
        """
        Affiche le plateau.
        """
        for x in range(8):
            for y in range(8):
                couleur = (200, 200, 200) if (x + y) % 2 == 0 else (50, 50, 50)  # Couleurs ajustées
                rect = pygame.Rect(x * 100, y * 100, 100, 100)
                pygame.draw.rect(screen, couleur, rect)
                pion = self.cases[y][x]
                if pion:
                    pion.afficher(screen)

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
