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

    def afficher(self, screen, mouvements_possibles=None, cases_fin_manger=None):
        """
        Affiche le plateau.
        :param screen: L'écran Pygame.
        :param mouvements_possibles: Liste des positions disponibles pour un déplacement.
        :param cases_fin_manger: Liste des positions où un pion peut finir après avoir mangé.
        """
        for x in range(8):
            for y in range(8):
                couleur = (200, 200, 200) if (x + y) % 2 == 0 else (50, 50, 50)
                rect = pygame.Rect(x * 100, y * 100, 100, 100)
                pygame.draw.rect(screen, couleur, rect)

                # Mettre en surbrillance les mouvements possibles
                if mouvements_possibles and (x, y) in mouvements_possibles:
                    pygame.draw.rect(screen, (0, 255, 0), rect, 5)

                # Mettre en surbrillance les cases où un pion peut finir après avoir mangé
                if cases_fin_manger and (x, y) in cases_fin_manger:
                    pygame.draw.rect(screen, (255, 0, 0), rect, 5)

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
        Déplace un pion sur le plateau et mange un pion si nécessaire.
        """
        pion = self.get_pion(*mouvement.depart)
        self.cases[mouvement.depart[1]][mouvement.depart[0]] = None

        # Vérifier si un pion est mangé
        dx = mouvement.arrivee[0] - mouvement.depart[0]
        dy = mouvement.arrivee[1] - mouvement.depart[1]
        if abs(dx) == 2 and abs(dy) == 2:  # Saut
            milieu = ((mouvement.depart[0] + mouvement.arrivee[0]) // 2,
                      (mouvement.depart[1] + mouvement.arrivee[1]) // 2)
            self.cases[milieu[1]][milieu[0]] = None  # Supprimer le pion mangé

        self.cases[mouvement.arrivee[1]][mouvement.arrivee[0]] = pion
        pion.position = mouvement.arrivee

    def mouvements_possibles(self, pion):
        """
        Retourne une liste des mouvements possibles pour un pion donné.
        :param pion: Le pion pour lequel on veut les mouvements.
        :return: Liste des positions disponibles (x, y).
        """
        mouvements = []
        x, y = pion.position
        directions = [(-1, -1), (1, -1), (-1, 1), (1, 1)]  # Diagonales
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8 and self.get_pion(nx, ny) is None:
                mouvements.append((nx, ny))
        return mouvements

    def cases_fin_manger(self, pion):
        """
        Retourne une liste des cases où un pion peut finir après avoir mangé.
        :param pion: Le pion pour lequel on veut les cases.
        :return: Liste des positions disponibles (x, y).
        """
        cases = []
        x, y = pion.position
        directions = [(-2, -2), (2, -2), (-2, 2), (2, 2)]  # Sauts possibles
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                milieu = ((x + nx) // 2, (y + ny) // 2)
                pion_milieu = self.get_pion(*milieu)
                if pion_milieu and pion_milieu.couleur != pion.couleur and self.get_pion(nx, ny) is None:
                    cases.append((nx, ny))
        return cases
