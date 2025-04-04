import pygame
from Pion import Pion
from Dame import Dame

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
        Promeut un pion en dame s'il atteint le fond du plateau.
        """
        pion = self.get_pion(*mouvement.depart)
        self.cases[mouvement.depart[1]][mouvement.depart[0]] = None

        # Vérifier si un pion est mangé
        dx = mouvement.arrivee[0] - mouvement.depart[0]
        dy = mouvement.arrivee[1] - mouvement.depart[1]
        if abs(dx) > 1 and abs(dy) > 1:  # Capture
            step_x = dx // abs(dx)
            step_y = dy // abs(dy)
            x, y = mouvement.depart
            while (x + step_x, y + step_y) != mouvement.arrivee:
                x += step_x
                y += step_y
                if self.get_pion(x, y) and self.get_pion(x, y).couleur != pion.couleur:
                    self.cases[y][x] = None  # Supprimer le pion mangé
                    break

        # Déplacer le pion
        self.cases[mouvement.arrivee[1]][mouvement.arrivee[0]] = pion
        pion.position = mouvement.arrivee

        # Promouvoir en dame si le pion atteint le fond du plateau
        if (pion.couleur == "blanc" and mouvement.arrivee[1] == 0) or \
           (pion.couleur == "noir" and mouvement.arrivee[1] == 7):
            self.cases[mouvement.arrivee[1]][mouvement.arrivee[0]] = Dame(pion.couleur, pion.position)

    def peut_manger(self, pion):
        """
        Vérifie si un pion peut manger un autre pion.
        :param pion: Le pion à vérifier.
        :return: True si le pion peut manger, False sinon.
        """
        return len(self.cases_fin_manger(pion)) > 0

    def mouvements_possibles(self, pion, forcer_manger=False):
        """
        Retourne une liste des mouvements possibles pour un pion ou une dame.
        :param pion: Le pion ou la dame pour lequel on veut les mouvements.
        :param forcer_manger: Si True, ne retourne que les mouvements de capture.
        :return: Liste des positions disponibles (x, y).
        """
        if isinstance(pion, Dame):
            return pion.mouvements_possibles(self)  # Utiliser la logique spécifique aux dames

        if forcer_manger and self.peut_manger(pion):
            return self.cases_fin_manger(pion)  # Retourne uniquement les mouvements de capture

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
        Retourne une liste des cases où un pion ou une dame peut finir après avoir mangé.
        :param pion: Le pion ou la dame pour lequel on veut les cases.
        :return: Liste des positions disponibles (x, y).
        """
        if isinstance(pion, Dame):
            return pion.cases_fin_manger(self)  # Utiliser la logique spécifique aux dames

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

    def peut_encore_manger(self, pion):
        """
        Vérifie si un pion peut encore manger après un mouvement.
        :param pion: Le pion à vérifier.
        :return: True si le pion peut encore manger, False sinon.
        """
        return len(self.cases_fin_manger(pion)) > 0

    def peut_manger_joueur(self, couleur):
        """
        Vérifie si un joueur a des pions capables de manger.
        :param couleur: La couleur des pions du joueur ('blanc' ou 'noir').
        :return: True si au moins un pion peut manger, False sinon.
        """
        for ligne in self.cases:
            for pion in ligne:
                if pion and pion.couleur == couleur and self.peut_manger(pion):
                    return True
        return False
