import pygame
from Pion import Pion
from Dame import Dame
from copy import deepcopy

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
        self.cases[mouvement.depart[1]][mouvement.depart[0]] = None  # Retirer le pion de sa position initiale

        # Vérifier si c'est une capture
        dx = mouvement.arrivee[0] - mouvement.depart[0]
        dy = mouvement.arrivee[1] - mouvement.depart[1]
        if abs(dx) >= 2 and abs(dy) >= 2:  # Capture (ajusté pour les dames)
            step_x = dx // abs(dx) if dx != 0 else 0
            step_y = dy // abs(dy) if dy != 0 else 0
            x, y = mouvement.depart
            while (x + step_x, y + step_y) != mouvement.arrivee:
                x += step_x
                y += step_y
                pion_milieu = self.get_pion(x, y)
                if pion_milieu and pion_milieu.couleur != pion.couleur:
                    self.cases[y][x] = None  # Supprimer le pion mangé
                    print(f"Pion mangé à ({x}, {y}), dame déplacée à {mouvement.arrivee}")
                    break  # Sortir après avoir trouvé et supprimé le pion adverse

        # Placer le pion ou la dame à la position d'arrivée
        self.cases[mouvement.arrivee[1]][mouvement.arrivee[0]] = pion
        pion.position = mouvement.arrivee

        # Promouvoir en dame si nécessaire (pour un pion normal)
        if not isinstance(pion, Dame) and (
            (pion.couleur == "blanc" and mouvement.arrivee[1] == 0) or
            (pion.couleur == "noir" and mouvement.arrivee[1] == 7)
        ):
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
            return pion.mouvements_possibles(self, forcer_manger=forcer_manger)  # Utiliser la logique spécifique aux dames

        if forcer_manger and self.peut_manger(pion):
            print(f"Capture obligatoire pour le pion à {pion.position}############################")
            return self.cases_fin_manger(pion)  # Retourne uniquement les mouvements de capture

        mouvements = []
        x, y = pion.position
        if pion.couleur == "blanc":
            directions = [(-1, -1), (1, -1)]  # Pion blanc avance vers le haut
        else:
            directions = [(-1, 1), (1, 1)]  # Pion noir avance vers le bas
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8 and self.get_pion(nx, ny) is None:
                if pion.mouvement_valide_direction(dx, dy, est_capture=False):
                    mouvements.append((nx, ny))
        return mouvements

    def cases_fin_manger(self, pion):
        """
        Retourne une liste des cases où un pion ou une dame peut finir après avoir mangé.
        :param pion: Le pion ou la dame pour lequel on veut les cases.
        :return: Liste des positions disponibles (x, y).
        """
        if isinstance(pion, Dame):
            return pion.cases_fin_manger(self)
        cases = []
        x, y = pion.position
        directions = [(-2, -2), (2, -2), (-2, 2), (2, 2)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                milieu = ((x + nx) // 2, (y + ny) // 2)
                pion_milieu = self.get_pion(*milieu)
                if pion_milieu and pion_milieu.couleur != pion.couleur and self.get_pion(nx, ny) is None:
                    cases.append((nx, ny))
                    print(f"Capture possible pour pion à ({x}, {y}) vers ({nx}, {ny})")
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
    
    def partie_terminee(self):
        """
        Vérifie si la partie est terminée.
        Retourne un tuple (est_terminee, gagnant) où :
        - est_terminee : True si la partie est terminée, False sinon.
        - gagnant : "blanc", "noir", ou None si la partie n'est pas terminée.
        """
    # Compter les pions de chaque joueur
        pions_blancs = 0
        pions_noirs = 0
        for i in range(8):
            for j in range(8):
                pion = self.get_pion(i, j)
                if pion:
                    if pion.couleur == "blanc":
                        pions_blancs += 1
                    elif pion.couleur == "noir":
                        pions_noirs += 1

    # Si un joueur n'a plus de pions, l'autre gagne
        if pions_blancs == 0:
            return True, "noir"
        if pions_noirs == 0:
            return True, "blanc"

    # Vérifier si le joueur actuel peut bouger
        joueur_actuel = self.jeu.joueurs[self.jeu.joueur_actuel].couleur if hasattr(self, 'jeu') else "blanc"
        peut_bouger = False
        for i in range(8):
            for j in range(8):
                pion = self.get_pion(i, j)
                if pion and pion.couleur == joueur_actuel:
                # Vérifier si ce pion peut bouger ou capturer
                    mouvements = self.mouvements_possibles(pion, forcer_manger=False)
                    captures = self.cases_fin_manger(pion)
                    if mouvements or captures:
                        peut_bouger = True
                        break
            if peut_bouger:
                break

    # Si le joueur actuel ne peut pas bouger, l'autre joueur gagne
        if not peut_bouger:
            gagnant = "noir" if joueur_actuel == "blanc" else "blanc"
            return True, gagnant

        return False, None
    
    def copie_sans_surface(self):
        """
        Crée une copie du plateau sans inclure les objets non sérialisables comme pygame.Surface.
        """
        copie = Plateau()
        copie.cases = deepcopy(self.cases)  # Copie profonde des cases
        # Ne pas copier les références à pygame.Surface ou autres objets non sérialisables
        copie.jeu = None  # Exclure la référence au jeu
        return copie