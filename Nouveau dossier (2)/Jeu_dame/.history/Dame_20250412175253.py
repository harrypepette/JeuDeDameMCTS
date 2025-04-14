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

    def mouvements_possibles(self, plateau,forcer_manger=False):
        """
        Retourne une liste des déplacements disponibles pour une dame.
        Si des captures sont possibles (via check_manger), elles sont prioritaires.
        :param plateau: Le plateau de jeu.
        :return: Liste des positions disponibles (x, y).
        """
        
        cases_a_manger = self.check_manger(plateau)
        if cases_a_manger:
            return cases_a_manger  # Prioriser les captures si disponibles

        mouvements = []
        directions = [(-1, -1), (1, -1), (-1, 1), (1, 1)]  # Diagonales
        for dx, dy in directions:
            x, y = self.position
            while True:
                x += dx
                y += dy
                if 0 <= x < 8 and 0 <= y < 8:
                    if plateau.get_pion(x, y) is None:
                        mouvements.append((x, y))  # Case vide, déplacement possible
                    else:
                        break  # Arrêter si un pion bloque le chemin
                else:
                    break
        return mouvements

    def check_manger(self, plateau):
        """
        Vérifie si la dame peut manger un pion adverse sur toutes les diagonales.
        Retourne une liste des cases derrière les pions adverses capturables.
        :param plateau: Le plateau de jeu.
        :return: Liste des positions disponibles (x, y) derrière les pions adverses capturables.
        """
        cases_a_manger = []
        directions = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        for dx, dy in directions:
            x, y = self.position
            pion_trouve = False
            while True:
                x += dx
                y += dy
                if not (0 <= x < 8 and 0 <= y < 8):
                    break
                pion = plateau.get_pion(x, y)
                if pion:
                  # ! print(f"Pion trouvé à ({x}, {y}), couleur : {pion.couleur}, dame couleur : {self.couleur}")
                    if pion.couleur != self.couleur:
                        pion_trouve = True
                        # Continuer à explorer la diagonale pour trouver toutes les cases libres derrière
                        while True:
                            x += dx
                            y += dy
                            if not (0 <= x < 8 and 0 <= y < 8):
                                break
                            if plateau.get_pion(x, y) is None:
                                cases_a_manger.append((x, y))
                           #!     print(f"Case possible : ({x}, {y})")
                            else:
                                break
                    break
        return cases_a_manger
    
    def cases_fin_manger(self, plateau):
        """
        Retourne une liste des cases où une dame peut finir après avoir mangé.
        """
        cases = []
        directions = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        for dx, dy in directions:
            x, y = self.position
            pion_trouve = False
            while True:
                x += dx
                y += dy
                if not (0 <= x < 8 and 0 <= y < 8):
                    break
                pion = plateau.get_pion(x, y)
                if pion:
                  #!  print(f"Pion trouvé à ({x}, {y}), couleur : {pion.couleur}, dame couleur : {self.couleur}")
                    if pion.couleur != self.couleur:
                        pion_trouve = True
                        # Continuer à explorer la diagonale pour trouver toutes les cases libres derrière
                        while True:
                            x += dx
                            y += dy
                            if not (0 <= x < 8 and 0 <= y < 8):
                                break
                            if plateau.get_pion(x, y) is None:
                                cases.append((x, y))
                     #!           print(f"Case possible : ({x}, {y})")
                            else:
                             break
                    break
        return cases
    def obtenir_case_derriere(self, position_pion, direction):
        """
        Retourne la case derrière un pion adverse si elle est valide.
        :param position_pion: La position du pion adverse (tuple (x, y)).
        :param direction: La direction de déplacement (tuple (dx, dy)).
        :return: La position de la case derrière le pion (tuple (x, y)) ou None si invalide.
        """
        x, y = position_pion
        dx, dy = direction
        case_derriere = (x + dx, y + dy)

        # Vérifier si la case derrière est dans les limites du plateau
        if 0 <= case_derriere[0] < 8 and 0 <= case_derriere[1] < 8:
            return case_derriere
        return None
