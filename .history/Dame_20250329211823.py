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

    def mouvements_possibles(self, plateau, forcer_manger=False):
        """
        Retourne une liste des mouvements possibles pour une dame.
        :param plateau: Le plateau de jeu.
        :param forcer_manger: Si True, ne retourne que les mouvements de capture.
        :return: Liste des positions disponibles (x, y).
        """
        mouvements = []
        directions = [(-1, -1), (1, -1), (-1, 1), (1, 1)]  # Diagonales
        for dx, dy in directions:
            x, y = self.position
            capture_possible = False
            while True:
                x += dx
                y += dy
                if 0 <= x < 8 and 0 <= y < 8:
                    pion = plateau.get_pion(x, y)
                    if pion is None:
                        if not forcer_manger or capture_possible:
                            mouvements.append((x, y))  # Case vide, déplacement possible
                    else:
                        if pion.couleur != self.couleur and not capture_possible:
                            # Vérifier si la case derrière le pion est libre pour une capture
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < 8 and 0 <= ny < 8 and plateau.get_pion(nx, ny) is None:
                                mouvements.append((nx, ny))  # Capture possible
                                capture_possible = True
                        break  # Arrêter si un pion bloque le chemin
                else:
                    break
        return mouvements

    def mouvements_disponibles(self, plateau):
        """
        Retourne une liste des déplacements disponibles pour une dame sur toutes les diagonales.
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
        directions = [(-1, -1), (1, -1), (-1, 1), (1, 1)]  # Diagonales
        for dx, dy in directions:
            x, y = self.position
            while True:
                x += dx
                y += dy
                if 0 <= x < 8 and 0 <= y < 8:
                    pion = plateau.get_pion(x, y)
                    if pion and pion.couleur != self.couleur:
                        # Vérifier si la case derrière le pion est libre
                        case_derriere = self.obtenir_case_derriere((x, y), (dx, dy))
                        if case_derriere and plateau.get_pion(*case_derriere) is None:
                            cases_a_manger.append(case_derriere)  # Ajouter la case derrière le pion capturé
                        break  # Arrêter après avoir trouvé un pion à capturer
                    elif pion is not None:
                        break  # Arrêter si un pion bloque le chemin
                else:
                    break
        return cases_a_manger

    def cases_fin_manger(self, plateau):
        """
        Retourne une liste des cases où une dame peut finir après avoir mangé.
        :param plateau: Le plateau de jeu.
        :return: Liste des positions disponibles (x, y).
        """
        cases = []
        directions = [(-1, -1), (1, -1), (-1, 1), (1, 1)]  # Diagonales
        for dx, dy in directions:
            x, y = self.position
            while True:
                x += dx
                y += dy
                if 0 <= x < 8 and 0 <= y < 8:
                    pion = plateau.get_pion(x, y)
                    if pion and pion.couleur != self.couleur:
                        # Vérifier si la case derrière le pion est libre
                        case_derriere = self.obtenir_case_derriere((x, y), (dx, dy))
                        if case_derriere and plateau.get_pion(*case_derriere) is None:
                            cases.append(case_derriere)  # Ajouter la case derrière le pion capturé
                        break  # Arrêter après avoir trouvé un pion à capturer
                    elif pion is not None:
                        break  # Arrêter si un pion bloque le chemin
                else:
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
