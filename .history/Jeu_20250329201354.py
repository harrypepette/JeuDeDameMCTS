import pygame
from Plateau import Plateau
from Joueur import Joueur
from Regles import Regles
from Mouvement import Mouvement

class Jeu:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        pygame.display.set_caption("Jeu de Dames")
        self.clock = pygame.time.Clock()
        self.plateau = Plateau()
        self.joueurs = [Joueur("blanc"), Joueur("noir")]
        self.joueur_actuel = 0
        self.selection = None
        self.mouvements_possibles = None  # Liste des mouvements possibles pour le pion sélectionné
        self.cases_fin_manger = None  # Liste des cases où un pion peut finir après avoir mangé

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.on_mouse_press(event.pos)

            self.on_draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    def on_draw(self):
        """
        Méthode appelée pour dessiner l'écran.
        """
        self.screen.fill((0, 0, 0))
        self.plateau.afficher(self.screen, self.mouvements_possibles, self.cases_fin_manger)

    def on_mouse_press(self, pos):
        """
        Méthode appelée lorsqu'un clic de souris est détecté.
        """
        case_x, case_y = pos[0] // 100, pos[1] // 100
        if self.selection is None:
            pion = self.plateau.get_pion(case_x, case_y)
            if pion and pion.couleur == self.joueurs[self.joueur_actuel].couleur:
                self.selection = (case_x, case_y)
                # Forcer les mouvements de capture si disponibles
                self.mouvements_possibles = self.plateau.mouvements_possibles(pion, forcer_manger=True)
                self.cases_fin_manger = self.plateau.cases_fin_manger(pion)
        else:
            mouvement = Mouvement(self.selection, (case_x, case_y))
            pion = self.plateau.get_pion(*self.selection)
            if Regles.mouvement_valide(pion, mouvement, self.plateau):
                self.plateau.deplacer_pion(mouvement)
                # Vérifier si le mouvement est une prise
                dx = mouvement.arrivee[0] - mouvement.depart[0]
                dy = mouvement.arrivee[1] - mouvement.depart[1]
                est_prise = abs(dx) == 2 and abs(dy) == 2

                if est_prise and self.plateau.peut_encore_manger(pion):
                    # Si c'est une prise et que le pion peut encore manger, continuer le tour
                    self.selection = pion.position
                    self.mouvements_possibles = self.plateau.cases_fin_manger(pion)
                    self.cases_fin_manger = self.plateau.cases_fin_manger(pion)
                else:
                    # Si c'est un déplacement simple ou si le pion ne peut plus manger, terminer le tour
                    self.selection = None
                    self.mouvements_possibles = None
                    self.cases_fin_manger = None
                    self.joueur_actuel = 1 - self.joueur_actuel  # Changer de joueur
