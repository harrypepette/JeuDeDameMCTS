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
        self.plateau.jeu = self
        self.joueurs = [Joueur("blanc"), Joueur("noir")]
        self.joueur_actuel = 0
        self.selection = None
        self.mouvements_possibles = None  # Liste des mouvements possibles pour le pion sélectionné
        self.cases_fin_manger = None  # Liste des cases où un pion peut finir après avoir mangé
        self.partie_terminee = False

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

    def stop(self):
        """
        Méthode pour arrêter le jeu.
        """
        print("Le jeu a été arrêté.")


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
        if self.partie_terminee:
            print("La partie est terminée !")
            return
        
        case_x, case_y = pos[0] // 100, pos[1] // 100
        if self.selection is None:
            # Sélectionner un pion
            pion = self.plateau.get_pion(case_x, case_y)
            if pion and pion.couleur == self.joueurs[self.joueur_actuel].couleur:
                if self.plateau.peut_manger_joueur(self.joueurs[self.joueur_actuel].couleur):
                    if not self.plateau.peut_manger(pion):
                        print(f"Capture obligatoire ailleurs, ce pion ne peut pas être sélectionné : ({case_x}, {case_y})")
                        return
                self.selection = (case_x, case_y)
                self.mouvements_possibles = self.plateau.mouvements_possibles(pion, forcer_manger=True)
                self.cases_fin_manger = self.plateau.cases_fin_manger(pion)
                print(f"Mouvements possibles : {self.mouvements_possibles}")
                print(f"Cases de capture : {self.cases_fin_manger}")
        elif self.selection == (case_x, case_y):
            # Désélectionner le pion
            self.selection = None
            self.mouvements_possibles = None
            self.cases_fin_manger = None
        else:
            # Déplacer le pion
            mouvement = Mouvement(self.selection, (case_x, case_y))
            pion = self.plateau.get_pion(*self.selection)
            if Regles.mouvement_valide(pion, mouvement, self.plateau):
                # Vérifier si le mouvement est une capture
                est_prise = (case_x, case_y) in self.cases_fin_manger
                if not est_prise and (case_x, case_y) not in self.mouvements_possibles:
                    print(f"Mouvement invalide : ({case_x}, {case_y}) n'est pas dans les mouvements possibles {self.mouvements_possibles}")
                    return  # Mouvement simple non valide

                self.plateau.deplacer_pion(mouvement)

                if est_prise and self.plateau.peut_encore_manger(pion):
                    self.selection = pion.position
                    self.mouvements_possibles = self.plateau.cases_fin_manger(pion)
                    self.cases_fin_manger = self.plateau.cases_fin_manger(pion)
                    print(f"Mouvements possibles après capture : {self.mouvements_possibles}")
                    print(f"Cases de capture après capture : {self.cases_fin_manger}")
                else:
                    self.selection = None
                    self.mouvements_possibles = None
                    self.cases_fin_manger = None
                    self.joueur_actuel = 1 - self.joueur_actuel  # Changer de joueur
                    
                est_terminee, gagnant = self.plateau.partie_terminee()
                if est_terminee:
                    self.partie_terminee = True
                    print(f"La partie est terminée ! Le joueur {gagnant} a gagné !")