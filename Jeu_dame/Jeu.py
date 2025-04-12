import pygame
from Plateau import Plateau
from Joueur import Joueur
from Regles import Regles
from Mouvement import Mouvement
from IA import IA  # Import the IA class
from Etat import Etat  # Import the Etat class

class Jeu:
    def __init__(self, ia_active=True, ia_couleur="noir", ia_iterations=1000):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        pygame.display.set_caption("Jeu de Dames avec IA")
        self.clock = pygame.time.Clock()
        self.plateau = Plateau()
        self.plateau.jeu = self
        self.joueurs = [Joueur("blanc"), Joueur("noir")]
        self.joueur_actuel = 0
        self.selection = None
        self.mouvements_possibles = None
        self.cases_fin_manger = None
        self.partie_terminee = False
        
        # Configuration de l'IA
        self.ia_active = ia_active
        self.ia_couleur = ia_couleur
        self.ia = IA(ia_couleur, iterations=ia_iterations) if ia_active else None
        
        # Flag pour indiquer si l'IA est en train de réfléchir
        self.ia_est_en_train_de_reflechir = False
        
        
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.ia_est_en_train_de_reflechir:
                    self.on_mouse_press(event.pos)
                    
            #print(f"IA active: {self.ia_active}, Partie terminée: {self.partie_terminee}, "
              #f"Couleur joueur actuel: {self.joueurs[self.joueur_actuel].couleur}, "
              #f"Couleur IA: {self.ia_couleur}, IA réfléchit: {self.ia_est_en_train_de_reflechir}")
            
            # Si c'est le tour de l'IA et que l'IA est activée
            if (self.ia_active and not self.partie_terminee and 
                self.joueurs[self.joueur_actuel].couleur == self.ia_couleur and 
                not self.ia_est_en_train_de_reflechir):
                self.faire_jouer_ia()
            
            self.on_draw()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        
        
    def faire_jouer_ia(self):
        import time
        debut = time.time()
        self.ia_est_en_train_de_reflechir = True

        # Vérifier si la partie est terminée
        est_terminee, gagnant = self.plateau.partie_terminee()
        if est_terminee:
            print(f"La partie est terminée ! Gagnant : {gagnant}")
            self.partie_terminee = True
            self.ia_est_en_train_de_reflechir = False
            return

        etat = Etat.depuis_jeu(self)
        print(f"État créé pour joueur {self.joueurs[self.joueur_actuel].couleur}")
        mouvement = self.ia.choisir_mouvement(etat)

        if mouvement:
            print(f"Mouvement choisi: {mouvement.depart} -> {mouvement.arrivee}")
            self.selection = mouvement.depart
            pion = self.plateau.get_pion(*self.selection)
            self.mouvements_possibles = self.plateau.mouvements_possibles(pion, forcer_manger=True)
            self.cases_fin_manger = self.plateau.cases_fin_manger(pion)

            pygame.time.delay(500)
            est_prise = (mouvement.arrivee[0], mouvement.arrivee[1]) in self.cases_fin_manger
            self.plateau.deplacer_pion(mouvement)

            self.on_draw()
            pygame.display.flip()

            pion = self.plateau.get_pion(*mouvement.arrivee)
            if est_prise and pion and self.plateau.peut_encore_manger(pion):
                print("Capture supplémentaire possible")
                self.selection = pion.position
                self.mouvements_possibles = self.plateau.cases_fin_manger(pion)
                self.cases_fin_manger = self.plateau.cases_fin_manger(pion)
                pygame.time.delay(500)
                self.faire_jouer_ia()
            else:
                self.selection = None
                self.mouvements_possibles = None
                self.cases_fin_manger = None
                self.joueur_actuel = 1 - self.joueur_actuel
                print(f"Changement de joueur vers {self.joueurs[self.joueur_actuel].couleur}")

                est_terminee, gagnant = self.plateau.partie_terminee()
                if est_terminee:
                    self.partie_terminee = True
                    print(f"La partie est terminée ! Gagnant : {gagnant}")
        else:
            print(f"Aucun mouvement possible pour {self.joueurs[self.joueur_actuel].couleur}")
            self.joueur_actuel = 1 - self.joueur_actuel
            est_terminee, gagnant = self.plateau.partie_terminee()
            if est_terminee:
                self.partie_terminee = True
                print(f"La partie est terminée ! Gagnant : {gagnant}")

        self.ia_est_en_train_de_reflechir = False
        print(f"Temps de calcul de l'IA: {time.time() - debut:.2f} secondes")
    
    def stop(self):
        """
        Méthode pour arrêter le jeu.
        """
        print("Arrêt du jeu.")


    def on_draw(self):
        """
        Méthode appelée pour dessiner l'écran.
        """
        self.screen.fill((0, 0, 0))
        self.plateau.afficher(self.screen, self.mouvements_possibles, self.cases_fin_manger)
        
        # Indiquer le joueur actuel
        font = pygame.font.SysFont(None, 36)
        couleur = "blanc" if self.joueurs[self.joueur_actuel].couleur == "blanc" else "noir"
        texte = f"Tour du joueur: {couleur}" + (" (IA)" if self.ia_active and couleur == self.ia_couleur else "")
        couleur_texte = (255, 255, 255) if couleur == "blanc" else (0, 0, 0)
        text_surface = font.render(texte, True, couleur_texte)
        self.screen.blit(text_surface, (10, 10))
        
        # Afficher un message si la partie est terminée
        if self.partie_terminee:
            est_terminee, gagnant = self.plateau.partie_terminee()
            message = f"Partie terminée ! {gagnant.capitalize()} a gagné !"
            text_surface = font.render(message, True, (255, 0, 0))
            self.screen.blit(text_surface, (250, 400))

    def on_mouse_press(self, pos):
        """
        Méthode appelée lorsqu'un clic de souris est détecté.
        """
        if self.partie_terminee:
            print("La partie est terminée !")
            return
        
        # Si c'est le tour de l'IA, ignorer les clics
        if self.ia_active and self.joueurs[self.joueur_actuel].couleur == self.ia_couleur:
            print("C'est au tour de l'IA !")
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
                for case in self.cases_fin_manger:
                    print(f"Capture possible pour pion à ({self.selection[0]}, {self.selection[1]}) vers {case}")
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
                    for case in self.cases_fin_manger:
                        print(f"Capture possible pour pion à ({self.selection[0]}, {self.selection[1]}) vers {case}")
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
                    