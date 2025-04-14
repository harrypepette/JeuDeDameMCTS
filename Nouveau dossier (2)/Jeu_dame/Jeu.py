import pygame
from Plateau import Plateau
from Joueur import Joueur
from Regles import Regles
from Mouvement import Mouvement
from Simulation import Simulation

class Jeu:

    def __init__(self, couleur_simulee, iterations=5000):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        pygame.display.set_caption("Jeu de Dames")
        self.clock = pygame.time.Clock()
        self.plateau = Plateau()
        self.plateau.jeu = self
        self.joueurs = [Joueur("blanc"), Joueur("noir")]
        self.joueur_actuel = 0
        self.selection = None
        self.mouvements_possibles = None
        self.cases_fin_manger = None
        self.partie_terminee = False
        self.couleur_simulee = couleur_simulee  # Stocker la couleur simulée
        self.iterations = iterations  # Initialize iterations attribute

    def afficher_tour(self):
        """
        Affiche le joueur actuel en haut de l'écran.
        """
        font = pygame.font.Font(None, 36)  # Police et taille du texte
        texte = f"Tour : {self.joueurs[self.joueur_actuel].couleur.capitalize()}"
        couleur = (255, 255, 255)  # Couleur du texte (blanc)
        surface_texte = font.render(texte, True, couleur)
        self.screen.blit(surface_texte, (10, 10))  # Position du texte (en haut à gauche)

    def run(self):
        running = True
    
        # Vérifier si le joueur initial est celui qui doit être simulé
        if self.joueurs[self.joueur_actuel].couleur == self.couleur_simulee:
            self.lancer_simulation()
    
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
        print("Arrêt du jeu.")


    def on_draw(self):
        """
        Méthode appelée pour dessiner l'écran.
        """
        self.screen.fill((0, 0, 0))  # Remplir l'écran avec une couleur noire
        self.plateau.afficher(self.screen, self.mouvements_possibles, self.cases_fin_manger)
        self.afficher_tour()  # Afficher le texte indiquant le tour actuel
        
    def on_mouse_press(self, pos):
        """
        Méthode appelée lorsqu'un clic de souris est détecté.
        """
        if self.partie_terminee:
            print("La partie est terminée !")
            return
        
        if self.joueurs[self.joueur_actuel].couleur == self.couleur_simulee:
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
                #!print(f"Mouvements possibles : {self.mouvements_possibles}")
                #!print(f"Cases de capture : {self.cases_fin_manger}")
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
                   #! print(f"Mouvement invalide : ({case_x}, {case_y}) n'est pas dans les mouvements possibles {self.mouvements_possibles}")
                    return  # Mouvement simple non valide
    
                self.plateau.deplacer_pion(mouvement)
    
                if est_prise and self.plateau.peut_encore_manger(pion):
                    self.selection = pion.position
                    self.mouvements_possibles = self.plateau.cases_fin_manger(pion)
                    self.cases_fin_manger = self.plateau.cases_fin_manger(pion)
                  #!  print(f"Mouvements possibles après capture : {self.mouvements_possibles}")
                    #!print(f"Cases de capture après capture : {self.cases_fin_manger}")
                else:
                    self.selection = None
                    self.mouvements_possibles = None
                    self.cases_fin_manger = None
                    self.joueur_actuel = 1 - self.joueur_actuel  # Changer de joueur
                    
                est_terminee, gagnant = self.plateau.partie_terminee()
                if est_terminee:
                    self.partie_terminee = True
                    print(f"La partie est terminée ! Le joueur {gagnant} a gagné !")
                else:
                    # Vérifier si le joueur actuel correspond à la couleur simulée
                    if self.joueurs[self.joueur_actuel].couleur == self.couleur_simulee:
                        self.lancer_simulation()

    def lancer_simulation(self):
        """Lance la simulation pour trouver le meilleur mouvement pour le joueur simulé"""
        if self.couleur_simulee == "blanc" and self.joueur_actuel == 0 or \
           self.couleur_simulee == "noir" and self.joueur_actuel == 1:
            pions = []
            for ligne in self.plateau.cases:
                for pion in ligne:
                    if pion and ((self.joueur_actuel == 0 and pion.couleur == "blanc") or
                                 (self.joueur_actuel == 1 and pion.couleur == "noir")):
                        pions.append(pion)
        
        # Utiliser le paramètre iterations stocké dans la classe
        simulation = Simulation(self.plateau, self.couleur_simulee)
        print(f"Lancement de la simulation pour les {self.couleur_simulee} avec {self.iterations} itérations...")
        meilleur_mouvement = simulation.simuler_meilleur_mouvement(iterations=self.iterations)
            
        if meilleur_mouvement:
            print(f"Le meilleur mouvement pour les {self.couleur_simulee} est : {meilleur_mouvement}")
            
            # Appliquer le meilleur mouvement pour le joueur simulé
            pion_position, destination = meilleur_mouvement
            pion = self.plateau.get_pion(*pion_position)
            mouvement = Mouvement(pion_position, destination)
            self.plateau.deplacer_pion(mouvement)
            
            # Vérifier si des captures supplémentaires sont possibles
            capture_effectuee = (abs(destination[0] - pion_position[0]) >= 2 and 
                                 abs(destination[1] - pion_position[1]) >= 2)

            pion_apres_mouvement = self.plateau.get_pion(*destination)
            
           # Tant que des captures supplémentaires sont possibles
            while capture_effectuee and pion_apres_mouvement and self.plateau.peut_encore_manger(pion_apres_mouvement):
            # Obtenir les prochaines positions de capture possibles
                prochaines_positions = self.plateau.cases_fin_manger(pion_apres_mouvement)
    
                if prochaines_positions:
                    # Vérifier que le mouvement respecte les règles du jeu
                    ancienne_position = pion_apres_mouvement.position
                    for nouvelle_destination in prochaines_positions:
            # S'assurer que le mouvement est strictement diagonal
                        dx = nouvelle_destination[0] - ancienne_position[0]
                        dy = nouvelle_destination[1] - ancienne_position[1]
            
                 # Vérifier que c'est un mouvement diagonal valide
                        if abs(dx) == abs(dy) and abs(dx) >= 2:
                            mouvement = Mouvement(ancienne_position, nouvelle_destination)
                            self.plateau.deplacer_pion(mouvement)
                            print(f"Capture multiple: {ancienne_position} -> {nouvelle_destination}")
                
                # Mettre à jour le pion pour la prochaine itération
                            pion_apres_mouvement = self.plateau.get_pion(*nouvelle_destination)
                            break
                else:
                    
                    break
            
            # Vérifier si la partie est terminée après le mouvement
            est_terminee, gagnant = self.plateau.partie_terminee()
            if est_terminee:
                self.partie_terminee = True
                print(f"La partie est terminée ! Le joueur {gagnant} a gagné !")
            else:
                self.joueur_actuel = 1 - self.joueur_actuel  # Passer au joueur suivant
        else:
            print(f"Aucun mouvement trouvé pour les {self.couleur_simulee}, ils perdent par forfait!")
            self.partie_terminee = True
