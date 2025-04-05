from Plateau import Plateau
from Joueur import Joueur
from Regles import Regles
from Mouvement import Mouvement
import random

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
        print("Arrêt du jeu.")

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
                self.selection = (case_x, case_y)
                self.mouvements_possibles = pion.mouvements_possibles(self.plateau)
                self.cases_fin_manger = pion.cases_fin_manger(self.plateau)
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
                # Effectuer le mouvement
                self.plateau.deplacer_pion(pion, mouvement.arrivee)
                self.selection = None
                self.mouvements_possibles = None
                self.cases_fin_manger = None
                self.joueur_actuel = (self.joueur_actuel + 1) % 2  # Changer de joueur

    def run_to_the_end(self, positions):
        """
        Démarre le jeu à partir d'une liste de positions et exécute des mouvements aléatoires jusqu'à ce qu'un joueur gagne.
        :param positions: Liste des positions initiales des pions.
        """
        self.plateau.initialiser_positions(positions)  # Méthode à définir pour initialiser le plateau
        while not self.partie_terminee:
            joueur = self.joueurs[self.joueur_actuel]
            mouvements_possibles = self.plateau.get_mouvements_possibles(joueur.couleur)  # Méthode à définir
            if mouvements_possibles:
                mouvement = random.choice(mouvements_possibles)
                pion = self.plateau.get_pion(*mouvement.depart)
                self.plateau.deplacer_pion(pion, mouvement.arrivee)
                # Vérifier si un joueur a gagné
                if self.plateau.verifier_victoire(joueur.couleur):  # Méthode à définir
                    self.partie_terminee = True
                    print(f"Le joueur {joueur.couleur} a gagné !")
            self.joueur_actuel = (self.joueur_actuel + 1) % 2  # Changer de joueur