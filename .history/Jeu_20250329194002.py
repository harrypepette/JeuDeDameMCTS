import arcade
from Plateau import Plateau
from Joueur import Joueur
from Regles import Regles
from Mouvement import Mouvement

class Jeu(arcade.Window):
    def __init__(self):
        super().__init__(800, 800, "Jeu de Dames")
        self.plateau = Plateau()
        self.joueurs = [Joueur("blanc"), Joueur("noir")]
        self.joueur_actuel = 0
        self.selection = None  # Case sélectionnée

    def on_draw(self):
        """
        Méthode appelée pour dessiner l'écran.
        """
        arcade.start_render()
        self.plateau.afficher()

    def on_mouse_press(self, x, y, button, modifiers):
        """
        Méthode appelée lorsqu'un clic de souris est détecté.
        """
        case_x, case_y = x // 100, y // 100
        if self.selection is None:
            # Sélectionner un pion
            pion = self.plateau.get_pion(case_x, case_y)
            if pion and pion.couleur == self.joueurs[self.joueur_actuel].couleur:
                self.selection = (case_x, case_y)
        else:
            # Déplacer le pion
            mouvement = Mouvement(self.selection, (case_x, case_y))
            pion = self.plateau.get_pion(*self.selection)
            if Regles.mouvement_valide(pion, mouvement, self.plateau):
                self.plateau.deplacer_pion(mouvement)
                self.joueur_actuel = 1 - self.joueur_actuel  # Changer de joueur
            self.selection = None
