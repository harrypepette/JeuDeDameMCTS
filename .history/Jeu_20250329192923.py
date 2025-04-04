import arcade
from Plateau import Plateau
from Joueur import Joueur

class Jeu(arcade.Window):
    def __init__(self):
        super().__init__(800, 800, "Jeu de Dames")
        self.plateau = Plateau()
        self.joueurs = [Joueur("blanc"), Joueur("noir")]
        self.joueur_actuel = 0

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
        # Logique pour gérer les mouvements
        pass
