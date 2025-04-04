import arcade
from Plateau import Plateau
from Joueur import Joueur
from Jeu import Jeu
from Interface import Interface

# Constantes pour la fenêtre et le plateau
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
SCREEN_TITLE = "Jeu de Dames"
CASE_SIZE = 100  # Taille d'une case (en pixels)
PLATEAU_TAILLE = 8  # Taille du plateau (8x8)

class JeuDeDames(arcade.Window):
    """
    Classe principale pour le jeu de dames utilisant Arcade.
    """

    def __init__(self, plateau, joueurs):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.WHITE)
        self.plateau = plateau
        self.joueurs = joueurs
        self.joueur_actuel = 0  # Index du joueur actuel

    def on_draw(self):
        """
        Méthode appelée pour dessiner à l'écran.
        """
        arcade.start_render()
        self.dessiner_plateau()
        self.dessiner_pions()

    def dessiner_plateau(self):
        """
        Dessine le plateau de jeu de dames.
        """
        for row in range(PLATEAU_TAILLE):
            for col in range(PLATEAU_TAILLE):
                # Alterner les couleurs des cases
                if (row + col) % 2 == 0:
                    couleur = arcade.color.BEIGE
                else:
                    couleur = arcade.color.BROWN
                x = col * CASE_SIZE + CASE_SIZE / 2
                y = row * CASE_SIZE + CASE_SIZE / 2
                arcade.draw_rectangle_filled(x, y, CASE_SIZE, CASE_SIZE, couleur)

    def dessiner_pions(self):
        """
        Dessine les pions sur le plateau.
        """
        for row in range(PLATEAU_TAILLE):
            for col in range(PLATEAU_TAILLE):
                pion = self.plateau.get_pion(row, col)
                if pion:
                    x = col * CASE_SIZE + CASE_SIZE / 2
                    y = row * CASE_SIZE + CASE_SIZE / 2
                    couleur = arcade.color.BLACK if pion.couleur == "noir" else arcade.color.WHITE
                    arcade.draw_circle_filled(x, y, CASE_SIZE / 3, couleur)

    def on_mouse_press(self, x, y, button, modifiers):
        """
        Méthode appelée lorsqu'un bouton de la souris est pressé.
        """
        col = x // CASE_SIZE
        row = y // CASE_SIZE
        print(f"Clic détecté sur la case : ({row}, {col})")
        # Implémenter la logique pour sélectionner et déplacer les pions

def main():
    """
    Point d'entrée principal pour lancer le jeu.
    """
    # Initialisation des joueurs
    joueur1 = Joueur("Joueur 1", "noir")
    joueur2 = Joueur("Joueur 2", "blanc")

    # Initialisation du plateau
    plateau = Plateau()

    # Lancer le jeu avec Arcade
    jeu = JeuDeDames(plateau, [joueur1, joueur2])
    arcade.run()

if __name__ == "__main__":
    main()