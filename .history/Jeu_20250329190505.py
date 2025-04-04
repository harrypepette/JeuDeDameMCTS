class Jeu:
    def __init__(self):
        """
        Initialise le jeu.
        """
        self.plateau = Plateau()
        self.joueurs = [Joueur("Joueur 1", "noir"), Joueur("Joueur 2", "blanc")]
        self.tour_actuel = 0

    def jouer_tour(self):
        """Gère un tour de jeu."""
        joueur = self.joueurs[self.tour_actuel]
        print(f"C'est au tour de {joueur.nom} ({joueur.couleur}).")
        # Logique pour jouer un tour
        self.tour_actuel = (self.tour_actuel + 1) % 2