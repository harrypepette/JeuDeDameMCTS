import random

class Ia:
    def __init__(self, name="IA"):
        """
        Initialise une instance de l'IA.
        :param name: Nom de l'IA (par défaut "IA").
        """
        self.name = name

    def choisir_coup(self, coups_possibles):
        """
        Choisit un coup aléatoire parmi les coups possibles.
        :param coups_possibles: Liste des coups possibles.
        :return: Un coup choisi aléatoirement.
        """
        if not coups_possibles:
            raise ValueError("Aucun coup possible pour l'IA.")
        return random.choice(coups_possibles)

    def jouer(self, jeu):
        """
        Joue un coup sur le jeu.
        :param jeu: Instance du jeu qui contient l'état actuel et les règles.
        """
        coups_possibles = jeu.get_coups_possibles()
        coup_choisi = self.choisir_coup(coups_possibles)
        jeu.jouer_coup(coup_choisi)
       # print(f"{self.name} a joué le coup : {coup_choisi}")