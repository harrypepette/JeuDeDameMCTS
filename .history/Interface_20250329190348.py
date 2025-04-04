class Interface:
    @staticmethod
    def afficher_plateau(plateau):
        """Affiche le plateau."""
        plateau.afficher()

    @staticmethod
    def demander_mouvement():
        """Demande au joueur de saisir un mouvement."""
        depart = input("Entrez la case de départ (x, y) : ")
        arrivee = input("Entrez la case d'arrivée (x, y) : ")
        return tuple(map(int, depart.split(','))), tuple(map(int, arrivee.split(',')))