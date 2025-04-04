class Regles:
    @staticmethod
    def verifier_mouvement(case_depart, case_arrivee, joueur):
        """
        Vérifie si un mouvement est valide.
        :param case_depart: Case de départ.
        :param case_arrivee: Case d'arrivée.
        :param joueur: Joueur effectuant le mouvement.
        :return: True si le mouvement est valide, sinon False.
        """
        # Exemple de règle : déplacement en diagonale
        dx = abs(case_depart.x - case_arrivee.x)
        dy = abs(case_depart.y - case_arrivee.y)
        return dx == dy == 1