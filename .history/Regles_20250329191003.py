class Regles:
    """
    Classe pour gérer les règles du jeu de dames.
    """

    @staticmethod
    def verifier_deplacement(case_depart, case_arrivee, joueur):
        """
        Vérifie si un déplacement est valide selon les règles du jeu.
        :param case_depart: Case de départ.
        :param case_arrivee: Case d'arrivée.
        :param joueur: Le joueur effectuant le déplacement.
        :return: True si le déplacement est valide, sinon False.
        """
        if case_depart.est_vide():
            return False  # La case de départ doit contenir un pion
        if not case_arrivee.est_vide():
            return False  # La case d'arrivée doit être vide

        pion = case_depart.pion

        # Vérification des règles de déplacement
        dx = abs(case_depart.x - case_arrivee.x)
        dy = abs(case_depart.y - case_arrivee.y)

        if pion.est_dame:
            # Les dames peuvent se déplacer en diagonale sur plusieurs cases
            return dx == dy
        else:
            # Les pions se déplacent d'une case en diagonale
            if dx != 1 or dy != 1:
                return False
            # Les pions ne peuvent avancer que dans une direction (selon leur couleur)
            direction = 1 if joueur.couleur == "blanc" else -1
            return (case_arrivee.y - case_depart.y) == direction

    @staticmethod
    def verifier_capture(case_depart, case_arrivee, plateau, joueur):
        """
        Vérifie si une capture est valide selon les règles du jeu.
        :param case_depart: Case de départ.
        :param case_arrivee: Case d'arrivée.
        :param plateau: Le plateau de jeu.
        :param joueur: Le joueur effectuant la capture.
        :return: True si la capture est valide, sinon False.
        """
        if case_depart.est_vide():
            return False  # La case de départ doit contenir un pion
        if not case_arrivee.est_vide():
            return False  # La case d'arrivée doit être vide

        pion = case_depart.pion

        # Vérification des règles de capture
        dx = abs(case_depart.x - case_arrivee.x)
        dy = abs(case_depart.y - case_arrivee.y)

        if dx != 2 or dy != 2:
            return False  # Une capture doit se faire en sautant une case en diagonale

        # Vérification de la présence d'un pion ennemi à capturer
        x_milieu = (case_depart.x + case_arrivee.x) // 2
        y_milieu = (case_depart.y + case_arrivee.y) // 2
        case_milieu = plateau.get_case(x_milieu, y_milieu)

        if case_milieu.est_vide():
            return False  # Il doit y avoir un pion à capturer
        if case_milieu.pion.couleur == joueur.couleur:
            return False  # Le pion à capturer doit appartenir à l'adversaire

        return True

    @staticmethod
    def verifier_victoire(joueur):
        """
        Vérifie si un joueur a gagné la partie.
        :param joueur: Le joueur à vérifier.
        :return: True si le joueur a gagné, sinon False.
        """
        return len(joueur.pions) == 0  # Le joueur gagne si l'adversaire n'a plus de pions