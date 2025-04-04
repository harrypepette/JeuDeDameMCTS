class Mouvement:
    """
    Classe pour gérer les mouvements des pions et des dames dans le jeu de dames.
    """

    @staticmethod
    def deplacer_pion(plateau, case_depart, case_arrivee):
        """
        Déplace un pion d'une case à une autre.
        :param plateau: Le plateau de jeu.
        :param case_depart: Coordonnées de la case de départ (x, y).
        :param case_arrivee: Coordonnées de la case d'arrivée (x, y).
        :raises ValueError: Si le mouvement est invalide.
        """
        case_d = plateau.get_case(case_depart[0], case_depart[1])
        case_a = plateau.get_case(case_arrivee[0], case_arrivee[1])

        if case_d.est_vide():
            raise ValueError("La case de départ est vide.")
        if not case_a.est_vide():
            raise ValueError("La case d'arrivée est occupée.")

        pion = case_d.pion

        # Vérification des règles de déplacement
        dx = abs(case_depart[0] - case_arrivee[0])
        dy = abs(case_depart[1] - case_arrivee[1])

        if pion.est_dame:
            # Déplacement spécifique pour une dame (diagonale illimitée)
            if dx != dy:
                raise ValueError("Les dames doivent se déplacer en diagonale.")
        else:
            # Déplacement spécifique pour un pion (diagonale de 1 case)
            if dx != 1 or dy != 1:
                raise ValueError("Les pions doivent se déplacer d'une case en diagonale.")

        # Déplacement effectif
        case_a.placer_pion(pion)
        case_d.retirer_pion()

    @staticmethod
    def capturer_pion(plateau, case_depart, case_arrivee):
        """
        Gère la capture d'un pion ennemi.
        :param plateau: Le plateau de jeu.
        :param case_depart: Coordonnées de la case de départ (x, y).
        :param case_arrivee: Coordonnées de la case d'arrivée (x, y).
        :raises ValueError: Si la capture est invalide.
        """
        case_d = plateau.get_case(case_depart[0], case_depart[1])
        case_a = plateau.get_case(case_arrivee[0], case_arrivee[1])

        if case_d.est_vide():
            raise ValueError("La case de départ est vide.")
        if not case_a.est_vide():
            raise ValueError("La case d'arrivée est occupée.")

        pion = case_d.pion

        # Vérification des règles de capture
        dx = abs(case_depart[0] - case_arrivee[0])
        dy = abs(case_depart[1] - case_arrivee[1])

        if dx != 2 or dy != 2:
            raise ValueError("La capture doit se faire en sautant une case en diagonale.")

        # Vérification de la présence d'un pion ennemi à capturer
        x_milieu = (case_depart[0] + case_arrivee[0]) // 2
        y_milieu = (case_depart[1] + case_arrivee[1]) // 2
        case_milieu = plateau.get_case(x_milieu, y_milieu)

        if case_milieu.est_vide() or case_milieu.pion.couleur == pion.couleur:
            raise ValueError("Aucun pion ennemi à capturer.")

        # Capture effective
        case_a.placer_pion(pion)
        case_d.retirer_pion()
        case_milieu.retirer_pion()