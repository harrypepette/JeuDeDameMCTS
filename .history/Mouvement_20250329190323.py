class Mouvement:
    @staticmethod
    def deplacer_pion(plateau, case_depart, case_arrivee):
        """
        Déplace un pion d'une case à une autre.
        :param plateau: Le plateau de jeu.
        :param case_depart: Case de départ.
        :param case_arrivee: Case d'arrivée.
        """
        if case_depart.est_vide():
            raise ValueError("La case de départ est vide.")
        if not case_arrivee.est_vide():
            raise ValueError("La case d'arrivée est occupée.")
        
        case_arrivee.placer_pion(case_depart.pion)
        case_depart.retirer_pion()