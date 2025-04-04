class Regles:
    @staticmethod
    def mouvement_valide(pion, mouvement, plateau):
        """
        Vérifie si un mouvement est valide.
        """
        dx = mouvement.arrivee[0] - mouvement.depart[0]
        dy = mouvement.arrivee[1] - mouvement.depart[1]

        # Vérifier si le pion doit manger
        if plateau.peut_manger(pion):
            if abs(dx) == 2 and abs(dy) == 2:  # Saut
                milieu = ((mouvement.depart[0] + mouvement.arrivee[0]) // 2,
                          (mouvement.depart[1] + mouvement.arrivee[1]) // 2)
                pion_milieu = plateau.get_pion(*milieu)
                return pion_milieu and pion_milieu.couleur != pion.couleur and plateau.get_pion(*mouvement.arrivee) is None
            return False  # Mouvement simple interdit si une capture est possible

        # Mouvement simple
        if abs(dx) == 1 and abs(dy) == 1:
            return plateau.get_pion(*mouvement.arrivee) is None

        # Saut par-dessus un pion
        if abs(dx) == 2 and abs(dy) == 2:
            milieu = ((mouvement.depart[0] + mouvement.arrivee[0]) // 2,
                      (mouvement.depart[1] + mouvement.arrivee[1]) // 2)
            pion_milieu = plateau.get_pion(*milieu)
            return pion_milieu and pion_milieu.couleur != pion.couleur and plateau.get_pion(*mouvement.arrivee) is None

        return False
