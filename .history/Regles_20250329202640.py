class Regles:
    @staticmethod
    def mouvement_valide(pion, mouvement, plateau):
        """
        Vérifie si un mouvement est valide.
        """
        dx = mouvement.arrivee[0] - mouvement.depart[0]
        dy = mouvement.arrivee[1] - mouvement.depart[1]

        # Vérifier si le pion est une dame
        if pion.est_dame:
            directions = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
            if (dx, dy) not in [(d[0] * i, d[1] * i) for d in directions for i in range(1, 8)]:
                return False  # Mouvement non valide pour une dame

            x, y = mouvement.depart
            step_x = dx // abs(dx) if dx != 0 else 0
            step_y = dy // abs(dy) if dy != 0 else 0
            x += step_x
            y += step_y
            while (x, y) != mouvement.arrivee:
                if plateau.get_pion(x, y) is not None:
                    return False  # Chemin bloqué
                x += step_x
                y += step_y
            return True

        # Vérifier si le pion doit manger
        if plateau.peut_manger(pion):
            if abs(dx) == 2 and abs(dy) == 2:  # Saut
                milieu = ((mouvement.depart[0] + mouvement.arrivee[0]) // 2,
                          (mouvement.depart[1] + mouvement.arrivee[1]) // 2)
                pion_milieu = plateau.get_pion(*milieu)
                return (pion_milieu and pion_milieu.couleur != pion.couleur and
                        plateau.get_pion(*mouvement.arrivee) is None and
                        pion.mouvement_valide_direction(dx, dy, est_capture=True))
            return False  # Mouvement simple interdit si une capture est possible

        # Mouvement simple
        if abs(dx) == 1 and abs(dy) == 1:
            return (plateau.get_pion(*mouvement.arrivee) is None and
                    pion.mouvement_valide_direction(dx, dy, est_capture=False))

        # Saut par-dessus un pion
        if abs(dx) == 2 and abs(dy) == 2:
            milieu = ((mouvement.depart[0] + mouvement.arrivee[0]) // 2,
                      (mouvement.depart[1] + mouvement.arrivee[1]) // 2)
            pion_milieu = plateau.get_pion(*milieu)
            return (pion_milieu and pion_milieu.couleur != pion.couleur and
                    plateau.get_pion(*mouvement.arrivee) is None and
                    pion.mouvement_valide_direction(dx, dy, est_capture=True))

        return False
