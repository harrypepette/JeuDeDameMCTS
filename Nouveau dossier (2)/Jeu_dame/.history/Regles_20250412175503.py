class Regles:
    @staticmethod
    def mouvement_valide(pion, mouvement, plateau):
        """
        Vérifie si un mouvement est valide.
        """
        dx = mouvement.arrivee[0] - mouvement.depart[0]
        dy = mouvement.arrivee[1] - mouvement.depart[1]

    # Vérifier si une capture est possible pour ce pion
        peut_manger = plateau.peut_manger(pion)

        if pion.est_dame:
            directions = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        # Vérifier si le mouvement est diagonal
            if (dx, dy) not in [(d[0] * i, d[1] * i) for d in directions for i in range(1, 8)]:
              #!  print(f"Mouvement non diagonal pour la dame : ({dx}, {dy})")
                return False  # Mouvement non diagonal

        # Si une capture est possible, forcer la capture
            if peut_manger:
                cases_a_manger = pion.check_manger(plateau)
                if (mouvement.arrivee[0], mouvement.arrivee[1]) not in cases_a_manger:
                    print(f"Capture obligatoire pour ce pion à {pion.position}, mouvement {mouvement.arrivee} refusé")
                    return False  # Mouvement invalide : une capture est obligatoire

        # Vérifier que le chemin est libre pour un mouvement simple
            if not peut_manger:
                x, y = mouvement.depart
                step_x = dx // abs(dx) if dx != 0 else 0
                step_y = dy // abs(dy) if dy != 0 else 0
                x += step_x
                y += step_y
                while (x, y) != mouvement.arrivee:
                    if plateau.get_pion(x, y) is not None:
                        print(f"Chemin bloqué pour la dame à ({x}, {y})")
                        return False  # Chemin bloqué
                    x += step_x
                    y += step_y
                return plateau.get_pion(*mouvement.arrivee) is None  # La case d'arrivée doit être vide

            return True  # Capture déjà validée

    # Logique pour un pion normal (non modifiée pour l'instant)
        if peut_manger:
            if abs(dx) == 2 and abs(dy) == 2:  # Saut
                milieu = ((mouvement.depart[0] + mouvement.arrivee[0]) // 2,
                        (mouvement.depart[1] + mouvement.arrivee[1]) // 2)
                pion_milieu = plateau.get_pion(*milieu)
                return (pion_milieu and pion_milieu.couleur != pion.couleur and
                    plateau.get_pion(*mouvement.arrivee) is None and
                        pion.mouvement_valide_direction(dx, dy, est_capture=True))
            print(f"Capture obligatoire pour ce pion à {pion.position}, mouvement simple {mouvement.arrivee} refusé")
            return False  # Mouvement simple interdit si une capture est possible

        if abs(dx) == 1 and abs(dy) == 1:
            return (plateau.get_pion(*mouvement.arrivee) is None and
                    pion.mouvement_valide_direction(dx, dy, est_capture=False))

        if abs(dx) == 2 and abs(dy) == 2:
            milieu = ((mouvement.depart[0] + mouvement.arrivee[0]) // 2,
                    (mouvement.depart[1] + mouvement.arrivee[1]) // 2)
            pion_milieu = plateau.get_pion(*milieu)
            return (pion_milieu and pion_milieu.couleur != pion.couleur and
                    plateau.get_pion(*mouvement.arrivee) is None and
                    pion.mouvement_valide_direction(dx, dy, est_capture=True))

        return False
