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
            # Vérifier si le mouvement est strictement diagonal
            if abs(dx) != abs(dy):
                print(f"Mouvement non diagonal pour la dame: {mouvement.depart} -> {mouvement.arrivee}")
                return False  # Mouvement non diagonal
                
            # Vérifier que le chemin est libre (sauf pour la capture)
            x, y = mouvement.depart
            step_x = dx // abs(dx) if dx != 0 else 0
            step_y = dy // abs(dy) if dy != 0 else 0
            
            # Compter les pions sur le chemin
            pions_sur_chemin = 0
            pion_adverse_trouve = False
            
            x += step_x
            y += step_y
            while (x, y) != mouvement.arrivee:
                pion_sur_case = plateau.get_pion(x, y)
                if pion_sur_case:
                    pions_sur_chemin += 1
                    if pion_sur_case.couleur != pion.couleur:
                        pion_adverse_trouve = True
                    else:
                        return False  # Propre pion sur le chemin = mouvement impossible
                x += step_x
                y += step_y
            
            # Pour une capture, il doit y avoir exactement un pion adverse sur le chemin
            # Pour un déplacement simple, le chemin doit être entièrement libre
            if peut_manger:
                return pions_sur_chemin == 1 and pion_adverse_trouve
            else:
                return pions_sur_chemin == 0

        # Logique pour un pion normal (non modifiée pour l'instant)
        if peut_manger:
            if abs(dx) == 2 and abs(dy) == 2:  # Saut
                milieu = ((mouvement.depart[0] + mouvement.arrivee[0]) // 2,
                        (mouvement.depart[1] + mouvement.arrivee[1]) // 2)
                pion_milieu = plateau.get_pion(*milieu)
                return (pion_milieu and pion_milieu.couleur != pion.couleur and
                    plateau.get_pion(*mouvement.arrivee) is None and
                        pion.mouvement_valide_direction(dx, dy, est_capture=True))
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