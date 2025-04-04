from Pion import Pion

class Dame(Pion):
    def __init__(self, couleur, position):
        super().__init__(couleur, position)
        self.est_dame = True
