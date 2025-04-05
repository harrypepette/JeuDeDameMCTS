class Mouvement:
    def __init__(self, depart, arrivee):
        """
        Initialise un mouvement.
        :param depart: Position de départ (tuple (x, y)).
        :param arrivee: Position d'arrivée (tuple (x, y)).
        """
        self.depart = depart
        self.arrivee = arrivee