
class NNO():
    def __init__(self,
                 dimensions: int,
                 tau: float
                 ):
        self.dimensions = dimensions
        self.tau = tau


    def fit(self, prototypes, support):
