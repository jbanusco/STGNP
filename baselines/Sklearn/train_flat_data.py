
from model.train_stgnp import Objective_Multiplex


class Objective_Flat(Objective_Multiplex):
    def __init__(self, *args, class_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_dim = class_dim
