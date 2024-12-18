
class SatInterface():
    def __init__(self):
        self.max_variables = 225
        self.max_clauses = 960
        self.encoding_size = 2 * self.max_variables * self.max_clauses
        self.input_shape = (self.encoding_size, 1)
