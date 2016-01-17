import algorithm_factory as factory

class Algorithm(object):

    def __init__(self, name, opts={}):
        self.name = name
        self.algorithm = factory.AlgorithmFactory.get(name)
