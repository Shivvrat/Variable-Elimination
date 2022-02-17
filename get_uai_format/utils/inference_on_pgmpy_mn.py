import numpy as np
from pgmpy.models import MarkovNetwork
from potential_classes import BPotential, UPotential
from pgmpy.factors.discrete import DiscreteFactor




def create_factor(potential):
    data = np.exp(potential.data)
    if isinstance(potential, UPotential):
        v1 = str(potential.v)
        v2 = None
    elif isinstance(potential, BPotential):
        v1 = str(potential.v1)
        v2 = str(potential.v2)
    if v2:
        factor = DiscreteFactor([v1, v2], cardinality=[2, 2], values=data)
    elif not v2:
        factor = DiscreteFactor([v1], cardinality=[2], values=data)
    return factor


def create_pgmpy_MN(potentials, edge_list):
    edge_list = list(map(tuple, edge_list))
    charades_markov = MarkovNetwork(edge_list)
    for each_potential in potentials:
        this_factor = []
        if isinstance(each_potential, UPotential):
            this_factor = create_factor(each_potential)
        elif isinstance(each_potential, BPotential):
            this_factor = create_factor(each_potential)
        charades_markov.add_factors(this_factor)
    return charades_markov
