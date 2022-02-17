import pandas as pd
from utils.inference_on_pgmpy_mn import create_pgmpy_MN
from utils.potential_classes import BPotential, UPotential
from pgmpy.readwrite import UAIWriter

def import_edge_list(edge_list_file_path):
    from numpy import genfromtxt
    edge_list = genfromtxt(edge_list_file_path, delimiter=" ", skip_header=True, dtype=str)
    return edge_list


def import_potentials(potential_file_path):
    data_frame = pd.read_pickle(potential_file_path)
    # logger.info(data_frame)
    potentials = []
    for index, row in pd.DataFrame.iterrows(data_frame):
        this_BPotential = BPotential(row['v1'], row['v2'], np.exp(row['data']))
        potentials.append(this_BPotential)
    return potentials


max_neighbour = 5
potentials = import_potentials("./train_data/threshold_0.3/num-iter_40000/MN-potential-" + str(5))
edge_list = import_edge_list("./train_data/threshold_0.3/num-iter_40000/MN-struct-" + str(5))
mn = create_pgmpy_MN(potentials, edge_list)
writer = UAIWriter(mn)
writer.write_uai("MN-max-neighbour-"+str(max_neighbour)+".uai")