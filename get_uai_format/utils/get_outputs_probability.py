import argparse
import itertools
import pickle
import sys
import numpy as np
from pgmpy.inference import VariableElimination
from loguru import logger

# from inference_on_pomegranate_mn import create_MN, get_prob


import os

# from inference_evidence_op_cnn_and_true_label.src.models.inference_on_pomegranate_mn import get_prob, \
#     create_pomegranate_MN
from inference_on_pgmpy_mn import create_pgmpy_MN, get_pgmpy_prob
from create_ILP_solver import ILP_Solver, import_potentials, import_edge_list
from evaluate_model import calculate_evaluation_metrics, print_and_save_eval_metrics
from config import get_cfg_defaults

# logger.info(base_path)
def combine_cfgs(args):
    # Priority 3: get default configs
    cfg_base = get_cfg_defaults()

    # Priority 2: merge from yaml config
    # if yaml_path is not None and Path(yaml_path).exists():
    #     cfg_base.merge_from_file(yaml_path)

    # Priority 1: merge from .env
    list_of_args = list(vars(args).items())
    cfg_base.merge_from_list(list(itertools.chain(*list_of_args)))
    # load_dotenv(find_dotenv(), verbose=True) # Load .env

    # Load variables

    return cfg_base


def parse(argv):
    """
    python inference_test.py --test_csv /home/sxa180157/ptg/tacos-PTG-less-data/pytorch_cnn/prepare_data/labels_as_text/true_labels/test/test_true_labels_with_image_name.csv \
        --checkpoint_location ../outputs/model_10_retrained_whole_True.pth \
        --output_file_name ../outputs/output_cnn_test_set_10_retrained_whole_True.csv &
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--POTENTIAL.FILE_LOCATION',
                        type=str,
                        help='Location of the potential file',
                        default='./temp/data/processed/train_data_0.5_num_iter_100000/')
    parser.add_argument('--TEST_SET_CNN.PREDICTIONS_AND_TRUE_LABELS_LOCATION',
                        type=str,
                        help='Test Set CNN predictions location',
                        default='./temp/data/external/output_of_SlowFast/Charades/validation/')
    parser.add_argument('--TEST_SET_CNN.PREDICTIONS_AND_TRUE_LABELS_NAME',
                        type=str,
                        help='Test Set CNN predictions Name',
                        default='validation.pkl')
    parser.add_argument('--POTENTIAL.FILE_NAME',
                        type=str,
                        help='Name of the potential file',
                        default='markov.train.data-potential-2')
    parser.add_argument('--EDGE_LIST.FILE_LOCATION',
                        type=str,
                        help='Location of the edge list file',
                        default='./temp/data/processed/train_data_0.5_num_iter_100000/markov.train.data-struct-2')

    args = parser.parse_args(argv)
    return combine_cfgs(args)


def main(args, logger=logger):
    if not logger:
        logger.add("./logger/logger_{time}.log", format="{time} {level} {message}")
        logger.info("The arguments given to run inference_evidence_op_cnn_and_true_label on trained training model are :-")
        # logger.info(pprint(convert_to_dict(cfg)))
    import os
    potential_file_location = os.path.join(os.path.abspath(args.POTENTIAL.FILE_LOCATION), args.POTENTIAL.FILE_NAME)
    edge_list_file_location = args.EDGE_LIST.FILE_LOCATION
    potentials = import_potentials(potential_file_location)
    edge_list = import_edge_list(edge_list_file_location)
    # pomegranate_MN = create_pomegranate_MN(potentials)
    pgmpy_mn = create_pgmpy_MN(potentials, edge_list)
    threshold = float(args.THRESHOLD)
    from pgmpy.inference import BeliefPropagation
    # BeliefPropagation
    # inference_module = BeliefPropagation(pgmpy_mn)
    # inference_module.calibrate()

    # VariableElimination
    inference_module = VariableElimination(pgmpy_mn)
    output_folder = args.POTENTIAL.FILE_LOCATION.split("/")[-3:]
    output_folder = args.OUTPUT_PATH + '/'.join(map(str, output_folder))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # How to predict probability using loopy belief propagation given evidence
    # print(pomegranate_MN.predict_proba([[1] * 312 + [None] * 2]))
    test_data = np.genfromtxt(args.TEST_SET_CNN.PREDICTIONS_AND_TRUE_LABELS_LOCATION, delimiter=",")
    logger.info(test_data.shape)
    test_actual_labels, test_CNN_predictions = test_data[:, :test_data.shape[1]//2], test_data[:, test_data.shape[1]//2:]
    # test_CNN_predictions = pd.read_csv(test_CNN_predictions_location, sep=",", header=None)
    test_actual_labels = test_actual_labels.astype('int')
    final_predictions = []
    predicted_probabilities = []
    # index = 0
    num_true_labels = test_CNN_predictions.shape[1]
    for index, row in enumerate(test_CNN_predictions):
        if index + 1 % 500:
            logger.info("Done with index {0}".format(str(index + 1)))
        row[row > threshold] = 1
        row[row <= threshold] = 0
        row = row.astype('int')
        initialization = {key: row[key - num_true_labels] for key in list(range(num_true_labels, 2 * num_true_labels))}
        solver = ILP_Solver(name="Markov", potentials=potentials, initialization=initialization,
                            num_var=num_true_labels, logger=logger)
        final_value = solver.create_solver()
        final_prob = get_pgmpy_prob(final_value, initialization=initialization, markov_model_inference=inference_module)
        # final_prob = get_prob(final_value, initialization=initialization, markov_model=pomegranate_MN)
        print(final_prob)
        print(final_value)
        predicted_probabilities.append(final_prob)
        final_predictions.append(final_value)
        # logger.info("Done with index {0}".format(str(index)))
    # final_value = create_solver("Markov", potentials, {1: 1}, 8)
    # File name - training.train.data-potential-10
    np.save(output_folder + 'output_ILP_neighbours_' + args.POTENTIAL.FILE_NAME.split("-")[-1],
            np.asarray(final_predictions))
    np.save(output_folder + 'output_probability_neighbours_' + args.POTENTIAL.FILE_NAME.split("-")[-1],
            np.asarray(predicted_probabilities))
    import os
    test_predictions_ILP = np.asarray(predicted_probabilities)
    hammingLoss, coverage, ranking_loss, average_precision, subset_accuracy, jaccard_score_val, label_ranking_average_precision_score \
        = calculate_evaluation_metrics(test_actual_labels, test_predictions_ILP)
    # timestr = time.strftime("%Y%m%d-%H%M%S")
    path = output_folder + 'accuracy_metrics/'
    if not os.path.exists(path):
        os.makedirs(path)
    output_filename = path + "ILP_results_neighbours_" + args.POTENTIAL.FILE_NAME.split("-")[-1] + ".txt"
    open(output_filename, "w").close()
    print_and_save_eval_metrics(hammingLoss, coverage, ranking_loss, average_precision, subset_accuracy,
                                jaccard_score_val, label_ranking_average_precision_score, logger, output_filename)

    # with open(test_CNN_predictions_location, 'r') as dest_f:
    #     data_iter = csv.reader(dest_f,
    #                            delimiter=',', quotechar=r'"'
    #                            )
    #     data = [data for data in data_iter]
    # data = np.asarray(data)[:, 1:]
    # test_CNN_predictions = data[:, 1:].astype(float)

    # potentials = [BPotential(0, 1, [[-0.47321777, 0.61765053], [1.11786674, -1.26229952]]),
    #               BPotential(0, 5, [[-0.3245857, 0.46901847], [0.28686378, -0.43129656]]),
    #               BPotential(6, 0, [[0.02752234, 0.11691043], [0.23903417, -0.38346694]]),
    #               BPotential(2, 3, [[-0.47321777, 0.61765053], [1.11786674, -1.26229952]]),
    #               BPotential(2, 4, [[-0.3245857, 0.46901847], [0.28686378, -0.43129656]]),
    #               BPotential(7, 1, [[0.02752234, 0.11691043], [0.23903417, -0.38346694]]),
    #               BPotential(2, 1, [[-0.47321777, 0.61765053], [1.11786674, -1.26229952]]),
    #               BPotential(3, 5, [[-0.3245857, 0.46901847], [0.28686378, -0.43129656]]),
    #               BPotential(6, 0, [[0.02752234, 0.11691043], [0.23903417, -0.38346694]])]


if __name__ == '__main__':
    main(parse(sys.argv[1:]))
