import os
import pickle
from loguru import logger

test_CNN_predictions_and_true_labels_location = os.path.join("./train_data/output_of_SlowFast/Charades/validation/", "validation.pkl")
# test_CNN_predictions = pd.read_csv(test_CNN_predictions_location, sep=",", header=None)
f = open(test_CNN_predictions_and_true_labels_location, 'rb')
train_predictions_and_actual_output = pickle.load(f)
f.close()
test_CNN_predictions, test_actual_labels = train_predictions_and_actual_output[0].numpy(), \
                                           train_predictions_and_actual_output[1].numpy()

output_lines = []
threshold = 0.3
num_true_var = test_CNN_predictions.shape[1]
num_samples = test_CNN_predictions.shape[0]
output_lines.append([str(num_samples)])
for index, row in enumerate(test_CNN_predictions):
    this_line = [str(num_true_var)]
    if index + 1 % 500:
        logger.info("Done with index {0}".format(str(index + 1)))
    row[row > threshold] = 1
    row[row <= threshold] = 0
    row = row.astype('int')
    for index, val in enumerate(row):
        this_line.append(str(num_true_var + index))
        this_line.append(str(val))
    output_lines.append(this_line)

with open("MN-max-neighbour-5.uai.evid", "w") as f:
    for row in output_lines:
        f.write("%s\n" % ' '.join(str(col) for col in row))