import variable_elimination
import sys
import warnings
warnings.filterwarnings("ignore")
arguments = list(sys.argv)

try:
    # Please give the actual directory of the files
    uai_file_name = str(arguments[1])
    evidence_file_name = str(arguments[2])
except:
    print("You have given less arguments")
    print("The code to run the algorithm :-")
    print("python main.py <uai_file_name> <evidence_file_name>")


def main():
    """
    This is the main function which is used to run all the algorithms
    :return:
    """
    partition_function = variable_elimination.variable_elimination(uai_file_name, evidence_file_name)
    print("The partition function for the given dataset is ", partition_function)

if __name__ == "__main__":
    main()