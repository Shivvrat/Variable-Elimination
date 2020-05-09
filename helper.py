from math import log10, exp
import numpy


def get_index_given_truth_values(variables, truth_values, cardinalities):
    """
    The function converts truth table tuples to array indices
    :param variables: The variable in factor
    :param truth_values: The values of given variables in the truth table (same order as variable)
    :param cardinalities: The cardinality of the given variables (same order as variable)
    :return:The index for the array
    """
    index = 0
    number = 0
    while number < len(variables):
        number_of_tuples_for_current_var = numpy.prod(cardinalities[number + 1:]) * truth_values[number]
        index = index + number_of_tuples_for_current_var
        number += 1
    return int(index)


def get_truth_values_given_index(variables, index_value, cardinalities):
    """
    Gives the truth value (values in tuple) for given index of the array
    @param variables: list containing all the variables in the graph which are in the factor
    @param index_value: Index value for which the tuple is to be founded
    @param cardinalities: Cardinalities of the given variables
    @return: the tuple for corresponding index value
    """
    number = 0
    truth_table_value = []
    while number < len(variables):
        truth_table_value.append(int(index_value // numpy.prod(cardinalities[number + 1:])))
        index_value = index_value - (index_value // numpy.prod(cardinalities[number + 1:])) * numpy.prod(
            cardinalities[number + 1:])
        number += 1
    return truth_table_value


def convert_to_log_space(distribution_array):
    new_distribution_array = []
    for each in distribution_array:
        if each is not list:
            each_new = log10(each)
        else:
            each_new = [(log10(each_value)) for each_value in each]
        new_distribution_array.append(each_new)
    return new_distribution_array


def convert_to_exponent_space(distribution_array):
    new_distribution_array = []
    for each in distribution_array:
        if each is not list:
            each_new = 10 ** each
        else:
            each_new = [10 ** each_value for each_value in each]
        new_distribution_array.append(each_new)
    return new_distribution_array


def compute_ordering(num_of_var, var_in_clique, evidence):
    """

    @param num_of_var: Number of variable for which the ordering is needed
    @param var_in_clique: the number of variables in each clique
    @param evidence: The evidence provided
    @return: variables according to their degree for elimination
    """
    min_degree_for_each_var = [0] * num_of_var
    evidence_var = [x[0] for x in evidence]
    for each in var_in_clique:
        each_without_evidence = each
        for each_var in each_without_evidence:
            min_degree_for_each_var[each_var] += len(each_without_evidence) - 1
    sorted_variable = numpy.argsort(min_degree_for_each_var)
    return min_degree_for_each_var, sorted_variable


def instantiate(num_of_var, evidence, cardinalities, var_in_clique, distribution_array):
    """
    Used to instantiate the factors given the evidence
    @param num_of_var: The number of variables in the graph
    @param evidence: The given evidence
    @param cardinalities: THe cardinalities of the variables
    @param var_in_clique: Variables in the clique
    @param distribution_array: The probability distribution array
    @return: The instantiated probability distribution table and updated variable list
    """
    for each in evidence:
        variable = each[0]
        value = each[1]
        var_in_clique = numpy.array(var_in_clique)
        cardinalities = numpy.array(cardinalities)
        for each_clique in range(len(var_in_clique)):
            if variable in var_in_clique[each_clique]:
                for each_tuple in range(len(distribution_array[each_clique])):
                    # Truth value is calculated for each tuple and then compared if it is equal to the evidence variable
                    truth_value = get_truth_values_given_index(var_in_clique[each_clique], each_tuple,
                                                               cardinalities[var_in_clique[each_clique]])
                    if truth_value[(list(var_in_clique[each_clique])).index(variable)] != value:
                        index_to_delete = get_index_given_truth_values(var_in_clique[each_clique], truth_value,
                                                                       cardinalities[var_in_clique[each_clique]])
                        # It the tuple does not match with the evidence then we ignore it
                        distribution_array[each_clique][index_to_delete] = -1
                var_in_clique = list(var_in_clique)
                val = list(var_in_clique[each_clique]).index(variable)
                # deleting the evidence variable from the array
                var_in_clique[each_clique] = numpy.ndarray.tolist(numpy.delete(var_in_clique[each_clique], val))
                num_of_var[each_clique] -= 1
                # removing the values that does not correspond with the evidence
                distribution_array[each_clique] = list(filter(lambda a: a != -1, distribution_array[each_clique]))
    return var_in_clique, distribution_array


def product_of_factors(factor_1, factor_2, var_in_factor_1, var_in_factor_2, cardinalities):
    """
    Takes two factors and returns the product of those two
    @param factor_1: The first factor
    @param factor_2: The second factor
    @param var_in_factor_1: tThe variables in factor 1
    @param var_in_factor_2: The variables in factor 2
    @param cardinalities: The cardinalities of the variables in facotrs
    @return: The product of the given two factors
    """
    var_in_output = []
    factor_1 = convert_to_log_space(factor_1)
    factor_2 = convert_to_log_space(factor_2)
    for each_var in var_in_factor_1:
        var_in_output.append(each_var)
    for each_var in var_in_factor_2:
        if each_var not in var_in_output:
            var_in_output.append(each_var)
    common_var = list(set(var_in_factor_1).intersection(set(var_in_factor_2)))
    new_factor = []
    cardinalities = numpy.array(cardinalities)
    for each_index_value_of_factor_1 in range(numpy.product(cardinalities[var_in_factor_1])):
        # Get the truth values for both the tuples
        truth_value_1 = numpy.array(
            get_truth_values_given_index(var_in_factor_1, each_index_value_of_factor_1, cardinalities[var_in_factor_1]))
        for each_index_value_of_factor_2 in range(numpy.product(cardinalities[var_in_factor_2])):
            truth_value_2 = numpy.array(get_truth_values_given_index(var_in_factor_2, each_index_value_of_factor_2,
                                                                     cardinalities[var_in_factor_2]))
            # Check if both the tuples are compatible or not, if yes then take their product
            if (truth_value_1[numpy.where(numpy.isin(var_in_factor_1, common_var))] == truth_value_2[
                numpy.where(numpy.isin(var_in_factor_2, common_var))]).all():
                new_factor.append(factor_1[each_index_value_of_factor_1] + factor_2[each_index_value_of_factor_2])
    new_factor = convert_to_exponent_space(new_factor)
    return new_factor, var_in_output


def sum_out(factor, set_of_variable_to_sum_out, var_in_factor, cardinalities):
    """
    Takes a factor and a variable to sum and returns a factor with other variables
    @param factor: The factor in which the sum will take place
    @param set_of_variable_to_sum_out: THe set of variables for which sum needs to be done
    @param var_in_factor: The variables in the factor/clique
    @param cardinalities: The cardinalities of the variables in the factor
    @return:
    """
    cardinalities = numpy.array(cardinalities)
    set_of_variable_to_sum_out = [set_of_variable_to_sum_out]
    var_in_final_factor = list(filter(lambda x: x not in set_of_variable_to_sum_out, var_in_factor))
    # Getting the size of new factor
    num_tuple_new_factor = numpy.product(numpy.array(cardinalities)[var_in_final_factor])
    new_factor = [0] * num_tuple_new_factor
    for each_tuple in range(len(factor)):
        truth_value = numpy.array(
            get_truth_values_given_index(var_in_factor, each_tuple, cardinalities[var_in_factor]))
        index_for_new_factor = get_index_given_truth_values(var_in_final_factor, truth_value[
            numpy.where(numpy.isin(var_in_factor, var_in_final_factor))], cardinalities[var_in_final_factor])
        # Adding value compatible with the variable
        new_factor[index_for_new_factor] = new_factor[index_for_new_factor] + factor[each_tuple]
    return new_factor, var_in_final_factor