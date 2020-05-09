from math import log10
import get_data_uai
import get_evidence_data
import helper


def variable_elimination(uai_file_name, evidence_file_name):
    """
    Main function used to run the variable elimination algorithm
    @param uai_file_name: The name of the main uai file
    @param evidence_file_name: THe name of the evidence file
    @return: The value of the partition function
    """
    num_of_var, cardinalities, num_of_cliques, num_of_var_in_clique, var_in_clique, distribution_array = get_data_uai.get_uai_data(
        uai_file_name)
    evidence = get_evidence_data.get_evidence(evidence_file_name)
    min_degree_for_each_var, sorted_variable = helper.compute_ordering(num_of_var, var_in_clique, evidence)
    var_in_clique, distribution_array = helper.instantiate(num_of_var_in_clique, evidence, cardinalities, var_in_clique,
                                                           distribution_array)
    for each_var in sorted_variable:
        each_clique = 0
        while each_clique < len(var_in_clique):
            each_clique_2 = each_clique + 1
            while each_clique_2 < len(var_in_clique):
                # For each clique check if they have common variables (the var to be eliminated)
                if var_in_clique[each_clique] is not None and var_in_clique[each_clique].__len__() != 0 and \
                        var_in_clique[each_clique_2] is not None and var_in_clique[
                    each_clique_2].__len__() != 0 and each_var in var_in_clique[each_clique] and each_var in \
                        var_in_clique[each_clique_2]:
                    # Take the product if variables are common
                    new_factor, var_in_output = helper.product_of_factors(distribution_array[each_clique],
                                                                          distribution_array[each_clique_2],
                                                                          var_in_clique[each_clique],
                                                                          var_in_clique[each_clique_2], cardinalities)
                    distribution_array[each_clique_2] = None
                    # Update the probability array and variables array
                    distribution_array[each_clique] = new_factor
                    var_in_clique[each_clique] = var_in_output
                    var_in_clique[each_clique_2] = None
                    num_of_var_in_clique[each_clique_2] = 0
                    num_of_var_in_clique[each_clique] = len(var_in_output)
                    num_of_cliques -= 1
                each_clique_2 += 1
            if var_in_clique[each_clique] is not None and var_in_clique[each_clique].__len__() != 0 and each_var in \
                    var_in_clique[each_clique]:
                # If the var to be eliminated is in the clique then sum it out
                new_factor, var_in_output = helper.sum_out(distribution_array[each_clique], each_var,
                                                           var_in_clique[each_clique], cardinalities)
                # Update the distribution
                distribution_array[each_clique] = new_factor
                var_in_clique[each_clique] = var_in_output
            each_clique += 1
    z = 1
    # Take product of all values to get the partition function
    for each_clique in distribution_array:
        if each_clique is not None:
            for each_val in each_clique:
                if each_val != 0:
                    z *= each_val
    # Taking the logarithm of partition function
    ans = log10(z)
    return ans


if __name__ == "__main__":
    print(variable_elimination("homework2_files/BN_69.uai", "homework2_files/BN_69.uai.evid"))