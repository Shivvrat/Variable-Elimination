def get_evidence(file_name):
    """
    Imports the evidence variable from the given file
    :param file_name: file which contains the evidence
    :return: evidence value as a array
    """
    try:
        with open(file_name) as FileObj:
            for lines in FileObj.readlines():
                if lines != '\n':
                    values = lines.split()
                    evidence = []
                    number_of_evidence_var = values[0]
                    count = 1
                    while count < values.__len__():
                        evidence.append(list(map(int, values[count: count + 2])))
                        count = count + 2
        return evidence
    except:
        print("The directory for the evidence file is not correct, please check.")
        exit()