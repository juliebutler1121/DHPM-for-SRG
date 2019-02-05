def write (filename, flowparams, Hs):
    F = open(filename, 'w')

    for s, hs in zip(flowparams, Hs):
        to_file = ""
        to_file = to_file + str(s) + "\n"
        for row in hs:
            for element in row:
                to_file = to_file + str(element) + ","
            to_file = to_file + "\n"

        to_file = to_file + "\n"
        F.write (to_file)
