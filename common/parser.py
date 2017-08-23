
# A simple class that builds python dictionnaries containing the directory structure of a given folder #
# ------------------------ #

import os
import collections
import DirectoryStack as utils

def load_dataset(folder):
    dirstack = utils.DirectoryStack()
    data = load_recursive(folder, dirstack)
    return data

def load_recursive(folder, dirstack):
    data = dict()
    dirstack.pushd(folder)
    for filename in os.listdir("./"):
        if os.path.isdir(filename):
            subdir_data = load_recursive(filename, dirstack)
            data[filename] = subdir_data
        else:
            file = open(filename, 'r')
            data[filename] = []
            lines = file.readlines()
            lines = [x.strip() for x in lines]
            for line in lines:
                data[filename].append(line.split())
    dirstack.popd()
    return collections.OrderedDict(sorted(data.items()))