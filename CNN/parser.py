
import os
import collections
from utilities import DirectoryStack as utils

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
            data[filename] = file.readlines()
    dirstack.popd()
    return collections.OrderedDict(sorted(data.items()))
