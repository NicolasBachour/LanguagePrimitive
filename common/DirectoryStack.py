
# An utility class to explore files in the filesystem.
# - Created by Nicolas Bachour, 2017
# ------------------------ #

import os

class DirectoryStack:
    def __init__(self):
        self.stack = []
        self.pushd(os.getcwd())
        return

    def pushd(self, dir):
        os.chdir(dir)
        self.stack.append(os.path.abspath("./"))
        return
    
    def getd(self):
        return self.stack[len(self.stack) - 1]

    def length(self):
        return len(self.stack)

    def popd(self):
        dir = self.stack.pop()
        os.chdir(self.getd())
        return dir