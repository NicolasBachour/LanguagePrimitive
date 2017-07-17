
# Utility class to explore files

import os

class DirectoryStack:
    def __init__(self):
        self.stack = []
        self.pushd(os.getcwd())
        return

    def pushd(self, dir):
        os.chdir(dir)
        #DEBUG : print("--- Entering " + dir)
        self.stack.append(os.path.abspath("./"))
        return
    
    def getd(self):
        return self.stack[len(self.stack) - 1]

    def length(self):
        return len(self.stack)

    def popd(self):
        dir = self.stack.pop()
        #DEBUG : print("--- Returning to " + self.getd())
        os.chdir(self.getd())
        return dir