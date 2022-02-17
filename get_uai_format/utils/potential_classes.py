import numpy as np

class UPotential:
    def __init__(self, v):
        self.v = v
        self.data = np.log(np.random.rand(2))

    def __init__(self, v, pot):
        self.v = v
        self.data = pot

    def getAllVar(self):
        return [self.v]

    def getVarPos(self, v):
        return 0

class BPotential:
    def __init__(self):
        self.v1 = -1
        self.v2 = -1
        self.data = np.log(np.random.random((2, 2)))

    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.data = np.log(np.random.random((2, 2)))

    def __init__(self, v1, v2, pot):
        self.v1 = v1
        self.v2 = v2
        self.data = np.array(pot)

    def getPotValue(self, var1, var2, i, j):
        i = int(i)
        j = int(j)
        if var1 == self.v1 and var2 == self.v2:
            return self.data[i, j]
        if var1 == self.v2 and var2 == self.v1:
            return self.data[j, i]
        raise ValueError('Wrong Potential Accessed')

    def reset(self):
        self.data = np.log(np.random.random((2, 2)))

    def getOtherVar(self, var):
        if var == self.v1:
            return self.v2
        if var == self.v2:
            return self.v1
        raise ValueError('Wrong Potential Accessed: Variable ' + str(var) + ' not present in Potential')

    def getAllVar(self):
        return [self.v1, self.v2]

    def getVarPos(self, v):
        if self.v1 == v:
            return 0
        else:
            return 1
