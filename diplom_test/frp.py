# this class implements the checker of the FRP (form of regulatory patterns)

class tripleEvaluator:
    # stability deviation:
    def __init__(self, curA, curB, curC, curK=0.01):
        self.a = curA
        self.b = curB
        self.c = curC
        self.k = curK
        self.d1 = abs(self.a - self.b)
        self.d2 = abs(self.b - self.c)
        self.h0 = abs(self.b) * self.k
        # conditions: [max,min,ascending,descending,stability]
        self.cMax = False
        self.cMin = False
        self.cAsc = False
        self.cDesc = False
        self.cStab = False

    def checkConditions(s):
        # if (((s.d1 > s.h0) and (s.d2 > s.h0))):
        if ((s.d1 < s.h0) and (s.d2 < s.h0)):
            s.cStab = True
            return
        else:
            if ((s.a < s.b) and (s.b > s.c)):
                s.cMax = True
                return
            if ((s.a > s.b) and (s.b < s.c)):
                s.cMin = True
                return
            if ((s.a < s.b) and (s.b < s.c) and (s.c - s.a) > s.h0):
                s.cAsc = True
                return
            if ((s.a > s.b) and (s.b > s.c) and (s.a - s.c) > s.h0):
                s.cDesc = True
                return
        '''
        if (d1>s.h0) and (s.d2<s.h0) ):
            if (s.a>s.b):
                s.cDesc = True
                return
            if (s.a<s.b):
                s.cAsc = True
                return
        if ( (s.d1<s.h0) and (s.d2>s.h0)):
            if (s.b<s.c):
                s.cAsc = True
                return
            if (s.b>s.c):
                s.cDesc = True
                return
            '''

    def consoleOutput(s):
        print("a=", s.a, "b=", s.b, "c=", s.c)
        print("Max: ", s.cMax)
        print("Min: ", s.cMin)
        print("Ascending: ", s.cAsc)
        print("Descending: ", s.cDesc)
        print("Stability: ", s.cStab)

    def getResults(s):
        return s.cMin, s.cMax, s.cAsc, s.cDesc, s.cStab


'''
A = tripleEvaluator(0.87041882, 0.77370561, 0.83391502)
A.checkConditions()
A.consoleOutput()
print(A.getResults()[3])
'''
