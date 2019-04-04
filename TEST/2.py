from collections import Counter

import numpy as np
import pandas as pd
x = [[2,222],[15,3],[15,4]]


# print(Counter(x[:,0]))
print(Counter(np.array(x)[:,0]).most_common()[0][0])
y= [1,2,3,4,5,6]
y = pd.Series(y)
y = (y/sum(y)*100).astype(int).tolist()
print(y)