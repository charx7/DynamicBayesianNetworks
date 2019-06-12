import sys
sys.path.append("../NhDBN") # Need to fix the imports
import unittest
from utils  import addMove 

import numpy as np

class TestSum(unittest.TestCase):
    def test_add(self):
        np.random.seed(42)
        dat = np.array([2,3,1])
        afterMove = addMove(dat, 10)
        print(afterMove)
        assert np.allclose(np.array([2, 3, 1, 10]), afterMove)

if __name__ == '__main__':
    unittest.main()
