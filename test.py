import sys
import numpy as np
from ast import literal_eval as make_tuple

if __name__ == "__main__":
    

    with open(sys.argv[1], 'r') as f:
        for line in f:
            line = make_tuple(line)
            x = np.array(line[1])
            print(line[0], x.shape)