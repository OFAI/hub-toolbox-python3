import numpy as np
import time

tic = time.clock()
cnt = 0
while True:
    cnt += 1
    if cnt % 1e7 == 0:
        print("Random numbers so far: {}".format(cnt))
    r = np.random.random()
    if r == 0:
        toc = time.clock() - tic
        print("Finally, we found 0 after {} trials (took {} ms).".format(r, toc))
        break
    
