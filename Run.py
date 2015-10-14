import numpy as np
from hub_toolbox.HubnessAnalysis import HubnessAnalysis, cosine_distance

vectors = np.random.rand(100,1000)
classes = np.random.rand(100)
classes[classes >= 0.5] = 2
classes[classes <  0.5] = 1
distances = cosine_distance(vectors)

vectors = 1
classes = 2
distances = 3


analysis = HubnessAnalysis(distances, classes, vectors)
analysis.analyse_hubness()