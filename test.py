from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
c = np.array([1,2,3])
d = np.array([4,5,6])
b = cosine_similarity([c,d])[0][1]
print(b)
