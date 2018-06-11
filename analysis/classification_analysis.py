import numpy as np
import matplotlib.pyplot as plt



votes = np.transpose(np.load('votes.npy'), (1,2,0))
labels = np.load('labels.npy')[:,0]

print votes[0][0]
exit()

# hit at least once
hit_once = 0
for idx in range(len(votes)):
  if labels[idx][0] in votes[idx]:
    hit_once = hit_once + 1
print "ACCURACY AT LEAST ONCE: ", float(hit_once) / len(votes)

# hit first
hit_first = 0
for idx in range(len(votes)):
  if labels[idx][0] == votes[idx][0]:
    hit_first = hit_first + 1
print "ACCURACY FIRST:         ", float(hit_first) / len(votes)

# hit most
def most_common(lst):
  counts = np.bincount(lst)
  return np.argmax(counts)
hit_most = 0
for idx in range(len(votes)):
  if most_common(votes[idx]) == labels[idx][0]:
    hit_most = hit_most + 1
print "ACCURACY MOST:          ", float(hit_most) / len(votes)
