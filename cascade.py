import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

true_0 = np.load('true_0.npy')
pred_1 = np.load('pred_1.npy')
pred_2 = np.load('pred_2.npy')
pred_3 = np.load('pred_3.npy')
prob_1 = np.load('prob_1.npy')
prob_2 = np.load('prob_2.npy')
prob_3 = np.load('prob_3.npy')

#prob_1 = np.concatenate(prob_1)
#prob_2 = np.concatenate(prob_2)
#prob_3 = np.concatenate(prob_3)

ALREADY_GOOD_1 = [0, 2, 5, 7, 8, 20, 30, 35]
ALREADY_GOOD_2 = [0, 1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 30, 35, 21, 22, 24, 25, 29, 33]


data = np.stack((pred_1[:100], prob_1[:100], pred_2[:100], prob_2[:100], pred_3[:100], prob_3[:100], true_0[:100]))
df = pd.DataFrame(data)
df.to_csv("data.csv")

print data.shape
print "ACC0", float(np.sum(pred_1 == true_0)) / len(true_0)
exit()

#print repr(pred_1[:100])
#print repr(prob_1[:100])
#print repr(pred_2[:100])
#print repr(prob_2[:100])
#print repr(pred_3[:100])
#print repr(prob_3[:100])
#print repr(true_0[:100])
#exit()


max_acc = 0.
max_thr = -1.

for threshold in np.linspace(0.01, 0.99, 999):

  pred = []
  for idx in range(len(true_0)):
    if pred_1[idx] in ALREADY_GOOD_1:
      pred.append(pred_1[idx])
    elif prob_1[idx] >= threshold:
      pred.append(pred_1[idx])
    elif pred_2[idx] in ALREADY_GOOD_2:
      pred.append(pred_2[idx])
    elif pred_2[idx] >= threshold:
      pred.append(pred_2[idx])
    else:
      pred.append(pred_3[idx])

  pred = np.array(pred)
  acc = float(np.sum(pred == true_0)) / len(true_0)
  if acc > max_acc:
    max_acc = acc
    max_thr = threshold
  print "thresh = ", threshold, "acc = ", acc
  

print "MAX ACC = ", max_acc, " for THRESH = ", max_thr

#fpr, tpr, _ = roc_curve(true_0, pred)
#roc_auc = auc(fpr, tpr)
