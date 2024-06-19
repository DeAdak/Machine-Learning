import pandas as pd
import numpy as np

######################### Classification Metrics ######################

predicted = np.array(["Y","Y","N","N","Y","N","Y","Y","N","N","N","N","Y","N"], 
                     dtype=object)

existing = np.array(["Y","N","N","N","Y","N","Y","N","Y","Y","Y","N","Y","N"], 
                     dtype=object)

eval_df = pd.DataFrame({'existing':existing, 'predicted':predicted})

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(existing, predicted))
print(classification_report(existing, predicted))
print(accuracy_score(existing,predicted))



##########################################################################

comp = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_\1. ML Basics/comp_prob.csv")

# Import necessary modules
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob_1 = comp["yprob_2"]
y_test = comp["y_test"]

# Generate ROC curve values: fpr, tpr, thresholds
m1spec, sens, thresholds = roc_curve(y_test, y_pred_prob_1)

# Plot ROC curve 
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(m1spec, sens)
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curve')
plt.show()
roc_auc_score(y_test, y_pred_prob_1)


from sklearn.metrics import log_loss
y_pred_prob_1 = comp["yprob_1"]
log_loss(y_test, y_pred_prob_1)
y_pred_prob_1 = comp["yprob_2"]
log_loss(y_test, y_pred_prob_1)

############### Regression Metrics #######################

y_pred = np.array([13.4,45.4,89.3,90.4,87.3,45.9,16.5])
y_true = np.array([12.3,46.4,90,100.4,86.3,46,17])
from sklearn.metrics import mean_squared_error
mean_squared_error(y_true, y_pred)  


y_pred = np.array([13.4,45.4,89.3,90.4,87.3,45.9,16.5])
y_true = np.array([12.3,46.4,90,100.4,86.3,46,17])
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_true, y_pred)  

y_pred = np.array([13.4,45.4,89.3,90.4,87.3,45.9,16.5])
y_true = np.array([12.3,46.4,90,100.4,86.3,46,17])
from sklearn.metrics import r2_score
r2_score(y_true, y_pred)  
