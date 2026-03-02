import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import rich

from modules.evaluate_completely import compute_accuracy_keras_metrics, compute_precision_recall_f1



# Data
num_classes =                3
y_true = np.array(          [0, 1, 2, 1])       # Shape: (n,)
y_probabilities = np.array([[0.9, 0.1, 0.2],    # Shape: (n, num_classes)
                            [0.2, 0.8, 0.1],
                            [0.1, 0.3, 0.6],
                            [0.3, 0.3, 0.9]])   
# y_pred = np.array(        [0, 1, 2, 2])     # Shape: (n,)
y_pred = np.argmax(y_probabilities, axis=1)  # Convert probabilities to predicted class labels

# print these variables using rich
rich.print("Ground truth (y_true):", y_true)
rich.print("Predicted probabilities (y_probabilities):", y_probabilities)
rich.print("Predicted class labels (y_pred):", y_pred)

# Sparse Categorical Accuracy
accuracy = compute_accuracy_keras_metrics(y_true, y_probabilities)
print("Accuracy (after update state):", accuracy)

# Precision, Recall, F1-score
f1_metrics = compute_precision_recall_f1(y_true, y_pred, num_classes)
for class_id, metrics in f1_metrics.items():
    if class_id != "macro_avg" and class_id != "weighted_avg":
        print(f"Class {class_id}: Precision={metrics['precision']}, Recall={metrics['recall']}, F1-score={metrics['f1_score']}")

print("Macro Average:", f1_metrics["macro_avg"])
print("Weighted Average:", f1_metrics["weighted_avg"])

