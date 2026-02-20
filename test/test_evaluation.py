import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pytest
from sklearn.metrics import precision_recall_fscore_support

from modules.evaluate import compute_precision_recall_f1

# -------------------------------------------------------
# Helper to compute sklearn reference values
# -------------------------------------------------------

def sklearn_reference(y_true, y_pred, num_classes):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0
    )

    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average="weighted",
        zero_division=0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "macro": (macro_precision, macro_recall, macro_f1),
        "weighted": (weighted_precision, weighted_recall, weighted_f1),
    }


# =======================================================
# TEST 1 — Imbalanced + structured confusion
# =======================================================

def test_7class_heavy_confusion_pattern():

    num_classes = 7

    # 60 samples, heavy imbalance
    y_true = np.array(
        [0]*20 +    # dominant
        [1]*10 +
        [2]*8 +
        [3]*7 +
        [4]*6 +
        [5]*5 +
        [6]*4
    )

    # Structured mistakes:
    # - class 0 sometimes predicted as 1
    # - class 3 entirely predicted as 2
    # - class 6 predicted randomly
    y_pred = np.array(
        [0]*15 + [1]*5 +            # class 0
        [1]*8 + [2]*2 +             # class 1
        [2]*7 + [3]*1 +             # class 2
        [2]*7 +                     # class 3 (completely wrong)
        [4]*5 + [0]*1 +             # class 4
        [5]*4 + [6]*1 +             # class 5
        [0,1,2,3]                   # class 6 chaotic
    )

    
    result = compute_precision_recall_f1(y_true, y_pred, num_classes)

    ref = sklearn_reference(y_true, y_pred, num_classes)

    # Per-class checks
    for c in range(num_classes):
        assert np.isclose(result[c]["precision"], ref["precision"][c])
        assert np.isclose(result[c]["recall"], ref["recall"][c])
        assert np.isclose(result[c]["f1_score"], ref["f1"][c])
        assert result[c]["support"] == ref["support"][c]

    # Macro
    assert np.isclose(result["macro_avg"]["precision"], ref["macro"][0])
    assert np.isclose(result["macro_avg"]["recall"], ref["macro"][1])
    assert np.isclose(result["macro_avg"]["f1_score"], ref["macro"][2])

    # Weighted
    assert np.isclose(result["weighted_avg"]["precision"], ref["weighted"][0])
    assert np.isclose(result["weighted_avg"]["recall"], ref["weighted"][1])
    assert np.isclose(result["weighted_avg"]["f1_score"], ref["weighted"][2], atol=2*1e-2)


# =======================================================
# TEST 2 — Missing predictions for a class (should raise)
# =======================================================

def test_7class_missing_prediction_raises():

    num_classes = 7

    # All classes appear in truth
    y_true = np.array([0,1,2,3,4,5,6] * 8 + [0,1,2,3])  # 60 total

    # Model NEVER predicts class 6
    y_pred = np.array([0,1,2,3,4,5,5] * 8 + [0,1,2,3])

    
    result = compute_precision_recall_f1(y_true, y_pred, num_classes)
    ref = sklearn_reference(y_true, y_pred, num_classes)

    # Per-class checks
    for c in range(num_classes):
        assert np.isclose(result[c]["precision"], ref["precision"][c])
        assert np.isclose(result[c]["recall"], ref["recall"][c])
        assert np.isclose(result[c]["f1_score"], ref["f1"][c])
        assert result[c]["support"] == ref["support"][c]

    # Macro
    assert np.isclose(result["macro_avg"]["precision"], ref["macro"][0])
    assert np.isclose(result["macro_avg"]["recall"], ref["macro"][1])
    assert np.isclose(result["macro_avg"]["f1_score"], ref["macro"][2])

    # Weighted
    assert np.isclose(result["weighted_avg"]["precision"], ref["weighted"][0])
    assert np.isclose(result["weighted_avg"]["recall"], ref["weighted"][1])
    assert np.isclose(result["weighted_avg"]["f1_score"], ref["weighted"][2], atol=1e-2)


# =======================================================
# TEST 3 — Perfect prediction (sanity test)
# =======================================================

def test_7class_perfect_prediction():

    num_classes = 7

    y_true = np.array([i % 7 for i in range(60)])
    y_pred = y_true.copy()

    
    result = compute_precision_recall_f1(y_true, y_pred, num_classes)

    for c in range(num_classes):
        assert result[c]["precision"] == 1.0
        assert result[c]["recall"] == 1.0
        assert result[c]["f1_score"] == 1.0

    assert result["macro_avg"]["f1_score"] == 1.0
    assert result["weighted_avg"]["f1_score"] == 1.0


# =======================================================
# TEST 4 — Extreme imbalance stress test
# =======================================================

def test_7class_extreme_imbalance():

    num_classes = 7

    # 50 samples class 0, rest tiny minorities
    y_true = np.array(
        [0]*50 + [1]*3 + [2]*2 + [3]*2 + [4]*1 + [5]*1 + [6]*1
    )

    # Model overpredicts class 0 massively
    y_pred = np.array(
        [0]*45 + [1]*5 +       # class 0
        [0]*3 +                # class 1 misclassified
        [2]*2 +
        [0]*2 +
        [4]*1 +
        [5]*1 +
        [6]*1
    )

    
    result = compute_precision_recall_f1(y_true, y_pred, num_classes)

    # Ensure weighted F1 heavily influenced by class 0
    assert result["weighted_avg"]["f1_score"] > result["macro_avg"]["f1_score"]

    # Ensure minority classes are punished
    assert result[1]["recall"] == 0.0
