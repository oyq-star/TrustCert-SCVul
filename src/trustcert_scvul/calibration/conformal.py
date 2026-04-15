"""Split conformal prediction for selective classification."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score


def compute_conformal_scores(y_true, y_prob):
    """Compute nonconformity scores for split conformal prediction.
    Score = 1 - probability of the true class.
    """
    scores = np.where(y_true == 1, 1 - y_prob, y_prob)
    return scores


def calibrate_conformal(cal_y_true, cal_y_prob, alpha=0.10):
    """Compute conformal threshold from calibration set.

    Returns threshold q_hat such that prediction sets have
    >= (1 - alpha) marginal coverage.
    """
    scores = compute_conformal_scores(cal_y_true, cal_y_prob)
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = np.quantile(scores, min(q_level, 1.0))
    return q_hat


def predict_with_abstention(y_prob, q_hat):
    """Make predictions with abstention using conformal threshold.

    Returns:
        predictions: array of -1 (abstain), 0 (safe), 1 (vulnerable)
        prediction_sets: list of sets for each sample
        accepted_mask: boolean array of accepted (non-abstained) samples
    """
    n = len(y_prob)
    predictions = np.full(n, -1)  # default: abstain
    prediction_sets = []
    accepted_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        p = y_prob[i]
        pred_set = set()

        # Check if class 0 is in prediction set
        score_0 = p  # score if true class were 0
        if score_0 <= q_hat:
            pred_set.add(0)

        # Check if class 1 is in prediction set
        score_1 = 1 - p  # score if true class were 1
        if score_1 <= q_hat:
            pred_set.add(1)

        prediction_sets.append(pred_set)

        if len(pred_set) == 1:
            # Singleton set -> accept
            predictions[i] = list(pred_set)[0]
            accepted_mask[i] = True
        # else: abstain (ambiguous set {0,1} or empty set)

    return predictions, prediction_sets, accepted_mask


def evaluate_selective(y_true, y_prob, q_hat, model_name='model'):
    """Evaluate selective prediction with conformal abstention."""
    predictions, pred_sets, accepted = predict_with_abstention(y_prob, q_hat)

    n_total = len(y_true)
    n_accepted = accepted.sum()
    acceptance_rate = n_accepted / n_total if n_total > 0 else 0

    results = {
        'model': model_name,
        'n_total': n_total,
        'n_accepted': n_accepted,
        'n_abstained': n_total - n_accepted,
        'acceptance_rate': acceptance_rate,
    }

    if n_accepted > 0:
        y_true_acc = y_true[accepted]
        y_pred_acc = predictions[accepted]
        y_prob_acc = y_prob[accepted]

        results['accepted_precision'] = precision_score(y_true_acc, y_pred_acc, zero_division=0)
        results['accepted_recall'] = recall_score(y_true_acc, y_pred_acc, zero_division=0)
        results['accepted_f1'] = f1_score(y_true_acc, y_pred_acc, zero_division=0)
        try:
            results['accepted_auprc'] = average_precision_score(y_true_acc, y_prob_acc)
        except Exception:
            results['accepted_auprc'] = 0.0

        # Coverage: fraction of true labels in prediction sets
        coverage_count = sum(
            1 for i in range(n_total) if y_true[i] in pred_sets[i]
        )
        results['empirical_coverage'] = coverage_count / n_total
    else:
        results['accepted_precision'] = 0
        results['accepted_recall'] = 0
        results['accepted_f1'] = 0
        results['accepted_auprc'] = 0
        results['empirical_coverage'] = 0

    # All-sample metrics (with abstention treated as negative)
    y_pred_all = np.where(accepted, predictions, 0)
    results['all_sample_f1'] = f1_score(y_true, y_pred_all, zero_division=0)

    return results


def compute_risk_coverage_curve(y_true, y_prob, n_points=50):
    """Compute risk-coverage curve for different abstention thresholds."""
    # Sort by confidence (higher prob or lower prob = more confident)
    confidence = np.abs(y_prob - 0.5) * 2  # 0 = uncertain, 1 = confident
    sorted_idx = np.argsort(-confidence)  # most confident first

    coverages = []
    risks = []

    for k in range(1, len(y_true) + 1, max(1, len(y_true) // n_points)):
        selected = sorted_idx[:k]
        coverage = k / len(y_true)
        y_pred_k = (y_prob[selected] >= 0.5).astype(int)
        risk = 1 - f1_score(y_true[selected], y_pred_k, zero_division=0)
        coverages.append(coverage)
        risks.append(risk)

    return np.array(coverages), np.array(risks)


def compute_aurc(coverages, risks):
    """Compute Area Under the Risk-Coverage curve."""
    try:
        return np.trapz(risks, coverages)
    except AttributeError:
        # numpy >= 2.0 moved trapz to np.trapezoid
        return np.trapezoid(risks, coverages)
