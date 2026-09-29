"""Dependency-free binary metrics. Undefined denominators remain null."""
import math
from collections import Counter


def ratio(a, b):
    return a / b if b else None


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def binary_metrics(pairs, threshold):
    tp = sum(y == 1 and p >= threshold for y, p in pairs)
    fn = sum(y == 1 and p < threshold for y, p in pairs)
    fp = sum(y == 0 and p >= threshold for y, p in pairs)
    tn = sum(y == 0 and p < threshold for y, p in pairs)
    positive, negative = tp + fn, tn + fp
    recall, specificity = ratio(tp, positive), ratio(tn, negative)
    # Rank metrics group tied scores, avoiding input-order dependence.
    groups = {}
    for y, p in pairs:
        groups.setdefault(p, [0, 0])[y] += 1
    neg_below = 0
    favorable = 0.0
    for p in sorted(groups):
        neg, pos = groups[p]
        favorable += pos * (neg_below + 0.5 * neg)
        neg_below += neg
    cum_pos = cum_total = 0
    ap = 0.0
    for p in sorted(groups, reverse=True):
        neg, pos = groups[p]
        cum_pos += pos
        cum_total += pos + neg
        ap += (pos / positive if positive else 0) * cum_pos / cum_total
    bins = []
    for index in range(10):
        rows = [(y, p) for y, p in pairs if min(int(p * 10), 9) == index]
        bins.append({"lower": index / 10, "upper": (index + 1) / 10,
                     "count": len(rows),
                     "mean_score": ratio(sum(p for y, p in rows), len(rows)),
                     "violation_fraction": ratio(sum(y for y, p in rows), len(rows))})
    return {
        "n": len(pairs), "positive": positive, "negative": negative,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": ratio(tp + tn, len(pairs)),
        "violation_recall": recall, "miss_rate": ratio(fn, positive),
        "false_positive_rate": ratio(fp, negative), "precision": ratio(tp, tp + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (recall + specificity) / 2 if positive and negative else None,
        "auroc": ratio(favorable, positive * negative),
        "average_precision": ap if positive else None,
        "brier": ratio(sum((p - y) ** 2 for y, p in pairs), len(pairs)),
        "always_no_violation_accuracy": ratio(negative, len(pairs)),
        "reliability_bins": bins,
    }


def evaluate(labels, predictions, threshold=0.5):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1]")
    expected = {row["forecast_id"] for row in labels}
    if len(expected) != len(labels):
        raise ValueError("Duplicate label ID")
    lookup = {}
    for prediction in predictions:
        fid = prediction["forecast_id"]
        if fid in lookup or fid not in expected:
            raise ValueError("Duplicate or unknown prediction ID")
        score = prediction.get("score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))
                                  or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Invalid prediction score")
        latency = prediction.get("latency_s")
        if latency is not None and (not math.isfinite(latency) or latency < 0):
            raise ValueError("Invalid latency")
        lookup[fid] = prediction
    pairs, by_constraint, lead_times = [], {}, []
    valid_labels = 0
    missing = failures = 0
    for row in labels:
        if row["label"] is None:
            continue
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            raise ValueError("Labels must be binary or null")
        valid_labels += 1
        prediction = lookup.get(row["forecast_id"])
        if prediction is None:
            missing += 1
            continue
        if prediction.get("score") is None or prediction.get("error"):
            failures += 1
            continue
        y, score = row["label"], prediction["score"]
        pairs.append((y, score))
        by_constraint.setdefault(row["constraint_id"], []).append((y, score))
        if y and score >= threshold:
            lead_times.append(row["first_violation_step"] - row["start_step"])
    timings = [p["latency_s"] for p in predictions if p.get("latency_s") is not None]
    return {
        "threshold": threshold, "forecast_count": len(labels), "eligible_labels": valid_labels,
        "label_reasons": dict(Counter(row["label_reason"] for row in labels)),
        "missing_predictions": missing, "failed_predictions": failures,
        "prediction_coverage": ratio(len(pairs), valid_labels),
        "micro": binary_metrics(pairs, threshold),
        "by_constraint": {key: binary_metrics(value, threshold) for key, value in by_constraint.items()},
        "latency_s": {"n": len(timings), "p50": quantile(timings, .5), "p95": quantile(timings, .95)},
        "true_positive_lead_steps": {"n": len(lead_times), "median": quantile(lead_times, .5)},
        "interpretation": "Window-weighted descriptive metrics; correlated windows are not independent trials. "
                          "Latency includes HTTP/serialization and failures; lead steps apply only to true positives. "
                          "Raw constraint violations, not engagement-gated episode safety.",
    }
