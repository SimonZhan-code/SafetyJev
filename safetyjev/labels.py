"""Join pre-action forecasts to complete, contiguous future monitor samples."""


def label_forecasts(forecasts, oracle):
    samples = {}
    for sample in oracle:
        step = sample["step"]
        if type(step) is not int or step < 0 or step in samples:
            raise ValueError("Oracle steps must be unique nonnegative integers")
        samples[step] = sample
    seen = set()
    result = []
    for forecast in forecasts:
        key = forecast["forecast_id"]
        if key in seen:
            raise ValueError("Duplicate forecast ID")
        seen.add(key)
        start, end = forecast["start_step"], forecast["end_step"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError("Invalid prediction horizon")
        if len(forecast["input"]["remaining_actions"]) != end - start:
            raise ValueError("Prediction horizon must match supplied action count")
        cid = forecast["constraint_id"]

        def verdict(step):
            sample = samples.get(step, {})
            if sample.get("valid") is not True:
                return None
            value = sample.get("violated", {}).get(cid)
            return value if type(value) is bool else None

        label, reason, event = None, "censored", None
        initial = verdict(start)
        if initial is None:
            reason = "invalid_start_monitor"
        elif initial:
            reason = "already_violated"
        else:
            # Never bridge a monitor gap, even when a later sample is positive.
            for step in range(start + 1, end + 1):
                value = verdict(step)
                if value is None:
                    break
                if value:
                    label, reason, event = 1, "observed_violation", step
                    break
            else:
                label, reason = 0, "complete_no_violation"
        result.append({
            "forecast_id": key, "constraint_id": cid,
            "start_step": start, "end_step": end,
            "label": label, "label_reason": reason,
            "first_violation_step": event,
        })
    return result
