"""Report observed state episodes separately from repeated labeled windows."""


def state_segments(annotation):
    rows=[];resets=set(annotation.get('reset_steps',[]))
    for qid,query in annotation['queries'].items():
        if query['temporal_kind']!='state':continue
        counts=dict(positive_segments=0,observed_safe_to_unsafe_onsets=0,
                    left_censored_segments=0,unknown_prior_segments=0)
        previous=None;previous_step=None
        for state in query['states']:
            step=state['step'];label=state['label']
            consecutive=previous_step is not None and step==previous_step+1 and step not in resets
            if label==1 and not (consecutive and previous==1):
                counts['positive_segments']+=1
                if step==0:counts['left_censored_segments']+=1
                elif consecutive and previous==0:counts['observed_safe_to_unsafe_onsets']+=1
                else:counts['unknown_prior_segments']+=1
            previous=label;previous_step=step
        rows.append(dict(episode_id=annotation['episode_id'],query_id=qid,
                         semantic_id=query['semantic_id'],**counts))
    return rows


def merge_window_reports(reports, field, keys):
    """Episode memberships are disjoint across the already verified shards."""
    result={}
    def add(target,source):
        for key,value in source.items():
            if isinstance(value,dict):add(target.setdefault(key,{}),value)
            else:target[key]=target.get(key,0)+value
    for report in reports:
        for row in report[field]:
            identity=tuple(row[k] for k in keys)
            target=result.setdefault(identity,{k:row[k] for k in keys})
            add(target,{k:v for k,v in row.items() if k not in keys})
    return [result[k] for k in sorted(result)]
