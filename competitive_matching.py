def competitive_matching(sorted_list):
    aligned=[]
    source=[]
    target=[]
    for pair in sorted_list.keys():
        if pair[0] not in source and pair[1] not in target:
            aligned.append(pair)
            source.append(pair[0])
            target.append(pair[1])
    return aligned