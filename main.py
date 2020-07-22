import json
from greedy_mover_distance import greedy_mover_distance
from competitive_matching import competitive_matching
import argparse

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='source_path', help="path to source embedding file",type=str)
    parser.add_argument(dest='target_path', help="path to target embedding file", type=str)
    parser.add_argument('-ns','--noofdocs_source', default=-1, help="no of documents to be aligned in source",type=int)
    parser.add_argument('-nt','--noofdocs_target', default=-1, help="no of documents to be aligned in target",type=int)
    return parser.parse_args()

def _main():
    args = _parse_args()
    source_path= args.source_path
    target_path= args.target_path

    file  = open(source_path,encoding='utf8')
    source_embed = json.load(file)
    file.close()

    file = open(target_path, encoding='utf8')
    target_embed = json.load(file)
    file.close()

    noofdocs_source =len(source_embed)
    noofdocs_target =len(target_embed)

    if args.noofdocs_source != -1:
        noofdocs_source=args.noofdocs_source

    if args.noofdocs_target != -1:
        noofdocs_target=args.noofdocs_target

    scores ={}

    for i in range(noofdocs_source):
        for j in range(noofdocs_target):
            scores[(i,j)]=greedy_mover_distance(source_embed["embed"].copy(),target_embed["embed"],
                                                source_embed["weight"].copy(),target_embed["weight"].copy())

    sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
    match=competitive_matching(sorted_scores)

    print("Document Alignment -----------------------------------------------------------")
    print(match)

if __name__ == '__main__':
    _main()