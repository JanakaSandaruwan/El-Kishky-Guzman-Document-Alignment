import json
from laser_embedder import get_embeddig_list
from weight_schema import *
import argparse

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input_path', help="path to data file",type=str)
    parser.add_argument(dest='output_path', help="path to embeding file", type=str)
    parser.add_argument(dest='lang', help="language abbreviaion", type=str)
    parser.add_argument('-ex', '--existing_path',default="" ,help="path to existing embedding file", type=str)
    parser.add_argument('-n','--noofdocs', default=-1, help="no of documents to be embeded",type=int)
    parser.add_argument('-w','--weight_schema',default="sent_len" ,help="path to existing embedding file", type=str)
    return parser.parse_args()

def _main():
    args = _parse_args()
    input_path = args.input_path
    output_path = args.output_path
    lang = args.lang

    file = open(input_path, encoding='utf8')
    data = json.load(file)

    noofdocs = len(data)
    if args.noofdocs != -1:
        noofdocs = args.noofdocs

    existing_path = args.existing_path
    try:
        file = open(existing_path, encoding='utf8')
        embed_data = json.load(file)
    except:
        embed_data = []

    docs_weights=[]

    docs_weights_sent_len=[]
    docs_weights_sent_len_normalized=[]
    documents=[]

    i=0
    for docs in data[:noofdocs]:
        i+=1
        print(i)
        doc = docs['content']
        documents.append(doc)

        #get frequency weights
        docs_weights.append(documentMassNormalization(get_sentence_frequency_list(doc, lang)))

        #get sentence length weights
        weight= get_sentence_length_weighting_list(doc,lang)
        docs_weights_sent_len.append(weight)
        docs_weights_sent_len_normalized.append(documentMassNormalization(weight))


    sentence_count= sentence_count_web_domain(documents,lang)
    N=noofdocs

    docs_weights_idf=[]
    docs_weights_idf_normalized=[]

    for doc in documents:
        weight_doc= get_idf_weighting_list(doc,sentence_count,N,lang)
        docs_weights_idf.append(weight_doc)
        docs_weights_idf_normalized.append(documentMassNormalization(weight_doc))

    docs_slidf_weight= get_slidf_weighting_list(docs_weights_sent_len.copy(),docs_weights_idf.copy())

    weight_schema = args.weight_schema

    weight = docs_weights_sent_len_normalized
    if weight_schema == 'sent_freq':
        weight = docs_weights
    if weight_schema == 'idf':
        weight = docs_weights_idf_normalized
    if weight_schema =='slidf':
        weight = docs_slidf_weight

    parallel =[]
    count = 0

    for docs in data:
        a ={}
        if count < len(embed_data):
            a = embed_data[count]

            a['weight'] = weight[count]
            print("cc",count)
            count += 1
        else:
            a = docs
            doc = docs['content']

            # Get laser embedding
            source_embedd = get_embeddig_list(doc,lang)
            a ['embed'] = source_embedd
            a ['weight'] = weight[count]
            count += 1
            print(count)

        parallel.append(a)

    if len(parallel) > 0 :
        with open(output_path, 'w', encoding="utf8") as outfile:
            json.dump(parallel, outfile, ensure_ascii=False)

if __name__ == '__main__':
    _main()