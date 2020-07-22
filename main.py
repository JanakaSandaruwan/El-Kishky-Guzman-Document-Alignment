import json
from greedy_mover_distance import greedy_mover_distance
from competitive_matching import competitive_matching
from weight_schema import *

file  = open('embedding.json', encoding='utf8')
data = json.load(file)

source_docs = []
target_docs = []
source_docs_weights=[]
target_docs_weights=[]

source_docs_weights_sent_len=[]
target_docs_weights_sent_len=[]
source_docs_weights_sent_len_normalized=[]
target_docs_weights_sent_len_normalized=[]

en_documents=[]
si_documents=[]

source_docs_weights_intra_doc_word_idf = []
target_docs_weights_intra_doc_word_idf = []

for docs in data:
    doc_en = docs['content_en']
    doc_si = docs['content_si']

    en_documents.append(doc_en)
    si_documents.append(doc_si)

    # Get laser embedding
    source_embedd = docs ['embed_en']
    target_embedd = docs ['embed_si']
    source_docs.append(source_embedd)
    target_docs.append(target_embedd)

    #get frequency weights
    source_docs_weights.append(documentMassNormalization(get_sentence_frequency_list(source_embedd)))
    target_docs_weights.append(documentMassNormalization(get_sentence_frequency_list(target_embedd)))

    #get sentence length weights
    # en_weight= get_sentence_length_weighting_list(doc_en)
    # si_weight = get_sentence_length_weighting_list(doc_si)
    # source_docs_weights_sent_len.append(en_weight)
    # target_docs_weights_sent_len.append(si_weight)
    # source_docs_weights_sent_len_normalized.append(documentMassNormalization(en_weight))
    # target_docs_weights_sent_len_normalized.append(documentMassNormalization(si_weight))

    source_docs_weights_sent_len_normalized.append(docs['weight_en'])
    target_docs_weights_sent_len_normalized.append(docs['weight_si'])

scores ={}

for i in range(len(source_docs)):
    for j in range(len(target_docs)):
        scores[(i,j)]=greedy_mover_distance(source_docs[i].copy(),target_docs[j].copy(),source_docs_weights[i].copy()
                                            ,target_docs_weights[j].copy())


sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
match=competitive_matching(sorted_scores)


count=0.0
for pair in match:
    if (pair[0]==pair[1]):
        count+=1
print ("Matched document pairs for sentence frequency ",count*100/len(match))

scores ={}

for i in range(len(source_docs)):
    for j in range(len(target_docs)):
        scores[(i,j)]=greedy_mover_distance(source_docs[i],target_docs[j],source_docs_weights_sent_len_normalized[i].copy()
                                            ,target_docs_weights_sent_len_normalized[j].copy())

sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
match=competitive_matching(sorted_scores)

count=0.0
for pair in match:
    if (pair[0]==pair[1]):
        count+=1
print ("Matched document pairs for sentence length ",count*100/len(match))


sentence_count_en= sentence_count_web_domain(en_documents)
sentence_count_si= sentence_count_web_domain(si_documents)
N_en=len(en_documents)
N_si=len(si_documents)
source_docs_weights_idf=[]
target_docs_weights_idf=[]

source_docs_weights_idf_normalized=[]
target_docs_weights_idf_normalized=[]

for doc in en_documents:
    weight_doc= get_idf_weighting_list(doc,sentence_count_en,N_en)
    source_docs_weights_idf.append(weight_doc)
    source_docs_weights_idf_normalized.append(documentMassNormalization(weight_doc))

for doc in si_documents:
    weight_doc=get_idf_weighting_list(doc,sentence_count_si,N_si)
    target_docs_weights_idf.append(weight_doc)
    target_docs_weights_idf_normalized.append(documentMassNormalization(weight_doc))

scores ={}

for i in range(len(source_docs)):
    for j in range(len(target_docs)):
        scores[(i,j)]=greedy_mover_distance(source_docs[i],target_docs[j],source_docs_weights_idf_normalized[i].copy()
                                            ,target_docs_weights_idf_normalized[j].copy())

sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
match=competitive_matching(sorted_scores)

count=0.0
for pair in match:
    if (pair[0]==pair[1]):
        count+=1
print ("Matched document pairs for idf weighting",count*100/len(match))

source_docs_slidf_weight= get_slidf_weighting_list(source_docs_weights_sent_len.copy()
                                                                             ,source_docs_weights_idf.copy())
target_docs_slidf_weight= get_slidf_weighting_list(target_docs_weights_sent_len.copy()
                                                                             ,target_docs_weights_idf.copy())

scores ={}

for i in range(len(source_docs)):
    for j in range(len(target_docs)):
        scores[(i,j)]=greedy_mover_distance(source_docs[i],target_docs[j],source_docs_slidf_weight[i].copy()
                                            ,target_docs_slidf_weight[j].copy())

sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
match=competitive_matching(sorted_scores)

count=0.0
for pair in match:
    if (pair[0]==pair[1]):
        count+=1
print ("Matched document pairs for slidf weighting",count*100/len(match))