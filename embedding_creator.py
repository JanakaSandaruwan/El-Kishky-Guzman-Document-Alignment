import json
from laserembedder import get_embeddig_list
from datetime import datetime
from weight_schema import *

file  = open('docalign/data/army_parallel_new.json',encoding='utf8')
data = json.load(file)

try:
    file  = open('embedding.json', encoding='utf8')
    embed_data = json.load(file)
except:
    embed_data =[]


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

source_digits = []
target_digits = []

source_names = []
source_designations = []

i=0
start=datetime.now()
for docs in data:
    i+=1
    print(i)
    doc_en = docs['content_en']
    doc_si = docs['content_si']

    en_documents.append(doc_en)
    si_documents.append(doc_si)


    #get frequency weights
    # source_docs_weights.append(documentMassNormalization(get_sentence_frequency_list(doc_en, "en")))
    # target_docs_weights.append(documentMassNormalization(get_sentence_frequency_list(doc_si, "si")))

    #get sentence length weights
    en_weight= get_sentence_length_weighting_list(doc_en, "en")
    si_weight = get_sentence_length_weighting_list(doc_si, "si")
    source_docs_weights_sent_len.append(en_weight)
    target_docs_weights_sent_len.append(si_weight)
    source_docs_weights_sent_len_normalized.append(documentMassNormalization(en_weight))
    target_docs_weights_sent_len_normalized.append(documentMassNormalization(si_weight))


sentence_count_en= sentence_count_web_domain(en_documents,'en')
sentence_count_si= sentence_count_web_domain(si_documents,'si')
N_en=len(en_documents)
N_si=len(si_documents)
source_docs_weights_idf=[]
target_docs_weights_idf=[]

source_docs_weights_idf_normalized=[]
target_docs_weights_idf_normalized=[]

for doc in en_documents:
    weight_doc= get_idf_weighting_list(doc,sentence_count_en,N_en,'en')
    source_docs_weights_idf.append(weight_doc)
    source_docs_weights_idf_normalized.append(documentMassNormalization(weight_doc))

for doc in si_documents:
    weight_doc=get_idf_weighting_list(doc,sentence_count_si,N_si,'si')
    target_docs_weights_idf.append(weight_doc)
    target_docs_weights_idf_normalized.append(documentMassNormalization(weight_doc))

source_docs_slidf_weight= get_slidf_weighting_list(source_docs_weights_sent_len.copy()
                                                                             ,source_docs_weights_idf.copy())
target_docs_slidf_weight= get_slidf_weighting_list(target_docs_weights_sent_len.copy()
                                                                             ,target_docs_weights_idf.copy())

parallel =[]
count = 0


for docs in data:
    a ={}
    if count < len(embed_data):
        a = embed_data[count]
        a['weight_en'] = source_docs_weights_intra_doc_word_idf[count]
        a['weight_si'] = source_docs_weights_intra_doc_word_idf[count]
        print("cc",count)
        count += 1
    else:
        a = docs
        doc_en = docs['content_en']
        doc_si = docs['content_si']
        # print(doc_en)
        # Get laser embedding
        source_embedd = get_embeddig_list(doc_en,lang='en')
        target_embedd = get_embeddig_list(doc_si,lang='si')
        a ['embed_si'] = target_embedd
        a ['embed_en'] = source_embedd
        a ['weight_en'] = source_docs_slidf_weight[count]
        a ['weight_si'] = target_docs_slidf_weight[count]
        count += 1
        print(count)

    parallel.append(a)

if len(parallel) > 0 :
    with open("embedding.json", 'w', encoding="utf8") as outfile:
        json.dump(parallel, outfile, ensure_ascii=False)
