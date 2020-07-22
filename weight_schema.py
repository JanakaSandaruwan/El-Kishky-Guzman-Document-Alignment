from doc_to_sentence import doc_to_sentence
import math


def get_sentence_frequency_list(doc, lang):
    # first weighting method
    sentences = doc_to_sentence(doc, lang)
    length = len(sentences)
    weights = []
    for sent in sentences:
        frequency = sentences.count(sent)
        weights.append(frequency / length)
    return weights


def get_sentence_count():
    # get no of same sentence in doc
    # assume all are distinct
    return 1.0


def get_sentence_length_weighting_list(doc, lang):
    weight = []
    # sentences= doc_to_sentence(doc)
    # sentences=[]
    sentences = doc_to_sentence(doc, lang)
    for sentence in sentences:
        weight.append(get_sentence_count() * len(sentence.split()))
    total_tokens = float(sum(weight))
    # print(weight)
    return [x / total_tokens for x in weight]

def documentMassNormalization(wieght_list):
    total_weight = float(sum(wieght_list))
    for i in range(len(wieght_list)):
        wieght_list[i] = wieght_list[i] / total_weight
    return wieght_list


def get_idf_weighting_list(doc, sentence_count, N, lang):
    weights = []
    sentences = doc_to_sentence(doc, lang)
    for sentence in sentences:
        sent = sentence.strip()
        weights.append(1 + math.log((1.0 + N) / (1.0 + sentence_count[sent])))
    return weights


def sentence_count_web_domain(documents, lang):
    sentence_count = {}
    for doc in documents:
        sentences = doc_to_sentence(doc, lang)
        for sentence in sentences:
            sent = sentence.strip()
            if (sent in sentence_count):
                sentence_count[sent] += 1
            else:
                sentence_count[sent] = 1
    return sentence_count


def get_slidf_weighting_list(sentence_weight, idf_weight):
    return [
        documentMassNormalization([sentence_weight[j][i] * idf_weight[j][i] for i in range(len(sentence_weight[j]))])
        for j in range(len(sentence_weight))]