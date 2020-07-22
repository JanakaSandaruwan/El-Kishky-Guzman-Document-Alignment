from laserembeddings import Laser
from doc_to_sentence import doc_to_sentence

laser = Laser()

def get_embeddig_list(doc,lang = 'en'):
    sentences = doc_to_sentence(doc,lang)
    return laser.embed_sentences(sentences,lang).tolist()

# if all sentences are in the same language:
def sent_embedding(sent,lang):
    embeddings = laser.embed_sentences([sent],lang)
    return embeddings[0]