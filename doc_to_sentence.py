from sentence_splitter import SentenceSplitter,split_text_into_sentences
from sinling import SinhalaTokenizer

tokenizer = SinhalaTokenizer()
splitter = SentenceSplitter(language='en')


def doc_to_sentence(doc,lang):
    if lang == 'en':
        return splitter.split(text=doc)
    elif lang == 'si':
        return tokenizer.split_sentences(doc)
    else:
        return splitter.split(text=doc)