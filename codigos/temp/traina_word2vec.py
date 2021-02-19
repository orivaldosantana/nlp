#fonte: https://rare-technologies.com/word2vec-tutorial/
#fonte: https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py 
#fonte: https://radimrehurek.com/gensim/models/word2vec.html#introduction   

from gensim.test.utils import datapath
from gensim import utils

# import modules & set up logging
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
class EmoctionCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath("/home/oriva/projetos_ml/nlp/dados/corpus_emocoes_2.txt")
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


import gensim.models

sentences = EmoctionCorpus()
model = gensim.models.Word2Vec(sentences=sentences, size=20)

model.save("/home/oriva/projetos_ml/nlp/dados/emotion_model") 