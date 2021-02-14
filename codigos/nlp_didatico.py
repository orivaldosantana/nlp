
from collections import defaultdict
from gensim import corpora 

class NLPDidatico: 
    """ Classe com recursos básicos NLP """

    def __init__(self, corpusIn):
        self.corpus = corpusIn
        self.palavrasVazias = 0
        self.corpusProcessadado = 0
        self.dicionario = 0 
        self.bowCorpus 

    # Entrada: um vetor com vários textos / sentenças 
    # Saída: Extração das principais palavras de cada texto. Cada texto vira um vetor de palavras mais importantes  
    # Descrição: Este pré-processamento torna todas palavras mínusculas, elimina as palavara vazias, eliminas as palavras que aparecem apenas uma vez  
    def preprocessamento(self, palavrasVaziasIn, limiarContPalavrasIn=1):
        self.palavrasVazias = palavrasVaziasIn
        # Torna mínusculas cada documento, separa as palavras por espaço em branco e elimina as palavras vazias  
        textos = [[palavra for palavra in docx.lower().split() if palavra not in self.palavrasVazias] for docx in self.corpus]

        # Conta a frequência das palavras 
        frequencia = defaultdict(int)
        for texto in textos:
            for token in texto:
                frequencia[token] += 1
        
        # Mantem as palavras que aparecem mais de uma vez 
        self.corpusProcessadado = [[token for token in texto if frequencia[token] > limiarContPalavrasIn] for texto in textos]
        return self.corpusProcessadado  

    # Gerar o bag of words (BoW) para um corpus processado 
    def geraBowCorpus(self):
        self.dicionario = corpora.Dictionary( self.corpusProcessadado ) 
        self.bowCorpus = [self.dicionario.doc2bow(texto) for texto in self.corpusProcessadado]
        return self.bowCorpus 