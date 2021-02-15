
from collections import defaultdict
from gensim import corpora 
from gensim import models 
from gensim import similarities
import numpy as np 

class NLPDidatico: 
    """ Classe com recursos básicos NLP """

    def __init__(self, corpusIn):
        self.corpus = corpusIn
        self.palavrasVazias = 0
        self.corpusProcessadado = 0
        self.dicionario = 0 
        self.bowCorpus = 0
        self.tfidfCorpus = 0 
        self.tfidfModel = 0 
        self.indexadorSim = 0 

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

    def geraTfidfCorpus(self): 
        # Gerar o "Corpus" TFIDF a apartir de um "Corpus" BoW  
        self.tfidfModel = models.TfidfModel( self.bowCorpus )
        self.tfidfCorpus = self.tfidfModel[self.bowCorpus]
        return self.tfidfCorpus 

    def geraIndexadorSimilaridade(self):
        self.indexadorSim = similarities.SparseMatrixSimilarity(self.tfidfCorpus, num_features=len( self.dicionario.token2id ))

    # Encontra os N elementos mais semelhantes 
    def encontraNmais(self, textoIn, n): 
        textoTokens = textoIn.split()    
        textoBow = self.dicionario.doc2bow( textoTokens )
        #print( textoBow )
        sims = self.indexadorSim[ self.tfidfModel[textoBow] ]
        #print(list(enumerate(sims)))
        resultado = np.zeros((n,2)) 
        cont = 0
        for numeroDoc, semelhanca in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
            cont = cont + 1
            print(numeroDoc," ", semelhanca, " ", self.corpus[numeroDoc]  )
            if cont < n :
                resultado[cont,0]  = numeroDoc
                resultado[cont,1] = semelhanca 
            else: 
                break 
        return resultado 

    def printTfidCorpus(self, numIn = 5):
        self.printCorpus(self.tfidfCorpus)

    def printCorpus(self, corpusIn, numIn = 5):
        cont = 0
        for doc in corpusIn: 
            print(cont, ' ', doc)
            cont = cont + 1
            if cont > numIn:
                break            

    def printBowCorpus(self, numIn = 5):
        self.printCorpus(self.bowCorpus, numIn)

    # Imprime o corpus pre processado 
    def printPreProCorpus(self, numIn = 5):
        self.printCorpus(self.corpusProcessadado, numIn)      