from nlp_didatico import NLPDidatico 
import pandas as pd

baseDados = pd.read_csv('https://raw.githubusercontent.com/orivaldosantana/nlp/main/dados/emotions_train.txt ', delimiter=';') 

dadosEmocao1 = baseDados.loc[ (baseDados.iloc[:,1] == 'fear') | (baseDados.iloc[:,1] == 'love'), :].values
corpusEmocao1 = dadosEmocao1[:,0]

print( dadosEmocao1.shape )
print( corpusEmocao1.shape )

# Cria uma lista das palavras muito frequentes ou vazias 
listaVazias = set('go get href ive id ill im ll m one s www ve t put feel i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for  with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very can will just should now'.split(' ')) 

nlp = NLPDidatico( corpusEmocao1 )

print("\n-- Corpus Original --")
nlp.printCorpus( corpusEmocao1 )

nlp.preprocessamento(listaVazias)
print("\n-- Corpus Processado --")
nlp.printPreProCorpus()

nlp.geraBowCorpus()
print("\n-- BoW Corpus --") 
nlp.printBowCorpus()

nlp.geraTfidfCorpus()
print("\n-- TF-IDF Corpus --")
nlp.printTfidCorpus()

nlp.geraIndexadorSimilaridade()

print("\n-- Encontra as N sentenças mais semelhantes -- ")
nlp.encontraNmais('i just know to begin with i am going to feel shy about it',5)

print("\n-- Encontra as N sentenças mais semelhantes -- ")
nlp.encontraNmais('i feel shy to him all the time',5)

