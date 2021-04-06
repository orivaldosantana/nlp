from nlp_didatico import NLPDidatico 

# Coleção de documentos 
textoCorpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

# Cria uma lista das palavras muito frequentes ou vazias 
listaVazias = set('go get href ive id ill im ll m one s www ve t put feel i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for  with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very can will just should now'.split(' ')) 

nlp = NLPDidatico(textoCorpus)

print("\n-- Corpus Original --")
nlp.printCorpus(textoCorpus)

nlp.preprocessamento(listaVazias)
print("\n-- Corpus Processado --")
nlp.printPreProCorpus()

nlp.geraBowCorpus()
print("\n-- BoW Corpus --") 
nlp.printBowCorpus()

nlp.geraTfidfCorpus()
print("\n-- TF-IDF Corpus --")
nlp.printTfidfCorpus()

nlp.geraIndexadorSimilaridade()

print("\n-- Encontra as N sentenças mais semelhantes -- ")
nlp.encontraNmais('system engineering',4)

print("\n -- Gera representação LSI e imprime -- ")
nlp.geraRepresentacaoLSI()
nlp.printLSICorpus() 

print("\n -- Gera gráfico com a representação LSI -- ")
nlp.geraVisualizacaoLSI()  

