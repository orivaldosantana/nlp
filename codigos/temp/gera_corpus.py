# Lê uma base de dados 

import pandas as pd

baseDados = pd.read_csv('https://raw.githubusercontent.com/orivaldosantana/nlp/main/dados/emotions_train.txt ', delimiter=';') 

print(baseDados.head())


#dadosEmocoes = baseDados.loc[ (baseDados.iloc[:,1] == 'fear') | (baseDados.iloc[:,1] == 'love'), :].values

dadosEmocoes = baseDados.loc[ :, :].values

corpusEmocao = dadosEmocoes[:,0]

print(corpusEmocao[:6])

f = open("../../dados/corpus_emocoes_2.txt", "a")
for doc in corpusEmocao:
    f.write(doc+"\n") 
f.close()
