from nlp_didatico import NLPDidatico 
import pandas as pd

base_dados = pd.read_csv('https://raw.githubusercontent.com/orivaldosantana/nlp/main/dados/emotions_val.txt ', delimiter=';') 

dados_emocao_1 = base_dados.loc[ (base_dados.iloc[:,1] == 'fear') | (base_dados.iloc[:,1] == 'love'), :].values
corpus_emocao_1 = dados_emocao_1[:,0]

print( dados_emocao_1.shape )
print( corpus_emocao_1.shape )