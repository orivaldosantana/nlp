import gensim.models

modeloEmocoes = gensim.models.Word2Vec.load("/home/oriva/projetos_ml/nlp/dados/emotion_model") 

print( modeloEmocoes.wv.most_similar( positive=['sunday', 'holiday' ], topn=10 )) 