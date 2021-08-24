import os

import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

root_dir = os.getcwd()

class SentenceEncoder():

    def __init__(self,name):

        self.name = name


    def load_model(self):

        sbert_model = SentenceTransformer(self.name)
        return sbert_model

    
    def encode_sentences(self, sbert_model, filtered_data):

        self.sbert_model = sbert_model
        self.sentences = filtered_data
        sentence_embeddings = {}
        for index in tqdm(range(len(filtered_data))):

            id = index + 1
            sentence = filtered_data['Questions'][index]
            sentence_embedding = sbert_model.encode(sentence)
            sentence_embeddings[id] = sentence_embedding

        if not os.path.exists(os.path.join(root_dir, 'models')):

            os.mkdir(os.path.join(root_dir, 'models'))

        with open('./models/sentence_encodings.pickle', 'wb') as file:

            pickle.dump(sentence_embeddings, file)




    
