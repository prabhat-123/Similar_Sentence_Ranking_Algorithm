import os

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from data_loader import DataLoader
from sentence_transformers import SentenceTransformer

root_dir = os.getcwd()

class SentenceEncoder():

    def __init__(self,name):

        self.name = name


    def load_model(self):

        sbert_model = SentenceTransformer(self.name)
        return sbert_model

    
    def encode_sentences(self, sbert_model, sentences):

        self.sbert_model = sbert_model
        self.sentences = sentences
        sentence_embeddings = {}
        for index in tqdm(range(len(sentences))):

            id = index + 1
            sentence = sentences['Questions'][index]
            sentence_embedding = sbert_model.encode(sentence)
            sentence_embeddings[id] = sentence_embedding

        if not os.path.exists(os.path.join(root_dir, 'models')):

            os.mkdir(os.path.join(root_dir, 'models'))

        with open('./models/sentence_encodings.pickle', 'wb') as file:

            pickle.dump(sentence_embeddings, file)




    
