import os
import operator

import pickle
import numpy as np
from data_loader import DataLoader
from prettytable import PrettyTable



class Ranking_Algorithm():

    def __init__(self, model_file_path):

        self.model_file_path = model_file_path
        

    def cosine_similarity(self, vector1, vector2):

        self.vector1 = vector1
        self.vector2 = vector2
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


    def predict_similarity_scores(self, query, sbert_model):

        self.query = query
        self.sbert_model = sbert_model
        query_vector = sbert_model.encode([query])[0]
        similarity_scores = {}
        with open(self.model_file_path, 'rb') as file:
            sentences_embeddings = pickle.load(file)
    
            for key in sentences_embeddings:
                score = self.cosine_similarity(query_vector, sentences_embeddings[key])
                similarity_scores[key] = score
        return similarity_scores



    def rank_similarity_scores(self, similarity_scores):

        self.similarity_scores = similarity_scores
        sorted_similarity_score = dict( sorted(similarity_scores.items(), key=operator.itemgetter(1),reverse=True)[:5])
        top_five_similarity_scores = list(sorted_similarity_score.values())
        top_five_similarity_scores_id = list(sorted_similarity_score.keys())
        filtered_df = DataLoader(file_name = 'quora_data.csv').load_data(row_numbers = 500)
        top_five_filtered_data = list(filtered_df.loc[filtered_df['id'].isin(top_five_similarity_scores_id), "Questions"])
        ranking_table = PrettyTable()
        ranking_table.add_column("Similar Sentences",top_five_filtered_data)
        ranking_table.add_column("Similarity Score", top_five_similarity_scores)
        return ranking_table

