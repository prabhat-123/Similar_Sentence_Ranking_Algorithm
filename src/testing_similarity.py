import os
import argparse

from sentences_encodings import SentenceEncoder
from rank_sentence_similarity import Ranking_Algorithm

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--model_name", required=True,
                    help="name of the the pickle model file")
args = vars(parser.parse_args())

root_dir = os.getcwd()
model_dir = 'models'
model_dir_path = os.path.join(root_dir, model_dir)
sentence_encoder_obj = SentenceEncoder(name = 'bert-base-nli-mean-tokens')
sbert_model = sentence_encoder_obj.load_model()

query = input("Ask your question here..")
ranking_obj = Ranking_Algorithm(model_file_path = os.path.join(model_dir_path, args["model_name"]))
similarity_scores = ranking_obj.predict_similarity_scores(query, sbert_model)
ranking_table = ranking_obj.rank_similarity_scores(similarity_scores)
print(ranking_table)