import os

import numpy as np
from data_loader import DataLoader
from sentences_encodings import SentenceEncoder
from sentence_transformers import SentenceTransformer

sentences = DataLoader(file_name = 'quora_data.csv').load_data(row_numbers = 500)
sentence_encoder_obj = SentenceEncoder(name = 'bert-base-nli-mean-tokens')
sbert_model = sentence_encoder_obj.load_model()
sentence_encoder_obj.encode_sentences(sbert_model = sbert_model, sentences = sentences)
