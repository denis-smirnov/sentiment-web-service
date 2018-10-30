# Input and output paths
data_dir = "../rusentiment/Dataset/"
embeddings_filename = "../fasttext.min_count_100.vk_posts_all_443550246.300d.vec"
model_output_dir = "./keras_model"

preprocessing_pipeline_filename = "keras_data_preprocessing.joblib"
keras_model_weights_filename = "keras_model_weights.h5"

# Preprocessing parameters
dictionary_size = 10000
padding_size = 200

import os
from shutil import rmtree
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten

from model_runtime.transformers import TextToFreqRankTransformer, SequencePaddingTransformer
from utils import load_embeddings

np.random.seed(291018)

preselected_posts = pd.read_csv(os.path.join(data_dir, "rusentiment_preselected_posts.csv"))
random_posts = pd.read_csv(os.path.join(data_dir, "rusentiment_random_posts.csv"))

raw_test = pd.read_csv(os.path.join(data_dir, "rusentiment_test.csv"))
raw_train = pd.concat([preselected_posts, random_posts]).reset_index(drop=True)  # only 12 posts are in both sets

num_classes = len(raw_train.label.value_counts())
label_encoder = LabelEncoder().fit(raw_train.label)
y_train = to_categorical(label_encoder.transform(raw_train.label))
y_test = to_categorical(label_encoder.transform(raw_test.label))

text_transformer = TextToFreqRankTransformer(dictionary_size).fit(raw_train.text)
sequence_padder = SequencePaddingTransformer(padding_size).fit(raw_train.text)

preprocessing_pipeline = Pipeline([
    ("text_to_freq_rank_sequences", text_transformer),
    ("pad_sequences", sequence_padder)
])


x_train = preprocessing_pipeline.transform(raw_train.text)
x_test = preprocessing_pipeline.transform(raw_test.text)

embedding_layer_size = dictionary_size+1

_, embedding_dim, embedding = load_embeddings(embeddings_filename)

embedding_matrix = np.zeros((embedding_layer_size, embedding_dim))
for token, rank in text_transformer.token_rank.items():
    embedding_vector = embedding.get(token)
    if embedding_vector is not None:
        embedding_matrix[rank] = embedding_vector

model = Sequential()
model.add(Embedding(embedding_layer_size, embedding_dim, input_length=padding_size))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False


model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["acc"])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_test, y_test))
# We don't use validation score for model selection here, so it is safe to use test data

if os.path.exists(model_output_dir):
    rmtree(model_output_dir)
os.mkdir(model_output_dir)

joblib.dump(preprocessing_pipeline, os.path.join(model_output_dir, preprocessing_pipeline_filename))
model.save(os.path.join(model_output_dir, keras_model_weights_filename))
