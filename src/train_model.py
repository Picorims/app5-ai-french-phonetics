# MIT License

# Copyright (c) 2024 Charly Schmidt aka Picorims<picorims.contact@gmail.com>,
# Alexis Hu, Maxime Lebot

# Based on: https://keras.io/examples/nlp/lstm_seq2seq/

import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
from pathlib import Path

dirpath = Path(__file__).parent.absolute()

# (see https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)
batch_size = 64  # Batch size for training. (bigger batch size = faster training, but less accurate)
epochs = 10  # Number of epochs to train for. (an epoch is a full iteration over samples, so the bigger, the more accurate and the slower the training is)
latent_dim = 256  # Latent dimensionality of the encoding space. (number of LSTMs in layer? TODO: search)
num_samples = 1000  # Number of samples to train on.
# Path to the data txt file on disk.
print("Using randomized csv with seed: 1")
data_path = os.path.join(dirpath, "..", "out", "lexique_minimal_seed-1.csv")



# === prepare the data ===
print("Preparing the data...")

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set() # TODO: handling of accents?
target_characters = set()
# space is a token required for blank leftovers at the end of a sample buffer.
# if length is 8, and we encode "hello", in reality we encode "hello___"
# with three spaces to fill the gap (here represented as underscores)
input_characters.add(" ")
target_characters.add(" ")
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
# 1: and num_sample+1 are here to ignore the headers line of the csv file
for line in lines[1 : min(num_samples+1, len(lines) - 1)]:
    input_text, target_text, _ = line.split(",")
    
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters) # each character is a token
num_decoder_tokens = len(target_characters)
# max_encoder_seq_length = max([len(txt) for txt in input_texts])
# max_decoder_seq_length = max([len(txt) for txt in target_texts])
max_encoder_seq_length = 64
max_decoder_seq_length = 64

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])


# for each sample, we consider the max char length, where each char can be any of the available tokens.
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype="float32",
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype="float32",
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype="float32",
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    # for each character of a sample, we put one value of the array to 1:
    # the token corresponding to the character we represent.
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    # remaining unallocated characters in the sample buffer are set to the " " token
    encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0







# === Build the model ===
print("Building the model...")

# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens)) # (varying sample length, each char can be any token)
encoder = keras.layers.LSTM(latent_dim, return_state=True) # https://keras.io/api/layers/recurrent_layers/lstm/
encoder_outputs, state_h, state_c = encoder(encoder_inputs) # h = hidden state (working memory), c = cell state (long term memory)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax") # LSTM working memory to token conversion
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)





# === Train the model ===
print("Training the model...")

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    [encoder_input_data, decoder_input_data], # inputs
    decoder_target_data, # outputs
    batch_size=batch_size, # size of batches triggering updates of the model based on errors
    epochs=epochs, # iterations
    validation_split=0.2, # fraction of validation data
)
# Save model
if not os.path.exists('models'):
    os.makedirs('models')
model.save(os.path.join(dirpath, "..", "models", f"fr2phon_{batch_size}b_{epochs}e_{num_samples}s_{latent_dim}ld_{max_encoder_seq_length}esl_{max_decoder_seq_length}dsl_seed-1.keras"))
