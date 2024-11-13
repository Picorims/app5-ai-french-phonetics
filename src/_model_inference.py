import os

import keras
import numpy as np

def restore_model(name, dirpath):
    print(f"restoring model {name}")
    # Define sampling models
    # Restore the model and construct the encoder and decoder.
    model = keras.models.load_model(os.path.join(dirpath,"..","models",name))

    latent_dim = int(name.split("_")[4][:-2])
    max_decoder_seq_length = int(name.split("_")[6][:-3])
    print(f"latent_dim: {latent_dim}, max_decoder_seq_length: {max_decoder_seq_length}")
    
    # h = hidden state (working memory), c = cell state (long term memory)

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = keras.Input(shape=(latent_dim,), name="decoder_state_input_h")
    decoder_state_input_c = keras.Input(shape=(latent_dim,), name="decoder_state_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states,
    )
    
    return (encoder_model, decoder_model, max_decoder_seq_length)


def decode_sequence(input_seq, encoder_model, decoder_model, target_token_index, max_decoder_seq_length):
    # Reverse-lookup token index to decode sequences back to
    # something readable.
    # reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
    
    num_decoder_tokens = len(target_token_index)
    
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )

        # Sample a token (find the char of the token with the highest probability)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence

def encode_text(text, max_encoder_seq_length, input_token_index):
    input_text = "\t" + text + "\n"
    num_encoder_tokens = len(input_token_index)
    
    encoder_input_data = np.zeros(
        (1, max_encoder_seq_length, num_encoder_tokens),
        dtype="float32",
    )
    
    # for each character of a sample, we put one value of the array to 1:
    # the token corresponding to the character we represent.
    for t, char in enumerate(input_text):
        if t >= max_encoder_seq_length:
            break
        if not char in input_token_index:
            char = " "
        else:
            encoder_input_data[0, t, input_token_index[char]] = 1.0
    # remaining unallocated characters in the sample buffer are set to the " " token
    encoder_input_data[0, t + 1 :, input_token_index[" "]] = 1.0
    return encoder_input_data