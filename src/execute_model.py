# MIT License

# Copyright (c) 2024 Charly Schmidt aka Picorims<picorims.contact@gmail.com>,
# Alexis Hu, Maxime Lebot

# Usage: python execute_model.py <model_name> <input_string>

import _model_inference as mi
import sys

from pathlib import Path
dirpath = Path(__file__).parent.absolute()

if len(sys.argv) != 3:
    print("Usage: python execute_model.py <model_name> <input_string>")
    sys.exit(1)
    
model_name = sys.argv[1]
input_text = sys.argv[2]

print("Restoring model...")
encoder_model, decoder_model, max_decoder_seq_length = mi.restore_model(model_name, dirpath)
restored_target_token_index = mi.restore_token_index(model_name, dirpath, "target")
restored_input_token_index = mi.restore_token_index(model_name, dirpath, "input")

print("Decoding sequence...")
input_seq = mi.encode_text(input_text, max_decoder_seq_length, restored_input_token_index)
decoded_sentence = mi.decode_sequence(input_seq, encoder_model, decoder_model, model_name, restored_target_token_index)

print("================================================")
print(f"Input sentence: {input_text}")
print(f"Decoded sentence: {decoded_sentence}")