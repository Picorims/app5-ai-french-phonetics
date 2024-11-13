# MIT License

# Copyright (c) 2024 Charly Schmidt aka Picorims<picorims.contact@gmail.com>,
# Alexis Hu, Maxime Lebot

import sys
import os
from pathlib import Path
from difflib import SequenceMatcher
dirpath = Path(__file__).parent.absolute()

import _model_inference as mi

if not len(sys.argv) == 4:
    print("Usage: python test_n_from.py <model_name> <from> <n>")
    sys.exit(1)
    
model_name = sys.argv[1]
from_ = int(sys.argv[2])
n = int(sys.argv[3])



print("Restoring model...")
encoder_model, decoder_model, max_decoder_seq_length = mi.restore_model(model_name, dirpath)
restored_target_token_index = mi.restore_token_index(model_name, dirpath, "target")
restored_input_token_index = mi.restore_token_index(model_name, dirpath, "input")



print("Reading dataset...")
data_path = os.path.join(dirpath, "..", "input_csv", "lexique_minimal_seed-1.csv")
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
    # remove header line
    lines = lines[1:]



print("launching decoding test...")
count = 0
successes = 0
failures = 0
similarities = []

for seq_index in range(from_, from_ + n):
    # trying out decoding.
    
    input_text, target_text, _ = lines[seq_index].split(",")

    input_seq = mi.encode_text(input_text, max_decoder_seq_length, restored_input_token_index)
    decoded_sentence = mi.decode_sequence(input_seq, encoder_model, decoder_model, model_name, restored_target_token_index)
    
    # trim whitespaces
    decoded_sentence = decoded_sentence.strip()
    target_text = target_text.strip()
    
    # stats
    success = decoded_sentence == target_text
    count += 1
    if success:
        successes += 1
    else:
        failures += 1
    s = SequenceMatcher(None, decoded_sentence, target_text)
    similarities.append(s.ratio())
    
    input_length = len(input_text)
    decoded_length = len(decoded_sentence)
    target_length = len(target_text)
    spaces_tab_input = " " * max(0,(25 - input_length))
    spaces_tab_decoded = " " * max(0,(20 - decoded_length))
    spaces_tab_target = " " * max(0,(20 - target_length))
    print(f"-{seq_index}, {seq_index-from_}- {input_text + spaces_tab_input} -> {decoded_sentence.replace("\n","") + spaces_tab_decoded},expect: {target_text + spaces_tab_target}, pass: {success}")
    
        
print(f"Successes: {successes}/{count}, ratio: {successes/count}")
print(f"Failures: {failures}/{count}")

# computing median similarity

similarities.sort()
n = len(similarities)
if n % 2 == 0:
    median = (similarities[n//2] + similarities[n//2 - 1]) / 2
else:
    median = similarities[n//2]
    
print(f"Median similarity: {median}")