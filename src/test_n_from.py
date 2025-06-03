# MIT License

# Copyright (c) 2024 Charly Schmidt aka Picorims<picorims.contact@gmail.com>,
# Alexis Hu, Maxime Lebot

import sys
import os
from pathlib import Path
from difflib import SequenceMatcher
dirpath = Path(__file__).parent.absolute()

import _model_inference as mi

batch_dir_name = None
print_lines = 1

if len(sys.argv) < 4:
    print("Usage: python test_n_from.py <model_name> <from> <n>")
    sys.exit(1)
    
for i in range(4, len(sys.argv), 2):
    param = sys.argv[i]
    value = sys.argv[i+1]
    if (param == "-b"):
        batch_dir_name = value
    elif (param == "-p"):
        print_lines = int(value)
    else:
        print(f"Unknown parameter: {param}")
        sys.exit(1)

    
model_name = sys.argv[1]
from_ = int(sys.argv[2])
n = int(sys.argv[3])

model_list = []
if batch_dir_name is not None:
    model_list = os.listdir(os.path.join(dirpath, "..", "models", batch_dir_name))
    # only keep the model files (with the .keras extension)
    model_list = [f for f in model_list if f.endswith(".keras")]
    if len(model_list) == 0:
        print(f"No model found in batch directory: {batch_dir_name}")
        sys.exit(1)
    print(f"Batch directory: {batch_dir_name}")
    print(f"Models found: {model_list}")
else:
    model_list.append(model_name)

print(f"Testing {n} sequences from {from_} for each model.")

model_list.sort(key =  lambda x: int(x.split("_")[2].replace("e", "")))




dataset_header_line = None

print("Reading dataset...")
data_path = os.path.join(dirpath, "..", "input_csv", "lexique_minimal_seed-1.csv")
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
    # remove header line
    dataset_header_line = lines[0]
    lines = lines[1:]

stats_file = None
stats_file_lines = None
if batch_dir_name is not None:
    stats_file = os.path.join(dirpath, "..", "tests_results", batch_dir_name, f"stats_{n}n_{from_}f.csv")
    stats_file_lines = []
    stats_file_lines.append("epoch, successes, failures, total, percent, median_similarity, model_name")

for model_name in model_list:
    print(f"========================= Testing model: {model_name} ======================")
    print("Restoring model...")
    model_path = dirpath
    if batch_dir_name is not None:
        model_path = os.path.join(model_path, "..", "models", batch_dir_name)
    else:
        model_path = os.path.join(model_path, "..", "models")
    encoder_model, decoder_model, max_decoder_seq_length = mi.restore_model(model_name, model_path, use_path_as_is=True)
    tokens_file_name = model_name
    if batch_dir_name is not None:
        tokens_file_name = batch_dir_name
    restored_target_token_index = mi.restore_token_index(tokens_file_name, model_path, "target", use_path_as_is=True)
    restored_input_token_index = mi.restore_token_index(tokens_file_name, model_path, "input", use_path_as_is=True)





    print("launching decoding test...")
    count = 0
    successes = 0
    failures = 0
    similarities = []
    out_file_lines = []
    out_csv_lines = []
    out_csv_lines.append("index, input, decoded, target, pass, " + dataset_header_line)

    for seq_index in range(from_, from_ + n):
        if seq_index % 100 == 0:
            print(f"Testing sequence {seq_index}...")
        # trying out decoding.
        
        line_data = lines[seq_index].split(",")
        input_text = line_data[0]
        target_text = line_data[1]

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
        
        log_line = f"-{seq_index}, {seq_index-from_}- {input_text + spaces_tab_input} -> {decoded_sentence.replace("\n","") + spaces_tab_decoded},expect: {target_text + spaces_tab_target}, pass: {success}"
        if print_lines == 1: print(log_line)
        out_file_lines.append(log_line)
        
        out_csv_lines.append(f"{seq_index}, {input_text}, {decoded_sentence}, {target_text}, {success}, " + lines[seq_index])
            
    log_line = f"Successes: {successes}/{count}, percent: {(successes/count) * 100}%"
    print(log_line)
    out_file_lines.append(log_line)
    log_line = f"Failures: {failures}/{count}"
    print(log_line)
    out_file_lines.append(log_line)

    # computing median similarity

    similarities.sort()
    n = len(similarities)
    if n % 2 == 0:
        median = (similarities[n//2] + similarities[n//2 - 1]) / 2
    else:
        median = similarities[n//2]
        
    log_line = f"Median similarity: {median}"
    print(log_line)
    out_file_lines.append(log_line)
    
    if stats_file is not None:
        epoch = int(model_name.split("_")[2].replace("e", ""))
        stats_file_lines.append(f"{epoch}, {successes}, {failures}, {count}, {(successes/count) * 100}%, {median}, {model_name}")

    if not os.path.exists(os.path.join(dirpath, "..", "tests_results")):
        os.makedirs(os.path.join(dirpath, "..", "tests_results"))
        
    out_file_name = f"test_{n}n_{from_}f__{model_name}.txt"
    out_path = None
    if (batch_dir_name is not None):
        if not os.path.exists(os.path.join(dirpath, "..", "tests_results", batch_dir_name)):
            os.makedirs(os.path.join(dirpath, "..", "tests_results", batch_dir_name))
        out_path = os.path.join(dirpath, "..", "tests_results", batch_dir_name, out_file_name)
    else:
        out_path = os.path.join(dirpath, "..", "tests_results", out_file_name)
    
    print("Saving output to:")
    print(out_path)
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_file_lines))
        
    with open(out_path.replace(".txt", ".csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_csv_lines))
        

if stats_file is not None:        
    print("Saving stats...")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_file_lines))


print("Done.")