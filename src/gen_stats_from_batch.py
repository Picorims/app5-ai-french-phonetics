# MIT License

# Copyright (c) 2024 Charly Schmidt aka Picorims<picorims.contact@gmail.com>


from difflib import SequenceMatcher
import sys
from numpy import NaN
import pandas as pd
import os

dirname = os.path.dirname(__file__)


# process args
if len(sys.argv) != 2 and len(sys.argv) != 3:
    print("Usage: python gen_stats_from_model_csv.py <csv_file> [batch]")
    sys.exit(1)

csv_file_name = sys.argv[1]
batch_name = None
if len(sys.argv) == 3:
    batch_name = sys.argv[2]
    
print(f"Generating stats from {csv_file_name}...")
if (batch_name is not None):
    print(f"Batch: {batch_name}")
    
# load csv file
if (batch_name is not None):
    df = pd.read_csv(os.path.join(dirname, "..", "tests_results", batch_name, csv_file_name))
else:
    df = pd.read_csv(os.path.join(dirname, "..", "tests_results", csv_file_name))
    


print("Loaded csv file")
print("trimming...")
# trim column names
df.columns = df.columns.str.strip()
print(df.columns)

# trim values
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


    
# get stats
print("Processing stats...")

class MetricEntry:
    successes: int
    failures: int
    total: int
    summed_similarity: float
    
    def __init__(self):
        self.successes = 0
        self.failures = 0
        self.total = 0
        self.summed_similarity = 0.0

# dict[metric] = (successes, failures, total, summed_similarity)
results_dict: dict[str, MetricEntry] = {}

def add_result_to_dict(metric: str, test_passed: bool, similarity: float):
    if metric not in results_dict:
        results_dict[metric] = MetricEntry()
    results_dict[metric].total += 1
    if test_passed:
        results_dict[metric].successes += 1
    else:
        results_dict[metric].failures += 1
    results_dict[metric].summed_similarity += similarity
    
    
   

for index, row in df.iterrows():
    if index % 1000 == 0:
        print(f"Processing row {index}...")
    
    
    test_passed = row["pass"] == "True"
    decoded = row["decoded"]
    target = row["target"]
    similarity = SequenceMatcher(None, decoded, target).ratio()
    
    add_result_to_dict(f"all_all", test_passed, similarity)
    add_result_to_dict(f"cgram_{row["cgram"]}", test_passed, similarity)
    genre = row["genre"] if row["genre"] != NaN else "None"
    add_result_to_dict(f"genre_{genre}", test_passed, similarity)
    nombre = row["nombre"] if row["nombre"] != NaN else "None"
    add_result_to_dict(f"number_{nombre}", test_passed, similarity)
    add_result_to_dict(f"nbhomogr_{row["nbhomogr"]}", test_passed, similarity)
    add_result_to_dict(f"nbhomoph_{row["nbhomoph"]}", test_passed, similarity)
    add_result_to_dict(f"islem_{row["islem"]}", test_passed, similarity)
    add_result_to_dict(f"nblettres_{row["nblettres"]}", test_passed, similarity)
    add_result_to_dict(f"nbphons_{row["nbphons"]}", test_passed, similarity)
    add_result_to_dict(f"nbsyll_{row["nbsyll"]}", test_passed, similarity)
    
    
# build csv
print("Building csv...")
csv_lines = []
for metric, entry in results_dict.items():
    mean_similarity = entry.summed_similarity / entry.total
    ratio = (entry.successes / entry.total)
    metric_type, metric_value = metric.split("_", 1)
    csv_lines.append(f"{metric_type},{metric_value},{entry.successes},{entry.failures},{entry.total},{ratio},{mean_similarity}")

# sort by metric_type, then metric_value
# if metric_value is a number, it will be sorted as a number
print("Sorting csv...")
def key_func(x):
    metric_type, metric_value = x.split(",")[0:2]
    return (metric_type, int(metric_value) if metric_value.isnumeric() else metric_value)
csv_lines.sort(key=key_func)

# add header
csv_lines.insert(0, "metric_type,metric_value,successes,failures,total,ratio,mean_similarity")

  
# write csv
print("Writing csv...")
if not os.path.exists(os.path.join(dirname, "..", "stats")):
    os.makedirs(os.path.join(dirname, "..", "stats"))
    
if (batch_name is not None) and not os.path.exists(os.path.join(dirname, "..", "stats", batch_name)):
    os.makedirs(os.path.join(dirname, "..", "stats", batch_name))

csv_file_name = f"stats_for__{csv_file_name}"
save_path = None
if (batch_name is not None):
    save_path = os.path.join(dirname, "..", "stats", batch_name, csv_file_name)
else:
    save_path = os.path.join(dirname, "..", "stats", csv_file_name)    
        
with open(save_path, "w", encoding="utf-8") as f:
    f.write("\n".join(csv_lines))
    
print(f"Stats saved to {save_path}")
print("Done")