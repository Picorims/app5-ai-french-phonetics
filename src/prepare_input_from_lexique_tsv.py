import pandas as pd
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python prepare_input_from_lexique_tsv.py <path_to_lexique_tsv>")
    sys.exit(1)

# Open TSV file
print("Reading TSV file...")
df = pd.read_csv(sys.argv[1], sep='\t')

# Only the ortho and phon columns are relevant, so we get rid of other columns
print("Extracting ortho and phon columns...")
minimalDf = df[["ortho","phon","syll"]] # word, phonetics, phonetics with syllabus decomposition

# Create a variant with randomized order
# see: https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
print("Creating a variant with randomized order...")
randomizedMinimalDf = minimalDf.copy().sample(frac=1).reset_index(drop=True)

# write to csv
print("Writing to CSV file...")
if not os.path.exists('out'):
    os.makedirs('out')
minimalDf.to_csv('out/lexique_minimal.csv', index=False)
randomizedMinimalDf.to_csv('out/lexique_minimal_randomized.csv', index=False)