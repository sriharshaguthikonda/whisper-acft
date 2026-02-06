import pandas as pd

# Load speaker scores
df = pd.read_csv('i:\\whisper-acft\\speaker_sort_scores.csv')
print("CSV columns:", df.columns.tolist())
print("First few rows:")
print(df.head())
print()

# Check what we're trying to create
score_dict = dict(zip(df['file'], df['score']))
print("Type of score_dict:", type(score_dict))
print("First 3 items in score_dict:")
for i, (k, v) in enumerate(score_dict.items()):
    if i >= 3:
        break
    print(f"  Key: {k} (type: {type(k)})")
    print(f"  Value: {v} (type: {type(v)})")
