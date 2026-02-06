import pandas as pd

# Load speaker scores
df = pd.read_csv('i:\\whisper-acft\\speaker_sort_scores.csv')
score_dict = {}
for _, row in df.iterrows():
    file_path = row['file']
    score = row['score']
    score_dict[file_path] = score

print("Checking score_dict contents:")
for i, (k, v) in enumerate(score_dict.items()):
    if i >= 5:
        break
    print(f"  Item {i}: key={k} (type: {type(k)}), value={v} (type: {type(v)})")

print("\nChecking for any non-string keys:")
non_string_keys = [(k, type(k)) for k in score_dict.keys() if not isinstance(k, str)]
if non_string_keys:
    print("Found non-string keys:")
    for k, t in non_string_keys[:5]:
        print(f"  {k} (type: {t})")
else:
    print("All keys are strings")

print("\nTrying to reproduce the error:")
try:
    normalized_score_dict = {k.lower(): v for k, v in score_dict.items()}
    print("Success! Normalized dictionary created.")
except Exception as e:
    print(f"Error: {e}")
    print("Checking which item causes the error:")
    for i, (k, v) in enumerate(score_dict.items()):
        try:
            result = k.lower()
        except Exception as e:
            print(f"  Error at item {i}: key={k} (type: {type(k)}), error={e}")
            break
