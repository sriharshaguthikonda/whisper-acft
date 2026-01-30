import os
import librosa

others_dir = 'I:/Record_others'
files = [f for f in os.listdir(others_dir) if f.lower().endswith(('.wav', '.m4a', '.mp3', '.flac'))]
print(f'Audio files found: {len(files)}')
print('Sample files:', files[:5])

if files:
    print('Checking first file with librosa...')
    first_file = os.path.join(others_dir, files[0])
    print(f'First file: {first_file}')
    try:
        y, sr = librosa.load(first_file, sr=None)
        print(f'Audio info: {sr}Hz, {len(y)/sr:.2f}s duration')
        print(f'Successfully loaded {len(y)} samples')
    except Exception as e:
        print(f'Error reading audio with librosa: {e}')
