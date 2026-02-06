import os
import soundfile as sf

others_dir = 'I:/Record_others'
files = [f for f in os.listdir(others_dir) if f.lower().endswith(('.wav', '.m4a', '.mp3', '.flac'))]
print(f'Audio files found: {len(files)}')
print('Sample files:', files[:5])

if files:
    print('Checking first file...')
    first_file = os.path.join(others_dir, files[0])
    print(f'First file: {first_file}')
    try:
        info = sf.info(first_file)
        print(f'Audio info: {info.samplerate}Hz, {info.duration:.2f}s, {info.channels} channels')
    except Exception as e:
        print(f'Error reading audio: {e}')
