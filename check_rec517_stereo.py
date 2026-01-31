import json

# Read the manifest and check New recording 517 files
manifest_file = r'I:\Record_chunks\pairs_manifest_stereo.jsonl'
rec517_entries = []

with open(manifest_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            if 'New recording 517' in entry.get('audio_path', ''):
                rec517_entries.append(entry)

print(f'Found {len(rec517_entries)} New recording 517 entries')
print()

# Check first few entries for stereo processing details
for i, entry in enumerate(rec517_entries[:10]):
    audio_path = entry.get('audio_path', '')
    print(f'Entry {i+1}: {audio_path}')
    print(f'  Channel: {entry.get("channel", "N/A")}')
    print(f'  Stereo policy: {entry.get("stereo_policy", "N/A")}')
    print(f'  Stereo duplicate: {entry.get("stereo_duplicate", "N/A")}')
    print(f'  Stereo dropped channel: {entry.get("stereo_dropped_channel", "N/A")}')
    print(f'  Stereo reason: {entry.get("stereo_reason", "N/A")}')
    print(f'  Stereo corr: {entry.get("stereo_corr", "N/A")}')
    print()

# Check chunk distribution
chunks = {}
for entry in rec517_entries:
    if 'chunk' in entry['audio_path']:
        chunk_id = entry['audio_path'].split('chunk')[1].split('.')[0]
        if chunk_id not in chunks:
            chunks[chunk_id] = []
        chunks[chunk_id].append(entry['channel'])

print('Sample chunk channel distribution:')
for chunk_id, channels in list(chunks.items())[:10]:
    print(f'  Chunk {chunk_id}: {channels}')

# Count unique chunks vs total entries
print(f'\nTotal unique New recording 517 chunks: {len(chunks)}')
print(f'Total New recording 517 entries: {len(rec517_entries)}')

# Check if any chunks have both L and R
both_channels = 0
single_channel = 0
for chunk_id, channels in chunks.items():
    if len(set(channels)) == 2:  # Both L and R present
        both_channels += 1
    else:
        single_channel += 1

print(f'Chunks with both L and R channels: {both_channels}')
print(f'Chunks with single channel only: {single_channel}')
