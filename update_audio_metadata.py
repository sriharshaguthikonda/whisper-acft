#!/usr/bin/env python3
"""
Script to populate metadata fields for audio files in i:\Record_harsha based on transcription content.
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse
import subprocess

def load_keyword_results(state_file_path):
    """Load the keyword search state file and extract matched file paths."""
    with open(state_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract all JSON file paths from the results
    matched_json_paths = []
    for result in data.get('results', []):
        json_path = result.get('path')
        if json_path:
            matched_json_paths.append(json_path)
    
    return matched_json_paths, data.get('results', [])

def json_to_audio_filename(json_path):
    """Convert JSON transcription filename to expected audio filename."""
    # Remove .json extension
    base_name = Path(json_path).stem
    
    # Common audio extensions to try
    audio_extensions = ['.mp3', '.wav', '.m4a', '.mp4', '.avi', '.mov', '.mkv']
    
    return base_name, audio_extensions

def find_audio_file(json_path, record_dir):
    """Find the corresponding audio file for a given JSON transcription file."""
    base_name, audio_extensions = json_to_audio_filename(json_path)
    
    # Try different audio extensions
    for ext in audio_extensions:
        audio_file = record_dir / f"{base_name}{ext}"
        if audio_file.exists():
            return audio_file
    
    # If not found with exact name, try case-insensitive search
    try:
        for file_path in record_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                if file_path.stem.lower() == base_name.lower():
                    return file_path
    except Exception:
        pass
    
    return None

def load_transcription_content(json_path):
    """Load and extract content from transcription JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract different types of content
        content = {
            'title': '',
            'description': '',
            'keywords': [],
            'year': '',
            'genre': 'Medical Consultation',
            'comments': ''
        }
        
        # Try to get title from filename or content
        filename_stem = Path(json_path).stem
        content['title'] = filename_stem.replace('_', ' - ').replace('.json', '')
        
        # Extract keywords from filename
        if 'harsha' in filename_stem.lower():
            content['keywords'].append('Dr Sri Harsha Guthikonda')
        
        # Try to extract medical conditions/topics from filename
        medical_terms = [
            'diabetes', 'hypertension', 'asthma', 'cancer', 'heart', 'stroke',
            'depression', 'anxiety', 'pain', 'infection', 'surgery', 'medication',
            'pregnancy', 'pediatric', 'emergency', 'consultation', 'follow-up'
        ]
        
        for term in medical_terms:
            if term in filename_stem.lower():
                content['keywords'].append(term.title())
        
        # Try to get content from groq_response if available
        if 'groq_response' in data:
            groq_data = data['groq_response']
            if isinstance(groq_data, dict):
                # Extract summary or content
                if 'summary' in groq_data:
                    content['description'] = groq_data['summary'][:200] + '...' if len(groq_data['summary']) > 200 else groq_data['summary']
                elif 'content' in groq_data:
                    content['description'] = groq_data['content'][:200] + '...' if len(groq_data['content']) > 200 else groq_data['content']
        
        # Extract date if available
        if 'date' in data:
            content['year'] = str(data.get('date', '')).split('-')[0] if data.get('date') else ''
        
        # Set contributing artist
        content['artist'] = 'Dr Sri Harsha Guthikonda'
        
        # Set album based on content type
        if 'emergency' in filename_stem.lower():
            content['album'] = 'Emergency Consultations'
        elif 'surgery' in filename_stem.lower() or 'operation' in filename_stem.lower():
            content['album'] = 'Surgical Consultations'
        elif 'pediatric' in filename_stem.lower() or 'child' in filename_stem.lower():
            content['album'] = 'Pediatric Consultations'
        else:
            content['album'] = 'Medical Consultations'
        
        return content
        
    except Exception as e:
        print(f"⚠️  Error loading transcription {json_path}: {e}")
        return None

def check_ffmpeg():
    """Check if ffmpeg is available for metadata editing."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def update_audio_metadata(audio_file, metadata, dry_run=False):
    """Update metadata for audio file using ffmpeg."""
    if not check_ffmpeg():
        print("❌ ffmpeg not found. Please install ffmpeg to update metadata.")
        return False
    
    # Prepare ffmpeg metadata commands
    metadata_args = []
    
    # Map our metadata to ffmpeg metadata tags
    tag_mapping = {
        'title': 'title',
        'artist': 'artist',
        'album': 'album',
        'genre': 'genre',
        'year': 'date',
        'description': 'description',
        'comments': 'comment'
    }
    
    for our_key, ffmpeg_tag in tag_mapping.items():
        if metadata.get(our_key):
            metadata_args.extend(['-metadata', f'{ffmpeg_tag}={metadata[our_key]}'])
    
    # Add keywords as comment if available
    if metadata.get('keywords'):
        keywords_str = ', '.join(metadata['keywords'])
        metadata_args.extend(['-metadata', f'comment=Keywords: {keywords_str}'])
    
    if not metadata_args:
        return True  # No metadata to update
    
    # Create backup
    backup_file = audio_file.with_suffix(audio_file.suffix + '.backup')
    if not dry_run:
        shutil.copy2(audio_file, backup_file)
    
    # Build ffmpeg command
    output_file = audio_file.with_suffix('.temp' + audio_file.suffix)
    cmd = [
        'ffmpeg', '-i', str(audio_file),
        '-c', 'copy',  # Copy streams without re-encoding
        *metadata_args,
        '-y',  # Overwrite output file
        str(output_file)
    ]
    
    if dry_run:
        print(f"🔄 Would update metadata for: {audio_file.name}")
        print(f"   Metadata: {metadata}")
        return True
    
    try:
        # Run ffmpeg
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Replace original with updated file
        shutil.move(str(output_file), str(audio_file))
        
        # Remove backup
        backup_file.unlink()
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error updating metadata for {audio_file.name}: {e}")
        print(f"   stderr: {e.stderr}")
        
        # Restore from backup if it exists
        if backup_file.exists():
            shutil.move(str(backup_file), str(audio_file))
        
        return False

def update_audio_metadata_batch(record_harsha_dir, state_file, dry_run=False):
    """Update metadata for all audio files in Record_harsha based on transcriptions."""
    
    # Convert to Path objects
    record_harsha_dir = Path(record_harsha_dir)
    state_file = Path(state_file)
    
    # Validate paths
    if not record_harsha_dir.exists():
        print(f"❌ Record_harsha directory not found: {record_harsha_dir}")
        return
    
    if not state_file.exists():
        print(f"❌ State file not found: {state_file}")
        return
    
    # Load matched JSON files and results
    print("Loading keyword search results...")
    matched_json_paths, results = load_keyword_results(state_file)
    print(f"Found {len(matched_json_paths)} matched transcription files")
    
    # Check ffmpeg availability
    if not check_ffmpeg() and not dry_run:
        print("❌ ffmpeg not found. Please install ffmpeg:")
        print("   Windows: choco install ffmpeg")
        print("   Or download from: https://ffmpeg.org/download.html")
        return
    
    # Track statistics
    updated_count = 0
    not_found_count = 0
    error_count = 0
    skipped_count = 0
    
    # Process each matched JSON file
    print("="*60)
    print("UPDATING AUDIO METADATA ({})".format('DRY RUN' if dry_run else 'ACTUAL'))
    print("="*60)
    print(f"📁 Directory: {record_harsha_dir}")
    print(f"📄 State file: {state_file}")
    
    for json_path in tqdm(matched_json_paths, desc="Processing files"):
        # Find corresponding audio file
        audio_file = find_audio_file(json_path, record_harsha_dir)
        
        if audio_file is None:
            not_found_count += 1
            continue
        
        # Load transcription content
        metadata = load_transcription_content(json_path)
        if metadata is None:
            error_count += 1
            continue
        
        # Update metadata
        if update_audio_metadata(audio_file, metadata, dry_run):
            if not dry_run:
                print(f"✅ Updated: {audio_file.name}")
                updated_count += 1
            else:
                print(f"🔄 Would update: {audio_file.name}")
                skipped_count += 1
        else:
            error_count += 1
    
    # Print summary
    print("="*60)
    print("SUMMARY ({})".format('DRY RUN' if dry_run else 'ACTUAL'))
    print(f"Total matched transcriptions: {len(matched_json_paths)}")
    if dry_run:
        print(f"Files that would be updated: {skipped_count}")
    else:
        print(f"Files successfully updated: {updated_count}")
    print(f"Audio files not found: {not_found_count}")
    print(f"Errors encountered: {error_count}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Update metadata for audio files in Record_harsha based on transcription content"
    )
    parser.add_argument(
        "--state-file",
        default=r"i:\P2GPT_google_drive\My Drive\Transcriptions\keyword_search_state.json",
        help="Path to keyword search state JSON file"
    )
    parser.add_argument(
        "--record-harsha-dir",
        default=r"i:\Record_harsha",
        help="Directory containing the moved audio files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without actually changing files"
    )
    
    args = parser.parse_args()
    
    print("🎵 Audio Metadata Updater")
    print("🔍 Processing keyword search results...")
    print(f"📁 Record_harsha directory: {args.record_harsha_dir}")
    print(f"📄 State file: {args.state_file}")
    
    update_audio_metadata_batch(
        args.record_harsha_dir,
        args.state_file,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
