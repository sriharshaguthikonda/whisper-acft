# Whisper Speech Detection Analysis Report

## Summary
Using Whisper tiny model to analyze the three manual_ audio files for actual spoken content.

## File-by-File Analysis

### 1. manual_ctrl_r_20260307_142310_489677_8502100ms.wav
- **Duration**: 141:42.100 (2h 21m 42s)
- **Speech Detected**: 0.400s (0.005% of total)
- **Words**: 3 words
- **Speech Segment**: 
  - **Time**: 141:41.640 → 141:42.040 (last 0.4 seconds)
  - **Text**: "it all means."
  - **Confidence**: Very low (0.007)

### 2. manual_ctrl_r_20260307_142251_084499_8501200ms.wav
- **Duration**: 141:41.200 (2h 21m 41s)
- **Speech Detected**: 10.12s (0.12% of total)
- **Words**: 20 words
- **Speech Segment**:
  - **Time**: 141:30.000 → 141:40.620 (last ~10 seconds)
  - **Text**: "But instead Mario did I give you the reason, it But I jump like that Six downs What was that?"
  - **Confidence**: Low (0.072)

### 3. manual_f24_20260321_011145_789681_6890200ms.wav
- **Duration**: 114:50.200 (1h 54m 50s)
- **Speech Detected**: 37:12.820 (32.4% of total)
- **Words**: 2,429 words
- **Speech Segments**: 221 segments
- **Longest Speech Gap**: 2:01.800
- **Sample Speech Segments**:
  - 00:30.000 → 00:39.320 (9.32s, 26 words)
  - 02:41.120 → 02:47.720 (6.60s, 18 words)
  - 02:57.780 → 03:01.160 (3.38s, 12 words)
  - ... and 218 more segments

## Key Findings

### 🎯 **Where is the actual speech?**

1. **manual_ctrl_r files**: Almost no speech detected
   - File 1: Only 3 words at the very end (last 0.4 seconds)
   - File 2: Only 20 words at the very end (last 10 seconds)
   - **99.9% of these files are silent/background noise**

2. **manual_f24 file**: Significant speech content
   - **32.4% speech content** (37 minutes of speech)
   - **221 speech segments** throughout the file
   - Speech appears throughout, not just at the end
   - **67.6% is silence/gaps** between speech segments

### 🔍 **Analysis Insights**

- The two `manual_ctrl_r` files appear to be recordings that were mostly silent/dead air with only brief speech at the very end
- The `manual_f24` file contains actual conversation/meeting content with regular speech segments
- Speech confidence levels are generally low, suggesting the audio quality may be poor or distant
- Long speech gaps in the f24 file (up to 2+ minutes) suggest it might be a meeting with pauses

### 📊 **Overall Statistics**
- **Total Duration**: 6h 38m 13s
- **Total Speech**: 37m 23s (9.4%)
- **Total Words**: 2,452
- **Most speech is in**: manual_f24 file (99.6% of all detected speech)

## Recommendations

1. **manual_ctrl_r files**: These appear to be mostly empty recordings - consider checking if they're corrupted or intended to be silent
2. **manual_f24 file**: This contains actual content and is worth processing/transcribing
3. **Audio Quality**: Low confidence scores suggest the recordings may need enhancement for better transcription accuracy

## Usage Commands

The analysis was performed using:
```bash
python whisper_speech_detector.py --files file1.wav file2.wav file3.wav --model tiny --language en --output-json results.json
```
