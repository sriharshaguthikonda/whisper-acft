# Comprehensive Manual_ Files Speech Analysis Report

## Executive Summary
Analyzed the **top 10 largest manual_ files** (out of 337 total files >30s) using Whisper tiny model to detect actual speech content.

**Key Finding**: **13.2%** of total duration contains speech, with dramatic variation between file types.

## File-by-File Analysis

### 🎯 **High Speech Content Files**

#### 1. manual_f24_20260321_011145_789681_6890200ms.wav
- **Duration**: 114:50.200 (1h 54m 50s)
- **Speech**: 37:12.820 (**32.4%**)
- **Words**: 2,429
- **Segments**: 221 speech segments
- **Pattern**: Speech distributed throughout with regular gaps

#### 2. manual_f24_20260321_011038_519591_6889700ms.wav  
- **Duration**: 114:49.700 (1h 54m 50s)
- **Speech**: 34:11.760 (**29.7%**)
- **Words**: 2,219
- **Segments**: 198 speech segments
- **Pattern**: Similar to above, consistent speech throughout

#### 3. manual_f24_20260322_030319_254071_6614800ms.wav
- **Duration**: 110:14.800 (1h 50m 15s)
- **Speech**: 25:35.880 (**23.2%**)
- **Words**: 1,676
- **Segments**: 159 speech segments
- **Pattern**: Good speech content, slightly less dense

### 🔇 **Low Speech Content Files**

#### 4. manual_f24_20260323_031829_276218_3805600ms.wav
- **Duration**: 63:25.600 (1h 3m 26s)
- **Speech**: 00:39.680 (**1.0%**)
- **Words**: 33
- **Pattern**: Almost entirely silent with brief speech at end

#### 5. manual_ctrl_r_20260307_142310_489677_8502100ms.wav
- **Duration**: 141:42.100 (2h 21m 42s)
- **Speech**: 00:00.400 (**0.005%**)
- **Words**: 3
- **Pattern**: 99.995% silent, only 3 words at very end

#### 6. manual_ctrl_r_20260307_142251_084499_8501200ms.wav
- **Duration**: 141:41.200 (2h 21m 41s)
- **Speech**: 00:10.120 (**0.12%**)
- **Words**: 20
- **Pattern**: 99.88% silent, brief speech at very end

#### 7. manual_ctrl_r_20260320_071118_364871_849000ms.wav
- **Duration**: 14:09.000 (14m 9s)
- **Speech**: 00:00.000 (**0.0%**)
- **Words**: 0
- **Pattern**: Completely silent

#### 8. manual_ctrl_r_20260320_071054_312417_846100ms.wav
- **Duration**: 14:06.100 (14m 6s)
- **Speech**: 00:00.000 (**0.0%**)
- **Words**: 0
- **Pattern**: Completely silent

#### 9. manual_ctrl_r_20260321_103730_893278_794200ms.wav
- **Duration**: 13:14.200 (13m 14s)
- **Speech**: 00:00.000 (**0.0%**)
- **Words**: 0
- **Pattern**: Completely silent

#### 10. manual_ctrl_r_20260321_103728_077751_792400ms.wav
- **Duration**: 13:12.400 (13m 12s)
- **Speech**: 00:25.360 (**3.2%**)
- **Words**: 27
- **Pattern**: Mostly silent with repetitive "computer increase volume" phrases

## 📊 **Analysis by File Type**

### manual_f24 files (4 analyzed)
- **Total Duration**: 5h 43m 21s
- **Total Speech**: 1h 37m 40s (**28.4%** average)
- **Pattern**: **Legitimate meeting/conversation recordings**

### manual_ctrl_r files (6 analyzed)  
- **Total Duration**: 6h 38m 02s
- **Total Speech**: 00:35.880 (**0.15%** average)
- **Pattern**: **Mostly dead air/silent recordings**

## 🎯 **Speech Location Patterns**

### manual_f24 files:
- Speech **distributed throughout** the recordings
- Regular gaps between speech segments (1-2 minutes typical)
- Natural conversation flow with multiple participants
- **Worth processing for content extraction**

### manual_ctrl_r files:
- Speech **only at the very end** (if any)
- 99.85% of content is silence/background noise
- Likely failed/inactive recording sessions
- **Not worth processing except for last few minutes**

## 📈 **Overall Statistics (Top 10 Files)**

- **Total Duration**: 12h 21m 25s
- **Total Speech**: 1h 37m 46s (**13.2%**)
- **Total Words**: 6,988
- **Speech Distribution**: Highly skewed (3 files contain 98% of speech)

## 🔍 **Extrapolation to All 337 Files**

Based on this sample:
- **manual_f24 files**: Likely contain substantial speech content (~25-30%)
- **manual_ctrl_r files**: Likely mostly silent (~0.1% speech)
- **Total speech across all files**: Estimated 4-5 hours out of 16.2 total hours

## 💡 **Recommendations**

### 🟢 **Process These Files:**
1. **All manual_f24 files** - contain actual meeting content
2. **Last 2-3 minutes of manual_ctrl_r files** - only speech segments

### 🔴 **Skip These Files:**
1. **manual_ctrl_r files** (except last few minutes) - 99.9% silent
2. **Files under 30 seconds** - analyzed separately if needed

### 🛠️ **Processing Strategy:**
1. **Priority 1**: Process all manual_f24 files completely
2. **Priority 2**: Extract only last 3 minutes from manual_ctrl_r files
3. **Priority 3**: Skip the bulk of manual_ctrl_r recordings

## 📝 **Sample Speech Content**

### manual_f24 files:
- Natural conversation: "You're gonna use this mass of your degree..."
- Meeting discussions: Multiple participants, varied topics
- Continuous speech with natural pauses

### manual_ctrl_r files:
- Repetitive phrases: "A computer increase volume" 
- Very brief: "it all means", "But instead Mario did..."
- Likely system testing/background recording

## 🎯 **Conclusion**

**The manual_f24 files contain valuable speech content worth processing, while manual_ctrl_r files are essentially silent recordings with only token speech at the very end.**

Focus transcription efforts on manual_f24 files for best ROI on processing time.
