import json

def analyze_channel_differences():
    # Load the report
    with open(r'i:\whisper-acft\channel_isolation_report.json', 'r') as f:
        data = json.load(f)

    # Find files with significant channel differences
    significant_files = []
    
    for file_path, decision in data['decisions'].items():
        margin = decision['margin']
        sims = decision['sims_by_channel']
        
        # Calculate percentage difference
        if len(sims) >= 2:
            max_sim = max(sims)
            min_sim = min(sims)
            if max_sim > 0:
                diff_percentage = (max_sim - min_sim) / max_sim * 100
            else:
                diff_percentage = 0
            
            # Consider significant if margin > 0.05 or diff > 10%
            if margin > 0.05 or diff_percentage > 10:
                significant_files.append({
                    'file': file_path.split('\\')[-1],  # Just filename
                    'margin': margin,
                    'diff_percentage': diff_percentage,
                    'channel_index': decision['channel_index'],
                    'sims': sims,
                    'selected_channel': 'Right' if decision['channel_index'] == 1 else 'Left'
                })

    # Sort by margin (descending)
    significant_files.sort(key=lambda x: x['margin'], reverse=True)

    print('Files with Significant Left/Right Channel Differences:')
    print('=' * 60)
    for i, file_info in enumerate(significant_files[:15], 1):  # Top 15
        print(f'{i:2d}. {file_info["file"]}')
        print(f'    Margin: {file_info["margin"]:.4f} ({file_info["diff_percentage"]:.1f}% difference)')
        print(f'    Selected: {file_info["selected_channel"]} channel')
        print(f'    Similarities: Left={file_info["sims"][0]:.3f}, Right={file_info["sims"][1]:.3f}')
        print()

    print(f'Found {len(significant_files)} files with significant channel differences')
    
    # Also show files with very small differences (potential issues)
    print('\nFiles with Very Small Channel Differences (Potential Issues):')
    print('=' * 60)
    small_diff_files = []
    
    for file_path, decision in data['decisions'].items():
        margin = decision['margin']
        sims = decision['sims_by_channel']
        
        if len(sims) >= 2 and margin < 0.01:  # Very small margin
            max_sim = max(sims)
            min_sim = min(sims)
            if max_sim > 0:
                diff_percentage = (max_sim - min_sim) / max_sim * 100
            else:
                diff_percentage = 0
                
            small_diff_files.append({
                'file': file_path.split('\\')[-1],
                'margin': margin,
                'diff_percentage': diff_percentage,
                'sims': sims
            })
    
    small_diff_files.sort(key=lambda x: x['margin'])
    
    for i, file_info in enumerate(small_diff_files[:10], 1):
        print(f'{i:2d}. {file_info["file"]}')
        print(f'    Margin: {file_info["margin"]:.4f} ({file_info["diff_percentage"]:.1f}% difference)')
        print(f'    Similarities: Left={file_info["sims"][0]:.3f}, Right={file_info["sims"][1]:.3f}')
        print()

if __name__ == "__main__":
    analyze_channel_differences()
