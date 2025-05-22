import csv
import os

# Reuse G2P mapping from utils/text.py (copied here for clarity)
grapheme_to_phoneme = {
    'अ': 'ə', 'आ': 'aː', 'इ': 'i', 'ई': 'iː', 'उ': 'u', 'ऊ': 'uː',
    'ए': 'eː', 'ऐ': 'ai', 'ओ': 'oː', 'औ': 'au',
    'क': 'k', 'ख': 'kʰ', 'ग': 'g', 'घ': 'gʱ', 'ङ': 'ŋ',
    'च': 't͡ʃ', 'छ': 't͡ʃʰ', 'ज': 'd͡ʒ', 'झ': 'd͡ʒʱ', 'ञ': 'ɲ',
    'ट': 'ʈ', 'ठ': 'ʈʰ', 'ड': 'ɖ', 'ढ': 'ɖʱ', 'ण': 'ɳ',
    'त': 't̪', 'थ': 't̪ʰ', 'द': 'd̪', 'ध': 'd̪ʱ', 'न': 'n',
    'प': 'p', 'फ': 'pʰ', 'ब': 'b', 'भ': 'bʱ', 'म': 'm',
    'य': 'j', 'र': 'ɾ', 'ल': 'l', 'व': 'ʋ',
    'श': 'ʃ', 'ष': 'ʂ', 'स': 's', 'ह': 'ɦ',
    'ा': 'aː', 'ि': 'i', 'ी': 'iː', 'ु': 'u', 'ू': 'uː',
    'े': 'eː', 'ै': 'ai', 'ो': 'oː', 'ौ': 'au',
    'ं': '̃', 'ः': 'ʰ', 'ँ': '̃', '्': '',  # Halant
    ' ': ' '
}

def nepali_to_phonemes(text):
    return ' '.join(grapheme_to_phoneme.get(ch, ch) for ch in text if ch in grapheme_to_phoneme)

# File paths
csv_files = [
    "vits_nepali/data/csv/train.csv",
    "vits_nepali/data/csv/val.csv",
    "vits_nepali/data/csv/test.csv"
]

for csv_file in csv_files:
    if not os.path.exists(csv_file):
        print(f"Skipping {csv_file}: File not found")
        continue
    
    output_csv = csv_file.replace(".csv", "_phonemes.csv")
    
    with open(csv_file, mode='r', encoding='utf-8') as infile, \
         open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = ['path', 'labels', 'phonemes']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            label = row['labels']
            phonemes = nepali_to_phonemes(label)
            writer.writerow({
                'path': row['path'],
                'labels': label,
                'phonemes': phonemes
            })
    
    print(f"✅ Phoneme conversion complete for {csv_file}. Output saved to: {output_csv}")