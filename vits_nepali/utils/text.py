# utils/text.py
# from typing import List
# import logging

# logger = logging.getLogger(__name__)

# def text_to_phonemes(text: str, language: str = 'ne') -> List[int]:
#     """Convert Nepali text to phoneme indices (placeholder)."""
#     try:
#         # Placeholder: Map Nepali characters to indices
#         # Replace with a proper Nepali phonemizer or G2P system
#         phoneme_map = {c: i for i, c in enumerate("अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")}
#         return [phoneme_map.get(c, 0) for c in text]
#     except Exception as e:
#         logger.error(f"Failed to phonemize text '{text}': {str(e)}")
#         raise


import logging
from typing import List

logger = logging.getLogger(__name__)

# Grapheme-to-phoneme mapping (for inference)
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

# Phoneme vocabulary (phonemes to indices)
phoneme_vocab = {
    'ə': 0, 'aː': 1, 'i': 2, 'iː': 3, 'u': 4, 'uː': 5, 'eː': 6, 'ai': 7, 'oː': 8, 'au': 9,
    'k': 10, 'kʰ': 11, 'g': 12, 'gʱ': 13, 'ŋ': 14,
    't͡ʃ': 15, 't͡ʃʰ': 16, 'd͡ʒ': 17, 'd͡ʒʱ': 18, 'ɲ': 19,
    'ʈ': 20, 'ʈʰ': 21, 'ɖ': 22, 'ɖʱ': 23, 'ɳ': 24,
    't̪': 25, 't̪ʰ': 26, 'd̪': 27, 'd̪ʱ': 28, 'n': 29,
    'p': 30, 'pʰ': 31, 'b': 32, 'bʱ': 33, 'm': 34,
    'j': 35, 'ɾ': 36, 'l': 37, 'ʋ': 38,
    'ʃ': 39, 'ʂ': 40, 's': 41, 'ɦ': 42,
    '̃': 43, 'ʰ': 44
}

def text_to_phonemes(text: str, language: str = 'ne') -> List[int]:
    """
    Convert Nepali text to a list of phoneme indices.
    
    Args:
        text (str): Input Nepali text (e.g., "नमस्ते") or space-separated phonemes (e.g., "n m s t̪ eː").
        language (str): Language code (default: 'ne' for Nepali).
    
    Returns:
        List[int]: List of phoneme indices (e.g., [29, 34, 41, 25, 6]).
    """
    try:
        # Check if input is already a phoneme string
        if all(word in phoneme_vocab for word in text.split()):
            phonemes = text.split()
        else:
            # Convert text to phoneme string
            phoneme_str = ' '.join(grapheme_to_phoneme.get(ch, ch) for ch in text if ch in grapheme_to_phoneme)
            phonemes = [p for p in phoneme_str.split() if p]
        
        # Map phonemes to indices
        phoneme_indices = [phoneme_vocab.get(p, 0) for p in phonemes]
        logger.debug(f"Text: {text}, Phonemes: {phonemes}, Indices: {phoneme_indices}")
        return phoneme_indices
    except Exception as e:
        logger.error(f"Failed to phonemize text '{text}': {str(e)}")
        raise