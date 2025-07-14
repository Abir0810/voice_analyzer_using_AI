# import librosa
# import numpy as np
# from scipy.signal import correlate

# def extract_mfcc(filepath):
#     y, sr = librosa.load(filepath)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
#     return np.mean(mfcc, axis=1)

# def compare_audio(file1, file2, threshold=0.9):
#     mfcc1 = extract_mfcc(file1)
#     mfcc2 = extract_mfcc(file2)

#     corr = np.corrcoef(mfcc1, mfcc2)[0, 1]
#     return corr > threshold

# # Example usage
# result = compare_audio(r"D:\other\voice analyzer\har.wav", r"D:\other\voice analyzer\harvard.wav")
# print("Match:", result)

import librosa
import numpy as np

def extract_mfcc(file, y=None, sr=None, n_mfcc=20):
    if y is None or sr is None:
        y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

def match_partial(reference_file, long_audio_file, threshold=0.85, hop_seconds=5):
    # Load reference audio
    ref_y, ref_sr = librosa.load(reference_file, sr=None)
    ref_mfcc = extract_mfcc(reference_file, y=ref_y, sr=ref_sr)
    ref_duration = librosa.get_duration(y=ref_y, sr=ref_sr)

    # Load test audio
    test_y, test_sr = librosa.load(long_audio_file, sr=None)
    test_duration = librosa.get_duration(y=test_y, sr=test_sr)

    segment_len = int(ref_sr * ref_duration)  # same length as reference
    hop_len = int(ref_sr * hop_seconds)

    print(f"Reference length: {ref_duration:.2f}s, Test length: {test_duration:.2f}s")

    for i in range(0, len(test_y) - segment_len + 1, hop_len):
        segment = test_y[i:i + segment_len]
        seg_mfcc = extract_mfcc(long_audio_file, y=segment, sr=test_sr)
        sim = np.corrcoef(ref_mfcc, seg_mfcc)[0, 1]
        print(f"Segment {i/test_sr:.2f}s - Similarity: {sim:.4f}")
        if sim > threshold:
            print("✅ Match found!")
            return True

    print("❌ No match found.")
    return False

# === Example Usage ===
reference_file = r"D:\other\voice analyzer\GPvoice.wav"
test_file = r"D:\other\voice analyzer\GPvoice.wav"

result = match_partial(reference_file, test_file)
print("Match:", result)