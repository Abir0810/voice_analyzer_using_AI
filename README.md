This Python script checks whether a short reference audio clip exists within a longer audio file, using MFCC features and cosine similarity.

# 📦 Requirements
Python 3.x

librosa, numpy

# Install dependencies:

pip install librosa numpy

# ▶️ How It Works
Extracts MFCC features from both reference and test audio.

Slides a window across the test audio to compute similarity.

Prints similarity scores and stops when a match (above threshold) is found.

# ⚙️ Parameters
threshold – Similarity cutoff (default: 0.85)

hop_seconds – Step size for sliding window (default: 5 seconds)

n_mfcc – Number of MFCC coefficients (default: 20)

# ✅ Output
Prints similarity scores for each segment.

Prints ✅ Match found! if similarity > threshold.

Returns True if matched, else False.
