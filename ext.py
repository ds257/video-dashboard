import numpy as np
import pandas as pd
import librosa
import os
import cv2

# Fix deprecated np.complex
np.complex = complex

# Load the dataset
df = pd.read_csv("final_with_multimodal.csv")

# -------- AUDIO COMPLEXITY --------
def compute_audio_features(audio_path):
    try:
        if not isinstance(audio_path, str) or not os.path.exists(audio_path):
            print(f" Missing audio: {audio_path}")
            return pd.Series([np.nan] * 4)
        y, sr = librosa.load(audio_path, sr=None)
        rms = librosa.feature.rms(y=y)[0]
        return pd.Series([
            round(rms.mean(), 4),
            round(rms.std(), 4),
            round(rms.min(), 4),
            round(rms.max(), 4)
        ])
    except Exception as e:
        print(f"⚠️ Error processing {audio_path}: {e}")
        return pd.Series([np.nan] * 4)

# -------- MULTIMODAL OVERLOAD --------
def compute_multimodal_score(row):
    score = 0
    if row.get('has_music') == True: score += 1
    if row.get('has_speech') == True: score += 1
    desc = str(row.get('video_description_x', ''))
    if any(char in desc for char in ['#', '@', '✨', '🔥', '🎵', '‼️', '😮']): score += 1
    transcript = str(row.get('whisper_voice_to_text', ''))
    if len(transcript.split()) > 10: score += 1
    return round(score / 4, 2)

# -------- PERSUASIVE LANGUAGE DETECTION --------
persuasive_keywords = [
    "subscribe", "follow", "share", "vote", "click", "buy", "support",
    "watch", "now", "don’t miss", "act fast", "you need to", "must see"
]

def is_persuasive(text):
    if not isinstance(text, str): return False
    text = text.lower()
    return any(kw in text for kw in persuasive_keywords)

# -------- PER-PIXEL MOTION INTENSITY --------
def compute_motion_intensity(video_filename):
    video_path = os.path.join("videos_by_date", video_filename)
    try:
        if not os.path.exists(video_path):
            print(f"❌ Missing video: {video_path}")
            return pd.Series([np.nan, np.nan])
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()
        if not ret:
            return pd.Series([np.nan, np.nan])
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        magnitudes = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(np.mean(mag))
            prev_gray = gray
        cap.release()
        return pd.Series([
            round(np.mean(magnitudes), 4),
            round(np.std(magnitudes), 4)
        ])
    except Exception as e:
        print(f"⚠️ Motion error in {video_path}: {e}")
        return pd.Series([np.nan, np.nan])

# -------- FEATURE PIPELINE --------
print("🔊 Extracting audio features...")
df[['volume_mean', 'volume_std', 'volume_min', 'volume_max']] = df['audio_file_path'].apply(compute_audio_features)

print("📺 Computing multimodal overload...")
df['multimodal_score'] = df.apply(compute_multimodal_score, axis=1)

print("🧠 Detecting persuasive messaging...")
df['is_persuasive'] = df['whisper_voice_to_text'].apply(is_persuasive)

# -------- SAVE FINAL OUTPUT --------
df.to_csv("final_with_all_features.csv", index=False)
print("✅ Done! Saved to final_with_all_features.csv")
