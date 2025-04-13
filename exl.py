import pandas as pd

# Load your CSV
df = pd.read_csv("final_with_all_features.csv")

def normalize_music_id(music_id):
    if pd.isna(music_id):
        return ""
    # Remove decimal point and 'E+18' if present
    cleaned = str(music_id).split("E")[0].replace('.', '')
    return cleaned

def check_original(row):
    video_id = str(row["video_id"])
    music_id = normalize_music_id(row["music_id"])
    return music_id.startswith(video_id[:7])

# Clean and compute
df["is_original_sound"] = df.apply(check_original, axis=1)

# Save output
df.to_csv("final_with_original_sound.csv", index=False)
print("✅ Saved with 'is_original_sound' added.")

