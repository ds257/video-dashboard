import pandas as pd

df = pd.read_csv("final_with_motion_features.csv", low_memory=False)

def classify_sound(row):
    try:
        video_id = int(float(row.get("video_id", 0)))
        music_id_raw = row.get("music_id", "")
        
        # Clean music_id
        if pd.isna(music_id_raw):
            return "External Music"
        try:
            music_id = int(float(music_id_raw))
        except:
            return "External Music"
        
        is_original_col = str(row.get("is_original_sound")).strip().upper() == "TRUE"

        # Rule 1: Explicit flag
        if is_original_col:
            return "Original Sound"

        # Rule 2: If music_id is very close numerically to video_id
        if abs(music_id - video_id) < 10000:
            return "Original Sound"

        return "External Music"
    except Exception as e:
        print(f"Error on row {row.name}: {e}")
        return "Unknown"

df["sound_type"] = df.apply(classify_sound, axis=1)
df.to_csv("final_with_sound.csv", index=False)

print("✅ Smart classification done using numerical proximity!")
