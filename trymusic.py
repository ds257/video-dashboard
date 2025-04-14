import pandas as pd

# Use the correct filename that exists in your folder
df = pd.read_csv("final_with_motion_features.csv", low_memory=False)

# Helper function to classify original vs external sound
def classify_sound(row):
    if row.get("is_original_sound") == True:
        return "Original Sound"
    elif pd.notna(row.get("music_id")) and str(row["music_id"]).startswith(str(row["video_id"])):
        return "Original Sound"
    else:
        return "External Music"

# Add new column
df["sound_type"] = df.apply(classify_sound, axis=1)

# Save to a new CSV
df.to_csv("final_with_sound_classification.csv", index=False)

print("✅ Sound type classification complete! File saved as 'final_with_sound_classification.csv'")

