import pandas as pd

# Load the CSV
df = pd.read_csv("final_with_all_features.csv")

# Define logic: if music_id is missing, short, or starts with a special pattern, it might be original
def is_original(row):
    music_id = str(row.get('music_id', '')).lower()
    if music_id == '' or music_id == 'nan':
        return True
    if len(music_id) < 10:
        return True
    return False

# Apply the rule
df['is_original_sound'] = df.apply(is_original, axis=1)

# Save to new CSV
df.to_csv("final_with_original_sound.csv", index=False)

print("✅ Saved with `is_original_sound` column added.")

