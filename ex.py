import pandas as pd

df = pd.read_csv("final_with_motion_features.csv")

# Heuristic overload: music + speech + emojis/symbols in description
df["has_multimodal_overload"] = (
    df["has_speech"].astype(bool) &
    df["has_music"].astype(bool) &
    df["video_description_x"].fillna("").str.contains(r"[^\w\s]", regex=True)
)

# Save updated file
df.to_csv("final_with_multimodal.csv", index=False)

