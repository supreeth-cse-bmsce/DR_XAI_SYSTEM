import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

base_path = "data"
csv_path = os.path.join(base_path, "train.csv")
image_path = os.path.join(base_path, "train_images")

df = pd.read_csv(csv_path)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["diagnosis"],
    random_state=42
)

def organize_data(dataframe, split):
    for _, row in dataframe.iterrows():
        label = str(row["diagnosis"])
        img_name = row["id_code"] + ".png"

        src = os.path.join(image_path, img_name)
        dst = os.path.join(base_path, split, label, img_name)

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

organize_data(train_df, "train")
organize_data(val_df, "val")

print("Dataset organized successfully.")