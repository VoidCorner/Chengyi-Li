import pandas as pd

df = pd.read_csv("./data/raw_vicon_data.csv")

df = df[["joint_x", "joint_y", "joint_z", "label"]]
df.to_csv("./data/motion_data.csv", index=False)

print("Data preprocessing completed!")
