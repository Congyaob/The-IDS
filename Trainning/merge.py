import os
import pandas as pd

data_dir = "F:/Project/Dataset/MachineLearningCVE"  

csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

combined_df_list = []
for file in csv_files:
    file_path = os.path.join(data_dir, file)
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df["source_file"] = file  
        combined_df_list.append(df)
        print(f"✔ Loaded: {file} — shape: {df.shape}")
    except Exception as e:
        print(f"❌ Failed to load {file}: {e}")

combined_df = pd.concat(combined_df_list, ignore_index=True)
print(f"\n✅ The merge is completed, the total number of samples：{len(combined_df)}，Number of features：{len(combined_df.columns)}")

output_file = os.path.join(data_dir, "cicids2017_combined.csv")
combined_df.to_csv(output_file, index=False)
print(f"📁 Saved as：{output_file}")

