import os
import pandas as pd

# 设置数据目录路径
data_dir = "F:/Project/Dataset/MachineLearningCVE"  # ← 请根据你的实际路径调整

# 获取目录下所有以 .csv 结尾的文件
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# 合并所有CSV文件
combined_df_list = []
for file in csv_files:
    file_path = os.path.join(data_dir, file)
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df["source_file"] = file  # 添加一列标记来源文件
        combined_df_list.append(df)
        print(f"✔ Loaded: {file} — shape: {df.shape}")
    except Exception as e:
        print(f"❌ Failed to load {file}: {e}")

# 合并为一个大 DataFrame
combined_df = pd.concat(combined_df_list, ignore_index=True)
print(f"\n✅ 合并完成，共计样本数：{len(combined_df)}，特征数：{len(combined_df.columns)}")

# 保存为新的文件
output_file = os.path.join(data_dir, "cicids2017_combined.csv")
combined_df.to_csv(output_file, index=False)
print(f"📁 已保存到：{output_file}")
