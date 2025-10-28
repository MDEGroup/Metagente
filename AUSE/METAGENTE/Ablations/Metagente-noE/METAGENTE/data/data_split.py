import pandas as pd
from sklearn.model_selection import train_test_split

# Đường dẫn tới file CSV gốc
file_path = "high_sim_short_description_085_above.csv"

# Đọc file
df = pd.read_csv(file_path)

# Đổi tên cột similarity cho dễ thao tác
df["similarity"] = df["cosine_similarity_embed"]


# Hàm phân loại theo bin similarity
def bin_similarity(score):
    if score < 0.9:
        return "0.85-0.9"
    else:
        return "0.9+"


# Áp dụng binning
df["similarity_bin"] = df["similarity"].apply(bin_similarity)

# Chọn các cột cần thiết
selected_columns = df[
    ["description_html_clean", "description_short", "similarity", "similarity_bin"]
]

# Chia dữ liệu train/test có phân tầng theo similarity_bin
train_df, test_df = train_test_split(
    selected_columns,
    test_size=0.2,
    random_state=42,
    stratify=selected_columns["similarity_bin"],
)

# Lưu ra file CSV
train_df.to_csv("train_descriptions_stratified.csv", index=False)
test_df.to_csv("test_descriptions_stratified.csv", index=False)

# Kiểm tra phân phối
print(
    "Train distribution:\n",
    train_df["similarity_bin"].value_counts(normalize=True).sort_index(),
)
print(
    "\nTest distribution:\n",
    test_df["similarity_bin"].value_counts(normalize=True).sort_index(),
)
