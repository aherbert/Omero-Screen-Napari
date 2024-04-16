import polars as pl

# Example DataFrame
df = pl.DataFrame({
    "images": ["cat", "cat", "dog", "dog", "bird", "bird"],
    "centroid-0": [1.0, 2.5, 1.2, 2.1, 2.8, 2.2],
    "centroid-1": [0.5, 0.4, 0.6, 0.3, 0.5, 0.2]
})
print(df["images"].unique())
centroid_dict = {}
for image_id in df["images"].unique():
    filtered_df = df.filter(df["images"] == image_id)
    centroid_dict[image_id] = (filtered_df["centroid-0"].to_list(), filtered_df["centroid-1"].to_list())

print(centroid_dict)