import csv
import sys

import pandas as pd

csv.field_size_limit(sys.maxsize)

# Path to your file
file_path = "reproduce-table1-our-tool.txt"


# Read CSV-like text file (comma-separated)
df = pd.read_csv(file_path, sep=",", engine="python", skipinitialspace=True)

print(df.groupby("bench"))

# Select only relevant columns
metrics = ["acc", "f1", "size"]

# Compute mean and standard deviation per benchmark
results = df.groupby("bench")[metrics].agg(["mean", "std"])

# Print neatly
pd.set_option("display.float_format", "{:.4f}".format)
print(results)


for f in ["acc", "size", "f1"]:
    for i in range(len(results[f]["mean"])):
        print(
            "${:.2f} \\pm  {:.2f}$ & ".format(
                results[f]["mean"][i], results[f]["std"][i]
            )
        )
    print("\\\\")

