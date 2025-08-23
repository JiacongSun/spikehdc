import csv

# Paths to your CSV files
file1 = 'response/all_accuracies_traditional.csv'
file2 = 'response/all_accuracies_DVT.csv'
transposed_file = 'response/all_accuracy_merged_transposed_reordered.csv'

header1 = ['NEO','AT','ED','delta','NEO-DVT','DVT','ED-DVT','delta-DVT']

# Reordered header
header2 = ['AT', 'DVT', 'ED', 'ED-DVT', 'delta', 'delta-DVT', 'NEO', 'NEO-DVT']

# actually used header (delta ->Δ)
header3 = ['AT', 'DVT', 'ED', 'ED-DVT', 'Δ', 'Δ-DVT', 'NEO', 'NEO-DVT']

# Read and merge the files
data = []

with open(file1, 'r') as f1, open(file2, 'r') as f2:
    reader1 = csv.reader(f1)
    reader2 = csv.reader(f2)
    
    # Skip the first line (header) of both files
    next(reader1, None)
    next(reader2, None)
    
    # Collect all rows from both files
    for row in reader1:
        data.append(row)
    for row in reader2:
        data.append(row)

# Transpose the data
transposed_data = list(zip(*data))

# Reorder the transposed data to match the new header order
# Create a mapping of header1 indices to header2 order
index_map = [header1.index(h) for h in header2]
reordered_data = [[row[i] for i in index_map] for row in transposed_data]


# Save the transposed data
with open(transposed_file, 'w', newline='') as tf:
    
    writer = csv.writer(tf)
    writer.writerow(header3)
    writer.writerows(reordered_data)

print(f"Transposed/reordered data saved into {transposed_file}")
