import subprocess
import ast
import numpy as np

# Define emphasizer types
emphasizer_types = [  "NEO", "raw","ED","delta"]

# Placeholder to store accuracy vectors
all_accuracies = []

# Path to the script
script_path = "quiroga_find_optimal_thr_factor.py"

# Run the script for each emphasizer_type
for emphasizer_type in emphasizer_types:
    # Call the script using subprocess
    result = subprocess.run(
        ["python", script_path, "--emphasizer_type", emphasizer_type],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Check for errors
    if result.returncode != 0:
        print(f"Error running script for {emphasizer_type}: {result.stderr}")
        continue
    # 获取最后一行输出
    output_lines = result.stdout.strip().split('\n')  # 按行分割输出
    last_line = output_lines[-1]  # 获取最后一行

    # 打印最后一行内容，检查格式
    print("Last line:", last_line)

    # 尝试解析最后一行内容
    try:
        acc = ast.literal_eval(last_line.strip())  # 解析为 Python 列表
        print("Accessed 'acc' from the script output:", acc)
        all_accuracies.append(acc)
        print("All accuracies so far:", all_accuracies)
    except Exception as e:
        print(f"Error parsing output: {e}")


np.savetxt("all_accuracies_traditional.csv", all_accuracies, delimiter=",", header=",".join(emphasizer_types), comments="")
#np.savetxt("all_accuracies_DVT.csv", all_accuracies, delimiter=",", header=",".join(emphasizer_types), comments="")
breakpoint()


