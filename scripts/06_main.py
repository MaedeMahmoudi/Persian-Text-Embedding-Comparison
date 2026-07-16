import subprocess
import sys
import os

scripts = ["01_preprocessing.py", "02_tfidf.py", "03_svd.py", "04_embedding.py", "05_similarity.py"]

for script in scripts:
    if not os.path.exists(script):
        print(f"Error: {script} not found")
        sys.exit(1)
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print(f"Failed at {script}")
        sys.exit(1)

print("All steps completed")