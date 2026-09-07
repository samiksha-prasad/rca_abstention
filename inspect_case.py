"""
Run this AFTER downloading RCAEval RE1 (see DOWNLOAD_RCAEVAL.md).

It finds the first downloaded case folder and prints its structure so we can
verify (and fix, if needed) the assumptions in src/data_loader.py's
load_rcaeval_case(). Paste the output of this script back into the chat.

NOTE: updated after discovering the real folder structure is
  data/RE1/RE1-OB/{service}_{fault}/{instance}/data.csv
  data/RE1/RE1-OB/{service}_{fault}/{instance}/inject_time.txt
rather than the {benchmark}_{service}_{fault}_{instance}/metrics.json
structure the official docs described.
"""
import os

DATA_ROOT = "data/RE1"


def main():
    if not os.path.exists(DATA_ROOT):
        print(f"'{DATA_ROOT}' doesn't exist yet -- did you run the download step?")
        return

    # Find the first case folder (one that has data.csv in it)
    case_dir = None
    for root, dirs, files in os.walk(DATA_ROOT):
        if "data.csv" in files:
            case_dir = root
            break

    if case_dir is None:
        print(f"No 'data.csv' found anywhere under {DATA_ROOT}. "
              f"Listing top-level contents instead:")
        print(os.listdir(DATA_ROOT))
        return

    print(f"Found case folder: {case_dir}\n")
    print("Files in this case folder:")
    for f in os.listdir(case_dir):
        path = os.path.join(case_dir, f)
        size = os.path.getsize(path)
        print(f"  {f}  ({size} bytes)")

    print("\n--- data.csv: first 5 lines (raw) ---")
    with open(os.path.join(case_dir, "data.csv")) as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            print(line.rstrip())

    print("\n--- data.csv: parsed with pandas ---")
    try:
        import pandas as pd
        df = pd.read_csv(os.path.join(case_dir, "data.csv"))
        print(f"Shape: {df.shape}")
        print(f"Columns ({len(df.columns)} total):")
        print(list(df.columns)[:20], "..." if len(df.columns) > 20 else "")
        print("\nFirst 3 rows:")
        print(df.head(3).to_string())
    except Exception as e:
        print(f"Couldn't parse with pandas: {e}")

    inject_time_path = os.path.join(case_dir, "inject_time.txt")
    if os.path.exists(inject_time_path):
        with open(inject_time_path) as f:
            print(f"\n--- inject_time.txt ---\n{f.read().strip()}")

    print(f"\n--- Folder path breakdown ---")
    parent = os.path.basename(os.path.dirname(case_dir))  # e.g. "cartservice_cpu"
    instance = os.path.basename(case_dir)                  # e.g. "1"
    print(f"Parent folder (service_fault): {parent}")
    print(f"Instance folder: {instance}")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
