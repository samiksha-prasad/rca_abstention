"""
Run this AFTER downloading an RCAEval dataset (RE1 or RE2 -- see
DOWNLOAD_RCAEVAL.md). It finds the first downloaded case folder and prints
its structure so we can verify (and fix, if needed) the assumptions in
src/data_loader.py. Paste the output of this script back into the chat.

Works for both RE1 (metrics only, file is "data.csv") and RE2 (metrics +
logs + traces, file is "metrics.csv" instead -- discovered by inspecting a
real download, since this differs from RE1's naming).
"""
import os
import sys

DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/RE1"

# RE1 calls its main metrics file "data.csv"; RE2 calls it "metrics.csv".
MAIN_METRICS_FILENAMES = ["data.csv", "metrics.csv"]


def main():
    if not os.path.exists(DATA_ROOT):
        print(f"'{DATA_ROOT}' doesn't exist yet -- did you run the download step?")
        return

    # Find the first case folder (one that has data.csv OR metrics.csv in it)
    case_dir = None
    found_filename = None
    for root, dirs, files in os.walk(DATA_ROOT):
        for fname in MAIN_METRICS_FILENAMES:
            if fname in files:
                case_dir = root
                found_filename = fname
                break
        if case_dir:
            break

    if case_dir is None:
        print(f"No {MAIN_METRICS_FILENAMES} found anywhere under {DATA_ROOT}. "
              f"Listing top-level contents instead:")
        print(os.listdir(DATA_ROOT))
        return

    print(f"Found case folder: {case_dir}  (main metrics file: {found_filename})\n")
    print("ALL files in this case folder:")
    for f in sorted(os.listdir(case_dir)):
        path = os.path.join(case_dir, f)
        size = os.path.getsize(path)
        print(f"  {f}  ({size} bytes)")

    metrics_path = os.path.join(case_dir, found_filename)
    print(f"\n--- {found_filename}: first 3 lines (raw) ---")
    with open(metrics_path) as f:
        for _ in range(3):
            line = f.readline()
            if not line:
                break
            print(line.rstrip()[:500])  # truncate very wide lines

    print(f"\n--- {found_filename}: parsed with pandas ---")
    try:
        import pandas as pd
        df = pd.read_csv(metrics_path)
        print(f"Shape: {df.shape}")
        print(f"Columns ({len(df.columns)} total):")
        print(list(df.columns)[:20], "..." if len(df.columns) > 20 else "")
    except Exception as e:
        print(f"Couldn't parse with pandas: {e}")

    inject_time_path = os.path.join(case_dir, "inject_time.txt")
    if os.path.exists(inject_time_path):
        with open(inject_time_path) as f:
            print(f"\n--- inject_time.txt ---\n{f.read().strip()}")

    for extra_file in ["logs.csv", "traces.csv", "simple_metrics.csv", "logts.csv",
                       "tracets_lat.csv", "tracets_err.csv"]:
        extra_path = os.path.join(case_dir, extra_file)
        if os.path.exists(extra_path):
            print(f"\n--- {extra_file} found! ({os.path.getsize(extra_path)} bytes) ---")
            print("First 3 lines (raw):")
            with open(extra_path) as f:
                for _ in range(3):
                    line = f.readline()
                    if not line:
                        break
                    print(line.rstrip()[:500])
            try:
                import pandas as pd
                df_extra = pd.read_csv(extra_path)
                print(f"Parsed shape: {df_extra.shape}, columns: {list(df_extra.columns)[:15]}")
            except Exception as e:
                print(f"Couldn't parse {extra_file} with pandas: {e}")

    print(f"\n--- Folder path breakdown ---")
    parent = os.path.basename(os.path.dirname(case_dir))  # e.g. "cartservice_cpu"
    instance = os.path.basename(case_dir)                  # e.g. "1"
    print(f"Parent folder (service_fault): {parent}")
    print(f"Instance folder: {instance}")


if __name__ == "__main__":
    main()