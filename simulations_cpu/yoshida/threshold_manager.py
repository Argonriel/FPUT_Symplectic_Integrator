import argparse
import concurrent.futures
import os
import subprocess
import sys

import pandas as pd


def run_task(args):
    binary, N, model, value, amplitude, outdir, logdir = args
    outname = f"{model}_N{N}_A{amplitude:.4f}.csv"
    outpath = os.path.join(outdir, outname)
    logpath = os.path.join(logdir, outname.replace(".csv", ".log"))

    if os.path.exists(outpath):
        return ("skipped", outname)

    cmd = [binary, str(N), model, str(value), str(amplitude), outpath,
           "--dt", "0.1", "--stride", "5000", "--nseg", "200", "--entropy"]

    try:
        with open(logpath, "w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            return ("failed", outname)
        return ("done", outname)
    except Exception as e:
        try:
            with open(logpath, "a") as log:
                log.write(f"\nEXCEPTION in run_task: {e}\n")
        except Exception:
            pass
        return ("failed", outname)


def main():
    parser = argparse.ArgumentParser(description="Parallel threshold simulation manager")
    parser.add_argument("--grid", required=True, help="Path to threshold_grid.csv")
    parser.add_argument("--machine", required=True, choices=["stay", "yoga", "mac"],
                        help="Which machine's rows to run")
    parser.add_argument("--binary", required=True, help="Path to compiled fput_yoshida binary")
    parser.add_argument("--outdir", required=True, help="Output directory for CSV files")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Number of parallel workers (default: CPU count - 1)")
    parser.add_argument("--limit", type=int, default=None,
                        help="(Test only) limit to first N rows after filtering")
    args = parser.parse_args()

    logdir = os.path.join(args.outdir, "logs")
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    if not os.path.isfile(args.binary):
        print(f"ERROR: binary not found: {args.binary}", file=sys.stderr)
        sys.exit(1)

    grid = pd.read_csv(args.grid)
    grid.columns = grid.columns.str.strip()
    filtered = grid[grid["Machine"] == args.machine].reset_index(drop=True)

    if args.limit is not None:
        filtered = filtered.iloc[: args.limit]

    total = len(filtered)
    if total == 0:
        print(f"No rows for machine '{args.machine}' in {args.grid}")
        sys.exit(0)

    print(f"Machine: {args.machine} | Tasks: {total} | Workers: {args.workers}")

    task_args = [
        (
            args.binary,
            int(row["N"]),
            str(row["Model"]),
            float(row["Value"]),
            float(row["Amplitude"]),
            args.outdir,
            logdir,
        )
        for _, row in filtered.iterrows()
    ]

    done_count = 0
    failed = []
    skipped_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_task, t): t for t in task_args}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            status, name = future.result()
            if status == "skipped":
                skipped_count += 1
                print(f"[{i}/{total}] skipped (exists): {name}")
            elif status == "done":
                done_count += 1
                print(f"[{i}/{total}] done: {name}")
            else:
                failed.append(name)
                print(f"[{i}/{total}] FAILED: {name}")

    print(f"\nSummary: {done_count} succeeded, {skipped_count} skipped, {len(failed)} failed")
    if failed:
        print("Failed tasks:")
        for name in failed:
            print(f"  {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
