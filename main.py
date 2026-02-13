"""CityBike — Bike-Sharing Analytics Platform (entry point)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys  #دسترسی به اطلاعات مربوط به اجرای پایتون

from analyzer import BikeShareSystem, OUTPUT_DIR, DATA_DIR
from visualization import (
    plot_trips_per_station,
    plot_monthly_trend,
    plot_duration_histogram,
    plot_duration_by_user_type,
)
from algorithms import benchmark_sort, benchmark_search, merge_sort


def ensure_input_csvs() -> str | None:
    
    #Ensure raw input CSVs exist.If missing, run generate_data.py 
   
    required = ["stations.csv", "trips.csv", "maintenance.csv"]
    missing = [f for f in required if not (DATA_DIR / f).exists()]

    if not missing:
        return None  # already exists, no generation message

    gen_script = Path(__file__).resolve().parent / "generate_data.py"
    if not gen_script.exists():
        raise FileNotFoundError(f"generate_data.py not found at: {gen_script}")

    # Run generator and capture its stdout/stderr
    result = subprocess.run(
        [sys.executable, str(gen_script)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent),
    )

    # If generator failed, raise an error with its message
    if result.returncode != 0:
        raise RuntimeError(
            "generate_data.py failed!\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}\n"
        )

    # Return what it printed (so we can print it *during* main.py output)
    return (result.stdout or "").strip() or None


def main() -> None:
    # ✅ Generate input CSVs only when running main.py (and show message now)
    gen_msg = ensure_input_csvs()
    if gen_msg:
        print(gen_msg)

    system = BikeShareSystem()

    print(">>> Loading data ...")
    system.load_data()

    print(">>> Cleaning data ...")
    system.clean_data()

    print(">>> Running analytics ...")
    summary = system.total_trips_summary()
    print(summary)

    # export analytics outputs
    system.export_outputs()
    system.generate_summary_report()

    # --- Algorithms demo/benchmarks ---
    durations = system.trips["duration_minutes"].dropna().astype(float).tolist()
    if len(durations) < 5:
        print("Not enough duration data for benchmarks.")
    else:
        sort_bench = benchmark_sort(durations, key=lambda x: x, repeats=5)
        sorted_durations = merge_sort(durations, key=lambda x: x)

        target = sorted_durations[len(sorted_durations) // 2]
        search_bench = benchmark_search(sorted_durations, target=target, key=lambda x: x, repeats=5)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        bench_path = Path(OUTPUT_DIR) / "search_sort_benchmarks.txt"
        lines: list[str] = []
        lines.append("=== Sorting benchmarks (ms) ===")
        for k, v in sort_bench.items():
            lines.append(f"{k}: {v}")
        lines.append("\n=== Searching benchmarks (ms) ===")
        for k, v in search_bench.items():
            lines.append(f"{k}: {v}")

        bench_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved: {bench_path}")

    # --- Visualizations ---
    print(">>> Generating visualizations ...")
    plot_trips_per_station(system.trips, system.stations)
    plot_monthly_trend(system.trips)
    plot_duration_histogram(system.trips)
    plot_duration_by_user_type(system.trips)

    print(">>> Done. Check output/ and data/ for generated files.")


if __name__ == "__main__":
    main()
