"""Data analysis engine for the CityBike platform."""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from numerical import detect_outliers_zscore, trip_duration_stats, calculate_fares
from pricing import CasualPricing, MemberPricing

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class BikeShareSystem:
    """Central analysis class — loads, cleans, and analyzes bike-share data."""

    def __init__(self) -> None:
        self.trips: pd.DataFrame | None = None
        self.stations: pd.DataFrame | None = None
        self.maintenance: pd.DataFrame | None = None

    def load_data(self) -> None:
        self.trips = pd.read_csv(DATA_DIR / "trips.csv")
        self.stations = pd.read_csv(DATA_DIR / "stations.csv")
        self.maintenance = pd.read_csv(DATA_DIR / "maintenance.csv")

    def inspect_data(self) -> None:
        for name, df in [("Trips", self.trips), ("Stations", self.stations), ("Maintenance", self.maintenance)]:
            print(f"\n{'='*40}\n  {name}\n{'='*40}")
            print(df.info())
            print(f"\nMissing values:\n{df.isnull().sum()}")
            print(f"\nFirst 3 rows:\n{df.head(3)}")

    def clean_data(self) -> None:
        """Clean all DataFrames and export to *_clean.csv."""
        if self.trips is None or self.stations is None or self.maintenance is None:
            raise RuntimeError("Call load_data() first")

        trips = self.trips.copy()
        stations = self.stations.copy()
        maint = self.maintenance.copy()

        # --- Trips ---
        trips = trips.drop_duplicates(subset=["trip_id"])

        trips["start_time"] = pd.to_datetime(trips["start_time"], errors="coerce")
        trips["end_time"] = pd.to_datetime(trips["end_time"], errors="coerce")

        trips["duration_minutes"] = pd.to_numeric(trips["duration_minutes"], errors="coerce")
        trips["distance_km"] = pd.to_numeric(trips["distance_km"], errors="coerce")

        trips["user_type"] = trips["user_type"].astype(str).str.lower().str.strip()
        trips["bike_type"] = trips["bike_type"].astype(str).str.lower().str.strip()
        trips["status"] = trips["status"].astype(str).str.lower().str.strip()

        trips.loc[trips["status"].isin(["nan", "none", ""]), "status"] = np.nan
        mode_status = trips["status"].mode(dropna=True)
        fill_status = mode_status.iloc[0] if len(mode_status) else "completed"
        trips["status"] = trips["status"].fillna(fill_status)

        trips["duration_minutes"] = trips["duration_minutes"].fillna(trips["duration_minutes"].median())
        trips["distance_km"] = trips["distance_km"].fillna(trips["distance_km"].median())

        trips = trips.dropna(subset=["start_time", "end_time"])
        trips = trips[trips["end_time"] >= trips["start_time"]]
        trips = trips[trips["duration_minutes"] >= 0]
        trips = trips[trips["distance_km"] >= 0]

        trips = trips[trips["user_type"].isin(["casual", "member"])]
        trips = trips[trips["bike_type"].isin(["classic", "electric"])]
        trips = trips[trips["status"].isin(["completed", "cancelled"])]

        trips["duration_minutes"] = (trips["end_time"] - trips["start_time"]).dt.total_seconds() / 60.0
        trips["duration_minutes"] = trips["duration_minutes"].clip(lower=0)

        # --- Stations ---
        stations = stations.drop_duplicates(subset=["station_id"])
        stations["capacity"] = pd.to_numeric(stations["capacity"], errors="coerce").astype("Int64")
        stations["latitude"] = pd.to_numeric(stations["latitude"], errors="coerce")
        stations["longitude"] = pd.to_numeric(stations["longitude"], errors="coerce")
        stations = stations.dropna(subset=["station_id", "station_name", "capacity", "latitude", "longitude"])
        stations = stations[stations["capacity"] > 0]
        stations = stations[(stations["latitude"].between(-90, 90)) & (stations["longitude"].between(-180, 180))]

        # --- Maintenance ---
        maint = maint.drop_duplicates(subset=["record_id"])
        maint["date"] = pd.to_datetime(maint["date"], errors="coerce")
        maint["cost"] = pd.to_numeric(maint["cost"], errors="coerce")
        maint["bike_type"] = maint["bike_type"].astype(str).str.lower().str.strip()
        maint["maintenance_type"] = maint["maintenance_type"].astype(str).str.lower().str.strip()
        maint = maint.dropna(subset=["record_id", "bike_id", "bike_type", "date", "maintenance_type"])
        maint = maint[maint["bike_type"].isin(["classic", "electric"])]

        overall_med = maint["cost"].median()
        maint["cost"] = maint.groupby("bike_type")["cost"].transform(lambda s: s.fillna(s.median()))
        maint["cost"] = maint["cost"].fillna(overall_med)

        self.trips, self.stations, self.maintenance = trips, stations, maint

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        trips.to_csv(DATA_DIR / "trips_clean.csv", index=False)
        stations.to_csv(DATA_DIR / "stations_clean.csv", index=False)
        maint.to_csv(DATA_DIR / "maintenance_clean.csv", index=False)

    # --------------------- Analytics (Q1–Q14) ---------------------

    def total_trips_summary(self) -> dict[str, float]:
        df = self._trips()
        return {
            "total_trips": int(len(df)),
            "total_distance_km": float(round(df["distance_km"].sum(), 2)),
            "avg_duration_min": float(round(df["duration_minutes"].mean(), 2)),
        }

    def top_start_stations(self, n: int = 10) -> pd.DataFrame:
        df = self._trips()
        st = self._stations()
        counts = df["start_station_id"].value_counts().head(n).rename_axis("station_id").reset_index(name="trip_count")
        return counts.merge(st[["station_id", "station_name"]], on="station_id", how="left")

    def top_end_stations(self, n: int = 10) -> pd.DataFrame:
        df = self._trips()
        st = self._stations()
        counts = df["end_station_id"].value_counts().head(n).rename_axis("station_id").reset_index(name="trip_count")
        return counts.merge(st[["station_id", "station_name"]], on="station_id", how="left")

    def peak_usage_hours(self) -> pd.Series:
        df = self._trips()
        return df["start_time"].dt.hour.value_counts().sort_index()

    def busiest_day_of_week(self) -> pd.Series:
        df = self._trips()
        return df["start_time"].dt.day_name().value_counts()

    def avg_distance_by_user_type(self) -> pd.Series:
        df = self._trips()
        return df.groupby("user_type")["distance_km"].mean().round(3)
    
    #محاسبه نرخ استفاده از دوچرخه‌ها.
    def bike_utilization_rate(self) -> float:
        df = self._trips()
        total_used_minutes = df["duration_minutes"].sum()
        num_bikes = df["bike_id"].nunique()
        window_minutes = (df["end_time"].max() - df["start_time"].min()).total_seconds() / 60.0
        if num_bikes == 0 or window_minutes <= 0:
            return 0.0
        return float(total_used_minutes / (num_bikes * window_minutes))

    def monthly_trip_trend(self) -> pd.Series:
        df = self._trips()
        return df.groupby(df["start_time"].dt.to_period("M"))["trip_id"].count().sort_index()

    def top_active_users(self, n: int = 15) -> pd.DataFrame:
        df = self._trips()
        out = df.groupby(["user_id", "user_type"])["trip_id"].count().reset_index(name="trip_count")
        return out.sort_values("trip_count", ascending=False).head(n)

    def maintenance_cost_by_bike_type(self) -> pd.Series:
        m = self._maint()
        return m.groupby("bike_type")["cost"].sum().round(2)

    def top_routes(self, n: int = 10) -> pd.DataFrame:
        df = self._trips()
        st = self._stations()
        routes = df.groupby(["start_station_id", "end_station_id"])["trip_id"].count().reset_index(name="trip_count")
        routes = routes.sort_values("trip_count", ascending=False).head(n)

        routes = routes.merge(
            st[["station_id", "station_name"]],
            left_on="start_station_id",
            right_on="station_id",
            how="left",
        ).drop(columns=["station_id"]).rename(columns={"station_name": "start_station_name"})

        routes = routes.merge(
            st[["station_id", "station_name"]],
            left_on="end_station_id",
            right_on="station_id",
            how="left",
        ).drop(columns=["station_id"]).rename(columns={"station_name": "end_station_name"})

        return routes

    def trip_completion_rate(self) -> pd.Series:
        df = self._trips()
        counts = df["status"].value_counts()
        return (counts / counts.sum()).round(4)

    def avg_trips_per_user_by_type(self) -> pd.Series:
        df = self._trips()
        user_counts = df.groupby(["user_type", "user_id"])["trip_id"].count()
        return user_counts.groupby("user_type").mean().round(3)

    def bikes_highest_maintenance_frequency(self, n: int = 10) -> pd.DataFrame:
        m = self._maint()
        freq = m.groupby(["bike_id", "bike_type"])["record_id"].count().reset_index(name="maintenance_count")
        return freq.sort_values("maintenance_count", ascending=False).head(n)

    def outlier_trips(self, threshold: float = 3.0) -> pd.DataFrame:
        df = self._trips()
        mask_dur = detect_outliers_zscore(df["duration_minutes"].to_numpy(), threshold=threshold)
        mask_dist = detect_outliers_zscore(df["distance_km"].to_numpy(), threshold=threshold)

        any_mask = mask_dur | mask_dist
        reasons = np.where(
            mask_dur & mask_dist,
            "duration+distance",
            np.where(mask_dur, "duration", "distance"),
        )

        out = df.loc[any_mask, ["trip_id", "duration_minutes", "distance_km", "user_type", "status"]].copy()
        out["outlier_reason"] = reasons[any_mask]
        return out.sort_values(["outlier_reason", "duration_minutes"], ascending=[True, False])
    
    #محاسبه درآمد بر اساس نوع کاربر.
    def revenue_by_user_type(self) -> pd.Series:
        df = self._trips()
        casual = CasualPricing()
        member = MemberPricing()

        rev = {}
        for utype, strat in [("casual", casual), ("member", member)]:
            sub = df[df["user_type"] == utype]
            fares = calculate_fares(
                durations=sub["duration_minutes"].to_numpy(),
                distances=sub["distance_km"].to_numpy(),
                per_minute=strat.PER_MINUTE,
                per_km=strat.PER_KM,
                unlock_fee=getattr(strat, "UNLOCK_FEE", 0.0),
            )
            rev[utype] = float(np.sum(fares))
        return pd.Series(rev).round(2)

    # --------------------- Reporting ---------------------
    ###
    def export_outputs(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.top_start_stations().to_csv(OUTPUT_DIR / "top_stations.csv", index=False)
        self.top_active_users().to_csv(OUTPUT_DIR / "top_users.csv", index=False)
        ms = self.maintenance_cost_by_bike_type().reset_index().rename(columns={"cost": "total_cost"})
        ms.to_csv(OUTPUT_DIR / "maintenance_summary.csv", index=False)
     
     ###
    def generate_summary_report(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = OUTPUT_DIR / "summary_report.txt"

        summary = self.total_trips_summary()
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("CityBike — Summary Report")
        lines.append("=" * 70)

        lines.append("\n--- Q1: Overall summary ---")
        lines.append(f"Total trips: {summary['total_trips']}")
        lines.append(f"Total distance: {summary['total_distance_km']} km")
        lines.append(f"Average duration: {summary['avg_duration_min']} minutes")

        lines.append("\n--- Q2: Top 10 start stations ---")
        lines.append(self.top_start_stations(10).to_string(index=False))

        lines.append("\n--- Q2b: Top 10 end stations ---")
        lines.append(self.top_end_stations(10).to_string(index=False))

        lines.append("\n--- Q3: Peak usage hours ---")
        lines.append(self.peak_usage_hours().to_string())

        lines.append("\n--- Q4: Busiest day of week ---")
        lines.append(self.busiest_day_of_week().to_string())

        lines.append("\n--- Q5: Avg distance by user type ---")
        lines.append(self.avg_distance_by_user_type().to_string())

        lines.append("\n--- Q6: Bike utilization rate (approx) ---")
        lines.append(f"{self.bike_utilization_rate():.4f} (share of time bikes are in use)")

        lines.append("\n--- Q7: Monthly trip trend ---")
        lines.append(self.monthly_trip_trend().to_string())

        lines.append("\n--- Q8: Top 15 active users ---")
        lines.append(self.top_active_users(15).to_string(index=False))

        lines.append("\n--- Q9: Maintenance cost by bike type ---")
        lines.append(self.maintenance_cost_by_bike_type().to_string())

        lines.append("\n--- Q10: Top 10 routes ---")
        lines.append(self.top_routes(10).to_string(index=False))

        lines.append("\n--- Q11: Trip completion rate ---")
        lines.append(self.trip_completion_rate().to_string())

        lines.append("\n--- Q12: Avg trips per user by type ---")
        lines.append(self.avg_trips_per_user_by_type().to_string())

        lines.append("\n--- Q13: Bikes with highest maintenance frequency ---")
        lines.append(self.bikes_highest_maintenance_frequency(10).to_string(index=False))

        lines.append("\n--- Q14: Outlier trips (z-score) ---")
        out = self.outlier_trips(threshold=3.0)
        lines.append(out.head(20).to_string(index=False))
        lines.append(f"Total outliers found: {len(out)}")

        lines.append("\n--- Extra: Estimated revenue by user type ---")
        lines.append(self.revenue_by_user_type().to_string())

        df = self._trips()
        stats = trip_duration_stats(df["duration_minutes"].to_numpy())
        lines.append("\n--- Extra: Duration stats (NumPy) ---")
        for k, v in stats.items():
            lines.append(f"{k}: {v:.3f}")

        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Report saved to {report_path}")

    # --------------------- Internal helpers ---------------------

    def _trips(self) -> pd.DataFrame:
        if self.trips is None:
            raise RuntimeError("Trips not loaded")
        return self.trips

    def _stations(self) -> pd.DataFrame:
        if self.stations is None:
            raise RuntimeError("Stations not loaded")
        return self.stations

    def _maint(self) -> pd.DataFrame:
        if self.maintenance is None:
            raise RuntimeError("Maintenance not loaded")
        return self.maintenance
