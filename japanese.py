"""
Japanese Journey – spreadsheet parser and statistics toolkit.

Usage:
    from japanese_journey import JourneyLog

    log = JourneyLog("Japanese Journey.xlsx")
    print(log.cumulative_total())
    print(log.cumulative_by_item())
    print(log.average_per_day(window=7))
    print(log.daily_totals())
    log.summary()
"""

import re
import pandas as pd
from datetime import datetime
from typing import Optional

MONTHS = {m: i + 1 for i, m in enumerate([
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
])}
YEARS = [2026, 2027]


def _is_month_sheet(name: str) -> bool:
    """Return True if the sheet name matches '<Month> <Year>'."""
    parts = name.strip().split()
    return (
        len(parts) == 2
        and parts[0].capitalize() in MONTHS
        and parts[1].isdigit()
        and int(parts[1]) in YEARS
    )


def _to_seconds(hr, mn, sec) -> int:
    """Convert h/m/s to total seconds, treating NaN as 0."""
    def _int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0
    return _int(hr) * 3600 + _int(mn) * 60 + _int(sec)


def _fmt(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    seconds = int(round(seconds))
    h, rem = divmod(abs(seconds), 3600)
    m, s = divmod(rem, 60)
    sign = "-" if seconds < 0 else ""
    return f"{sign}{h}:{m:02d}:{s:02d}"


def _parse_sheet(df: pd.DataFrame, month: str, year: int) -> pd.DataFrame:
    """
    Extract every activity row from a monthly sheet.

    Returns a DataFrame with columns:
        date, day_num, item, seconds
    """
    # ── Locate the "Daily Log" label row first, then find the header below it.
    # This skips the top "Time Adder" block which also has a row labelled Total.
    daily_log_row = None
    for i, row in df.iterrows():
        if str(row.iloc[0]).strip() == "Daily Log":
            daily_log_row = i
            break

    search_start = daily_log_row if daily_log_row is not None else 0

    # ── Locate the header row: col[1]=='Day 1', col[2]=='hr', col[3]=='min'
    header_row = None
    for i, row in df.iloc[search_start:].iterrows():
        if (
            str(row.iloc[1]) == "Day 1"
            and str(row.iloc[2]).lower() == "hr"
            and str(row.iloc[3]).lower() == "min"
        ):
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"Cannot locate header in {month} {year}")

    # ── Find delta: distance to the first "Total" row after header_row
    delta = None
    for i in range(header_row + 1, len(df)):
        if str(df.iloc[i, 1]) == "Total":
            delta = i - header_row
            break

    if delta is None:
        raise ValueError(f"Cannot find block delta in {month} {year}")

    # ── Walk every block
    records = []
    row_idx = header_row
    while row_idx + delta <= len(df):
        block_header = df.iloc[row_idx]
        day_label = str(block_header.iloc[1])   # e.g. "Day 1"
        current_date = block_header.iloc[0]      # datetime or NaT

        if not re.match(r"Day \d+", day_label):
            break  # no more day blocks

        day_num = int(day_label.split()[1])

        # Normalise date
        if isinstance(current_date, (datetime, pd.Timestamp)):
            current_date = current_date.date()
        else:
            current_date = None  # will be None for blocks past the first when date missing

        # Activity rows: row_idx+1 .. row_idx+delta-1  (last row is Total)
        for offset in range(1, delta):
            act_row = df.iloc[row_idx + offset]
            item = act_row.iloc[1]
            hr   = act_row.iloc[2]
            mn   = act_row.iloc[3]
            sec  = act_row.iloc[4]

            if pd.isna(item) or str(item) == "Total":
                continue

            secs = _to_seconds(hr, mn, sec)
            if secs == 0:
                continue  # skip empty entries

            records.append({
                "month":   month,
                "year":    year,
                "date":    current_date,
                "day_num": day_num,
                "item":    str(item).strip(),
                "seconds": secs,
            })

        row_idx += delta + 1  # skip past the Total row to the next Day header

    return pd.DataFrame(records)


class JourneyLog:
    """
    Parses all '<Month> <Year>' sheets and exposes statistics.

    Parameters
    ----------
    path : str
        Path to the Excel file.
    """

    def __init__(self, path: str):
        self.path = path
        raw = pd.read_excel(path, header=None, sheet_name=None)

        frames = []
        for sheet_name, df in raw.items():
            if not _is_month_sheet(sheet_name):
                continue
            month_str, year_str = sheet_name.strip().split()
            month = month_str.capitalize()
            year  = int(year_str)
            try:
                frame = _parse_sheet(df, month, year)
                frames.append(frame)
            except ValueError as e:
                print(f"[warning] Skipping sheet '{sheet_name}': {e}")

        if not frames:
            raise RuntimeError("No valid month sheets found.")

        self.df = pd.concat(frames, ignore_index=True)
        self.df.sort_values(["year", "month", "day_num", "item"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _filter(
        self,
        item: Optional[str] = None,
        month: Optional[str] = None,
        year: Optional[int] = None,
    ) -> pd.DataFrame:
        d = self.df
        if item:
            d = d[d["item"].str.lower() == item.lower()]
        if month:
            d = d[d["month"].str.lower() == month.lower()]
        if year:
            d = d[d["year"] == year]
        return d

    # ── public API ───────────────────────────────────────────────────────────

    def cumulative_total(
        self,
        month: Optional[str] = None,
        year: Optional[int] = None,
    ) -> str:
        """Total study time across all items (optionally filtered by month/year)."""
        secs = self._filter(month=month, year=year)["seconds"].sum()
        return _fmt(secs)

    def cumulative_by_item(
        self,
        month: Optional[str] = None,
        year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Time spent per activity item, sorted descending.
        Returns a DataFrame with columns: item, total_time, hours.
        """
        d = self._filter(month=month, year=year)
        g = d.groupby("item")["seconds"].sum().reset_index()
        g["total_time"] = g["seconds"].apply(_fmt)
        g["hours"] = (g["seconds"] / 3600).round(3)
        g.sort_values("seconds", ascending=False, inplace=True)
        g.reset_index(drop=True, inplace=True)
        return g[["item", "total_time", "hours"]]

    def daily_totals(
        self,
        month: Optional[str] = None,
        year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Total seconds per calendar day.
        Returns a DataFrame with columns: year, month, day_num, date, total_time, hours.
        """
        d = self._filter(month=month, year=year)
        g = d.groupby(["year", "month", "day_num", "date"])["seconds"].sum().reset_index()
        g["total_time"] = g["seconds"].apply(_fmt)
        g["hours"] = (g["seconds"] / 3600).round(3)
        return g

    def average_per_day(
        self,
        window: Optional[int] = None,
        item: Optional[str] = None,
        month: Optional[str] = None,
        year: Optional[int] = None,
    ) -> str:
        """
        Average daily study time (optionally over the last `window` days).

        Parameters
        ----------
        window : int, optional
            Number of most-recent days to consider. None = all days.
        item : str, optional
            Restrict to a specific activity.
        month / year : optional
            Further restrict by month or year.
        """
        d = self._filter(item=item, month=month, year=year)
        daily = d.groupby(["year", "month", "day_num"])["seconds"].sum().reset_index()
        daily.sort_values(["year", "day_num"], inplace=True)

        if window:
            daily = daily.tail(window)

        if daily.empty:
            return "0:00:00"

        avg_secs = daily["seconds"].mean()
        return _fmt(avg_secs)

    def item_names(self) -> list[str]:
        """List all unique activity names."""
        return sorted(self.df["item"].unique().tolist())

    def summary(self) -> None:
        """Print a human-readable summary to stdout."""
        days = self.df.groupby(["year", "month", "day_num"]).ngroups
        total = self.cumulative_total()
        avg_all = self.average_per_day()
        avg_7   = self.average_per_day(window=7)
        avg_14  = self.average_per_day(window=14)

        print("=" * 52)
        print("  Japanese Journey – Study Log Summary")
        print("=" * 52)
        print(f"  Sheets processed : {self.df['month'].unique().tolist()}")
        print(f"  Days logged      : {days}")
        print(f"  Total study time : {total}")
        print(f"  Avg / day (all)  : {avg_all}")
        print(f"  Avg / day (7d)   : {avg_7}")
        print(f"  Avg / day (14d)  : {avg_14}")
        print()
        print("  Time by activity:")
        by_item = self.cumulative_by_item()
        for _, row in by_item.iterrows():
            print(f"    {row['item']:<25} {row['total_time']:>10}  ({row['hours']:.2f} hrs)")
        print("=" * 52)


# ── CLI demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "~/Downloads/Japanese Journey.xlsx"
    log = JourneyLog(path)
    log.summary()

    print()
    print("--- Per-item cumulative (April 2026 only) ---")
    print(log.cumulative_by_item(month="April", year=2026).to_string(index=False))

    print()
    print("--- Daily totals ---")
    print(log.daily_totals().to_string(index=False))

    print()
    print("--- Average per day (last 7 days, all items) ---")
    print(log.average_per_day(window=7))

    print()
    print("--- Average per day for WaniKani only ---")
    print(log.average_per_day(item="WaniKani"))
