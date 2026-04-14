"""
fitbit_client.py — Fitbit API client with automatic token refresh.

Wraps all the API endpoints SOMA needs for Proto-Self physiological signals.
Tokens are loaded from ~/.fitbit_tokens.json and refreshed transparently.

Usage:
  from fitbit_client import FitbitClient
  client = FitbitClient()
  daily = client.get_daily_summary("2026-04-14")
"""

import json
import os
from datetime import date, datetime, timedelta
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth

CONFIG_FILE = os.path.expanduser("~/.fitbit_config.json")
TOKEN_FILE = os.path.expanduser("~/.fitbit_tokens.json")
BASE_URL = "https://api.fitbit.com"
TOKEN_URL = "https://api.fitbit.com/oauth2/token"


class FitbitClient:
    def __init__(self):
        self.config = self._load_json(CONFIG_FILE, "Config not found. Run fitbit_auth.py first.")
        self.token = self._load_json(TOKEN_FILE, "Tokens not found. Run fitbit_auth.py first.")

    # ── Auth ──────────────────────────────────────────────────────────────────

    def _load_json(self, path, err_msg):
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ {err_msg}\n   Expected: {path}")
        with open(path) as f:
            return json.load(f)

    def _save_token(self):
        with open(TOKEN_FILE, "w") as f:
            json.dump(self.token, f, indent=2)

    def _refresh_token(self):
        resp = requests.post(
            TOKEN_URL,
            auth=HTTPBasicAuth(self.config["client_id"], self.config["client_secret"]),
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.token["refresh_token"],
            },
        )
        resp.raise_for_status()
        self.token = resp.json()
        self._save_token()
        print("🔄 Fitbit token refreshed.")

    def _get(self, path: str, retry: bool = True) -> dict:
        headers = {"Authorization": f"Bearer {self.token['access_token']}"}
        resp = requests.get(f"{BASE_URL}{path}", headers=headers)

        if resp.status_code == 401 and retry:
            self._refresh_token()
            return self._get(path, retry=False)

        resp.raise_for_status()
        return resp.json()

    # ── Date helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(d) -> str:
        if isinstance(d, (date, datetime)):
            return d.strftime("%Y-%m-%d")
        return str(d)

    # ── Heart Rate ────────────────────────────────────────────────────────────

    def get_heart_rate_summary(self, target_date="today") -> dict:
        """Daily resting HR + zone summaries."""
        d = self._fmt(target_date)
        data = self._get(f"/1/user/-/activities/heart/date/{d}/1d.json")
        summary = data.get("activities-heart", [{}])[0].get("value", {})
        return {
            "resting_hr": summary.get("restingHeartRate"),
            "zones": summary.get("heartRateZones", []),
        }

    def get_heart_rate_intraday(self, target_date="today", detail_level="1min") -> list:
        """Minute-by-minute heart rate (requires Personal app scope)."""
        d = self._fmt(target_date)
        data = self._get(
            f"/1/user/-/activities/heart/date/{d}/1d/{detail_level}.json"
        )
        return data.get("activities-heart-intraday", {}).get("dataset", [])

    # ── HRV ───────────────────────────────────────────────────────────────────

    def get_hrv(self, target_date="today") -> dict:
        """Daily HRV summary (RMSSD, from overnight sleep)."""
        d = self._fmt(target_date)
        data = self._get(f"/1/user/-/hrv/date/{d}.json")
        hrv_list = data.get("hrv", [])
        if not hrv_list:
            return {}
        val = hrv_list[0].get("value", {})
        return {
            "rmssd": val.get("dailyRmssd") or val.get("rmssd"),
            "deep_rmssd": val.get("deepRmssd"),
            "coverage": val.get("coverage"),
            "hf": val.get("hf"),
            "lf": val.get("lf"),
        }

    # ── Sleep ─────────────────────────────────────────────────────────────────

    def get_sleep(self, target_date="today") -> dict:
        """Sleep stages, efficiency, duration."""
        d = self._fmt(target_date)
        data = self._get(f"/1.2/user/-/sleep/date/{d}.json")
        sleeps = data.get("sleep", [])

        # Use the "main" sleep record if multiple exist
        main = next((s for s in sleeps if s.get("isMainSleep")), sleeps[0] if sleeps else {})
        if not main:
            return {}

        summary = main.get("levels", {}).get("summary", {})
        stages = {
            "deep_min": summary.get("deep", {}).get("minutes", 0),
            "light_min": summary.get("light", {}).get("minutes", 0),
            "rem_min": summary.get("rem", {}).get("minutes", 0),
            "wake_min": summary.get("wake", {}).get("minutes", 0),
        }
        return {
            "duration_min": main.get("minutesAsleep", 0),
            "efficiency": main.get("efficiency"),
            "start_time": main.get("startTime"),
            "end_time": main.get("endTime"),
            "stages": stages,
        }

    # ── SpO2 ──────────────────────────────────────────────────────────────────

    def get_spo2(self, target_date="today") -> dict:
        """Blood oxygen saturation (nightly average)."""
        d = self._fmt(target_date)
        data = self._get(f"/1/user/-/spo2/date/{d}.json")
        val = data.get("value", {})
        return {
            "avg": val.get("avg"),
            "min": val.get("min"),
            "max": val.get("max"),
        }

    # ── Activity ──────────────────────────────────────────────────────────────

    def get_activity_summary(self, target_date="today") -> dict:
        """Steps, calories, active zone minutes, floors."""
        d = self._fmt(target_date)
        data = self._get(f"/1/user/-/activities/date/{d}.json")
        summary = data.get("summary", {})
        azm = summary.get("activeZoneMinutes", {})
        return {
            "steps": summary.get("steps", 0),
            "calories": summary.get("caloriesOut", 0),
            "active_zone_minutes": azm.get("totalMinutes", 0),
            "fairly_active_min": summary.get("fairlyActiveMinutes", 0),
            "very_active_min": summary.get("veryActiveMinutes", 0),
            "floors": summary.get("floors", 0),
        }

    # ── Composite daily pull ───────────────────────────────────────────────────

    def get_daily_summary(self, target_date="today") -> dict:
        """
        Pull all Proto-Self signals for a single day.
        Returns a flat dict ready for SOMA ingestion.
        """
        d = self._fmt(target_date) if target_date != "today" else date.today().strftime("%Y-%m-%d")

        print(f"  📡 Pulling Fitbit data for {d}...")

        hr = self.get_heart_rate_summary(d)
        hrv = self.get_hrv(d)
        sleep = self.get_sleep(d)
        activity = self.get_activity_summary(d)

        # SpO2 can fail if device doesn't support it — handle gracefully
        try:
            spo2 = self.get_spo2(d)
        except Exception:
            spo2 = {}

        stages = sleep.get("stages", {})

        return {
            "date": d,
            # Cardiac
            "resting_hr": hr.get("resting_hr"),
            "hrv_rmssd": hrv.get("rmssd"),
            "hrv_coverage": hrv.get("coverage"),
            # Sleep
            "sleep_duration_min": sleep.get("duration_min", 0),
            "sleep_efficiency": sleep.get("efficiency"),
            "sleep_start": sleep.get("start_time"),
            "sleep_end": sleep.get("end_time"),
            "deep_sleep_min": stages.get("deep_min", 0),
            "light_sleep_min": stages.get("light_min", 0),
            "rem_sleep_min": stages.get("rem_min", 0),
            "wake_min": stages.get("wake_min", 0),
            # Activity
            "steps": activity.get("steps", 0),
            "calories": activity.get("calories", 0),
            "active_zone_minutes": activity.get("active_zone_minutes", 0),
            "very_active_min": activity.get("very_active_min", 0),
            # Oxygenation
            "spo2_avg": spo2.get("avg"),
            "spo2_min": spo2.get("min"),
        }

    def get_date_range(self, start: str, end: str = None) -> list[dict]:
        """
        Pull daily summaries for a date range.
        start/end: 'YYYY-MM-DD'. end defaults to today.
        """
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.today()

        results = []
        current = start_dt
        while current <= end_dt:
            try:
                summary = self.get_daily_summary(current.strftime("%Y-%m-%d"))
                results.append(summary)
            except Exception as e:
                print(f"  ⚠️  Skipping {current.strftime('%Y-%m-%d')}: {e}")
            current += timedelta(days=1)

        return results


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = FitbitClient()
    summary = client.get_daily_summary("today")
    print("\n📊 Today's Fitbit snapshot:")
    for k, v in summary.items():
        print(f"  {k:<25} {v}")
