# SOMA CPAP Integration — Proto-Self Layer

Maps ResMed CPAP data (myAir + SD card EDF) into SOMA's Proto-Self layer
for sleep-disordered breathing analysis.

## Two data paths

### 1. myAir (automated, daily summary)
```
myAir API (Okta OAuth2)
    │
    ▼
myair_client.py         ← auth, GraphQL fetch
    │
    ▼
cpap_ingestor.py        ← normalize, embed, store in LanceDB
    │
    ▼
~/soma/soma.db / proto_self_cpap_daily table
```

Per night: AHI, usage hours, mask seal %, leak percentile, myAir score.

### 2. SD card EDF (manual, time-resolved)
```
ResMed SD card DATALOG/YYYYMMDD/
    │
    ▼
edf_parser.py           ← parse *_EVE.edf (events), *_PLD.edf (pressure/leak), *_BRP.edf (flow)
    │
    ▼
cpap_ingestor.py        ← merge with myAir record, store events table
    │
    ▼
proto_self_cpap_events  ← timestamped apnea/hypopnea events
```

Per event: timestamp, type, duration. Enables: "3am apnea cluster
preceded next-day HRV drop by 8 hours."

---

## Setup (one time)

### myAir (daily poll)
```bash
# 1. Create config
cat > ~/.resmed_config.json << 'EOF'
{
  "email": "your@myair_email.com",
  "password": "your_myair_password",
  "country": "US"
}
EOF

# 2. Run first auth
python -m soma.proto_self.cpap.myair_client
# → Saves tokens to ~/.resmed_tokens.json
```

### SD card (richer data, weekly)
```bash
# 1. Insert CPAP SD card into your Mac
# 2. Locate DATALOG directory (usually at SD card root)
# 3. Parse last 7 nights:
python -m soma.proto_self.cpap.edf_parser /Volumes/SD_CARD/DATALOG --last-n 7
```

---

## Daily usage

```bash
# Cron — runs at 6:30am (after Fitbit at 6:00am)
30 6 * * * cd ~/git/SOMA/SOMA_Foundation && .venv/bin/python -m soma.proto_self.cpap.daily_sync

# Manual
python -m soma.proto_self.cpap.daily_sync              # myAir, last 7 days
python -m soma.proto_self.cpap.daily_sync --days 30     # backfill 30 days
python -m soma.proto_self.cpap.daily_sync --edf ~/SD/DATALOG --days 7  # SD card
```

---

## What SOMA does with CPAP data

1. **State classifier integration** — high-AHI nights can shift state from
   `baseline` → `recovering`. Under-compliance flagged in reason text.

2. **Dashboard panel** — current AHI, usage, myAir score, 30-day compliance,
   14-day AHI chart, correlation with Fitbit recovery.

3. **Correlation analysis** — Pearson correlations between AHI and next-day:
   - Recovery score
   - HRV (Fitbit overnight)
   - Resting HR
   - SpO2

4. **Insight generation** — dashboard surfaces findings like:
   - "Higher AHI correlates with lower next-day HRV — autonomic cost is measurable"
   - "Higher AHI nights are followed by lower recovery scores"
   - "Higher AHI correlates with lower overnight SpO2 — hypoxic burden"

5. **Future (Week 6+):** feed CPAP events into hippocampus for episodic
   memory pattern detection. "The last 3 AHI > 20 nights were followed by
   poor decisions the next afternoon."

---

## Tables

### `proto_self_cpap_daily`
| Field | Type | Notes |
|---|---|---|
| date | str | YYYY-MM-DD |
| source | str | myair \| edf \| merged |
| ahi | float | events/hour |
| usage_min | int | minutes on CPAP |
| sleep_score | int | myAir 0-100 |
| leak_p95 | float | 95th percentile leak, L/min |
| mean_pressure | float | cmH2O |
| apneas, hypopneas | int | counts |
| narrative | str | natural language description |
| vector | float[384] | sentence-transformer embedding |

### `proto_self_cpap_events` (from SD card only)
| Field | Type |
|---|---|
| date | str |
| timestamp | str (ISO) |
| event_type | apnea \| hypopnea |
| duration_sec | float |
| hour_of_night | int (0-11 from sleep start) |

---

## Security

- myAir credentials in `~/.resmed_config.json` — gitignored
- Tokens in `~/.resmed_tokens.json` — gitignored
- No data leaves your machine except the myAir API call itself
