# SOMA Fitbit Integration — Proto-Self Layer

Maps Fitbit Inspire 3 physiological data into SOMA's **L1 Proto-Self** layer
(Damasio's continuous pre-conscious body-state representation).

```
Fitbit API
    │
    ▼
fitbit_client.py          ← API wrapper, auto token refresh
    │
    ▼
soma_fitbit_ingestor.py   ← builds narrative, embeds, stores in LanceDB
    │
    ▼
~/soma/soma.db / proto_self_fitbit table
    │
    ▼
soma_daily_context.py     ← LLM-ready context generator / cron script
```

---

## Setup (one time)

```bash
# 1. Install dependencies
pip install requests requests-oauthlib lancedb sentence-transformers

# 2. Register a Personal app at https://dev.fitbit.com/apps/new
#    OAuth 2.0 Application Type: Personal
#    Callback URL: http://localhost:8080

# 3. Create config file
python fitbit_auth.py
# → Creates ~/.fitbit_config.json template
# → Fill in CLIENT_ID and CLIENT_SECRET from dev.fitbit.com

# 4. Authenticate (opens browser)
python fitbit_auth.py
# → Saves tokens to ~/.fitbit_tokens.json
```

---

## Daily usage

```bash
# Ingest today
python soma_fitbit_ingestor.py

# Backfill last 90 days
python soma_fitbit_ingestor.py --start 2026-01-14 --end 2026-04-14

# Generate 7-day LLM context
python soma_daily_context.py --context

# Semantic search
python soma_fitbit_ingestor.py --query "poor sleep high stress"
```

---

## Cron (daily 8am sync)

```bash
# Add to crontab: crontab -e
0 8 * * * cd ~/soma_fitbit && python soma_daily_context.py --ingest-first >> ~/soma/fitbit_log.txt 2>&1
```

---

## Data collected (per day)

| Signal | Source | Damasio Layer |
|--------|---------|---------------|
| Resting HR | Fitbit | L1 Proto-Self |
| HRV (RMSSD) | Fitbit sleep | L1 Proto-Self |
| Sleep stages (deep/REM/light) | Fitbit | L1 Proto-Self |
| Sleep efficiency | Fitbit | L1 Proto-Self |
| Active Zone Minutes | Fitbit | L1 Proto-Self |
| Steps | Fitbit | L1 Proto-Self |
| SpO2 (nightly avg) | Fitbit | L1 Proto-Self |
| Recovery score (heuristic) | Computed | L1 → L2 bridge |

---

## LanceDB schema (`proto_self_fitbit` table)

```
date               TEXT    PK
resting_hr         INT
hrv_rmssd          FLOAT
hrv_coverage       FLOAT
sleep_duration_min INT
sleep_efficiency   INT
sleep_start        TEXT
sleep_end          TEXT
deep_sleep_min     INT
light_sleep_min    INT
rem_sleep_min      INT
wake_min           INT
steps              INT
calories           INT
active_zone_minutes INT
very_active_min    INT
spo2_avg           FLOAT
spo2_min           FLOAT
recovery_score     FLOAT   # computed 0-10
damasio_layer      TEXT    # "L1_proto_self"
narrative          TEXT    # NL description (embedded)
vector             FLOAT[] # 384-dim sentence embedding
ingested_at        TEXT
```

---

## Future extensions

- **Polar H10 integration** — raw RR intervals via `bleak` BLE, finer HRV resolution
- **CPAP data** — AHI, leak rate via OSCAR export → SQLite → same DB
- **L2 bridge** — correlate physiological state with cognitive load events
  (calendar density, deep work blocks, git commit frequency)
- **Tonglen coherence** — tag meditation sessions, correlate HRV before/after
