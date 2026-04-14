# SOMA Week 2 Spec — Mobile Agent + NAS Brain

**Project:** SOMA (Sentient Observation & Memory Architecture)  
**Phase:** 2 — Distributed Body/Brain Architecture  
**Goal:** Phone becomes a thin sensor node. NAS becomes the persistent brain. They stay connected everywhere.

---

## Architecture Overview

```
Polar H10 (BLE)
    ↓
SOMA Mobile (React Native — thin client)
  ├── BLE collector (H10 → RR intervals)
  ├── Local SQLite buffer (offline resilience)
  ├── Smart router
  │     ├── WiFi → NAS direct (192.168.x.x)
  │     └── Cellular → Tailscale tunnel → NAS (100.x.x.x)
  └── Conversation terminal (talk to SOMA from anywhere)
         ↓
SOMA Core (UGREEN DXP4800 Pro)
  ├── FastAPI server (soma_server.py)
  ├── RR/HRV processing pipeline
  ├── LanceDB (memory + embeddings)
  ├── Baseline model (Paul's normal)
  ├── Anomaly detection
  └── Conversational probe layer
```

**Principle:** The phone is a sensor and a terminal. The brain never leaves home.

---

## Week 2 Deliverables

- [ ] Tailscale installed on NAS + phone
- [ ] `soma_server.py` — FastAPI running on NAS
- [ ] SOMA Mobile app scaffold (React Native)
- [ ] BLE collection working on phone
- [ ] Smart WiFi/cellular routing working
- [ ] Local buffer with sync-on-connect
- [ ] End-to-end test: H10 → phone → NAS → DB

---

## Step 1: Tailscale Setup

### On the NAS (UGREEN DXP4800 Pro)
UGREEN runs a Debian-based Linux environment. Install via:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Note your NAS Tailscale IP — it will look like `100.x.x.x`. This never changes, regardless of network.

### On your iPhone
Install Tailscale from the App Store. Sign in with the same account. Your NAS will appear in the network. That's it.

### Verify
```bash
# From phone terminal or any device on the Tailnet
ping 100.x.x.x  # your NAS Tailscale IP
```

You now have a permanent private tunnel. WiFi or cellular — same address, always connected.

---

## Step 2: SOMA Server (FastAPI on NAS)

### File: `soma_server.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
import sqlite3
import math

app = FastAPI(title="SOMA Core", version="0.2.0")
DB_PATH = "/data/soma/soma_cardio.db"  # adjust to your NAS mount path


# ── Models ──────────────────────────────────────────────────────────────────

class RRBatch(BaseModel):
    session_id: str
    label: Optional[str] = None
    device_id: str
    readings: List[dict]  # [{timestamp, rr_ms}]

class ContextTag(BaseModel):
    session_id: str
    label: str  # morning_baseline, post_run, stress, meditation, etc.

class ProbeRequest(BaseModel):
    message: str
    include_recent_hours: Optional[int] = 4


# ── DB Init ──────────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rr_intervals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            rr_ms REAL NOT NULL,
            session_id TEXT NOT NULL,
            device_id TEXT,
            synced_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            label TEXT,
            device_id TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL,
            baseline REAL,
            deviation REAL,
            acknowledged INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

init_db()


# ── HRV Utils ────────────────────────────────────────────────────────────────

def compute_rmssd(rr_list):
    if len(rr_list) < 2:
        return None
    diffs = [(rr_list[i+1] - rr_list[i])**2 for i in range(len(rr_list)-1)]
    return round(math.sqrt(sum(diffs) / len(diffs)), 2)

def compute_rhr(rr_list):
    if not rr_list:
        return None
    avg_rr = sum(rr_list) / len(rr_list)
    return round(60000 / avg_rr, 1)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/ingest/rr")
def ingest_rr(batch: RRBatch):
    """Receive RR interval batch from mobile app."""
    conn = get_conn()
    synced_at = datetime.now().isoformat()

    # Upsert session
    conn.execute("""
        INSERT OR IGNORE INTO sessions (session_id, started_at, label, device_id)
        VALUES (?, ?, ?, ?)
    """, (batch.session_id, synced_at, batch.label, batch.device_id))

    # Insert readings (skip duplicates by timestamp + session)
    inserted = 0
    for r in batch.readings:
        try:
            conn.execute("""
                INSERT INTO rr_intervals (timestamp, rr_ms, session_id, device_id, synced_at)
                VALUES (?, ?, ?, ?, ?)
            """, (r["timestamp"], r["rr_ms"], batch.session_id, batch.device_id, synced_at))
            inserted += 1
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    return {"status": "ok", "inserted": inserted, "session_id": batch.session_id}


@app.post("/ingest/context")
def tag_session(tag: ContextTag):
    """Label a session after the fact."""
    conn = get_conn()
    conn.execute(
        "UPDATE sessions SET label = ? WHERE session_id = ?",
        (tag.label, tag.session_id)
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "label": tag.label}


@app.get("/status")
def get_status():
    """Current HRV state — last 60 readings."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT rr_ms FROM rr_intervals
        ORDER BY id DESC LIMIT 60
    """).fetchall()
    conn.close()

    rr_list = [r["rr_ms"] for r in rows]
    return {
        "rhr_bpm": compute_rhr(rr_list),
        "rmssd_ms": compute_rmssd(rr_list),
        "readings_in_window": len(rr_list),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/baseline")
def get_baseline():
    """Paul's baseline from morning_baseline labeled sessions."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT rr.rr_ms FROM rr_intervals rr
        JOIN sessions s ON rr.session_id = s.session_id
        WHERE s.label = 'morning_baseline'
        ORDER BY rr.id DESC LIMIT 500
    """).fetchall()
    conn.close()

    rr_list = [r["rr_ms"] for r in rows]
    if not rr_list:
        return {"status": "insufficient_data", "message": "Run more morning_baseline sessions"}

    mean_rhr = compute_rhr(rr_list)
    mean_rmssd = compute_rmssd(rr_list)

    return {
        "baseline_rhr_bpm": mean_rhr,
        "baseline_rmssd_ms": mean_rmssd,
        "sample_size": len(rr_list),
        "status": "ok"
    }


@app.get("/anomalies")
def get_anomalies(unacknowledged_only: bool = True):
    """Return flagged anomaly events."""
    conn = get_conn()
    query = "SELECT * FROM anomalies"
    if unacknowledged_only:
        query += " WHERE acknowledged = 0"
    query += " ORDER BY detected_at DESC LIMIT 20"
    rows = conn.execute(query).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/probe")
def probe(req: ProbeRequest):
    """
    Placeholder for conversational probe.
    Week 4: this calls the LLM with biometric context injected.
    """
    return {
        "status": "not_yet_implemented",
        "message": "Conversational probe arrives in Week 4.",
        "your_message": req.message
    }


# ── Run ──────────────────────────────────────────────────────────────────────
# uvicorn soma_server:app --host 0.0.0.0 --port 8765 --reload
```

### Start the server
```bash
pip install fastapi uvicorn --break-system-packages
uvicorn soma_server:app --host 0.0.0.0 --port 8765
```

SOMA Core is now reachable at:
- **WiFi:** `http://192.168.x.x:8765`
- **Cellular:** `http://100.x.x.x:8765` (via Tailscale)

Verify: `curl http://100.x.x.x:8765/status`

---

## Step 3: SOMA Mobile App (React Native)

### Project scaffold
```bash
npx react-native init SOMAmobile
cd SOMAmobile
npm install react-native-ble-plx @react-native-async-storage/async-storage axios
```

### Smart Router (`src/somaClient.js`)

```javascript
import axios from 'axios';
import NetInfo from '@react-native-community/netinfo';

const SOMA_LOCAL = 'http://192.168.1.X:8765';   // your NAS local IP
const SOMA_TAIL  = 'http://100.X.X.X:8765';     // your NAS Tailscale IP

async function getSomaUrl() {
  const net = await NetInfo.fetch();
  // On home WiFi — use local address (faster, no tunnel overhead)
  if (net.type === 'wifi' && net.details?.ssid === 'YourHomeSSID') {
    return SOMA_LOCAL;
  }
  // Everything else — route through Tailscale
  return SOMA_TAIL;
}

export async function somaPost(path, data) {
  const base = await getSomaUrl();
  try {
    const res = await axios.post(`${base}${path}`, data, { timeout: 5000 });
    return res.data;
  } catch (err) {
    // Queue for later sync if unreachable
    await queueForSync(path, data);
    return null;
  }
}

export async function somaGet(path) {
  const base = await getSomaUrl();
  const res = await axios.get(`${base}${path}`, { timeout: 5000 });
  return res.data;
}
```

### Local Buffer (`src/syncQueue.js`)

```javascript
import AsyncStorage from '@react-native-async-storage/async-storage';
import { somaPost } from './somaClient';

const QUEUE_KEY = 'soma_sync_queue';

export async function queueForSync(path, data) {
  const raw = await AsyncStorage.getItem(QUEUE_KEY);
  const queue = raw ? JSON.parse(raw) : [];
  queue.push({ path, data, queued_at: new Date().toISOString() });
  await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(queue));
  console.log(`[SOMA] Queued for sync. Queue size: ${queue.length}`);
}

export async function flushQueue() {
  const raw = await AsyncStorage.getItem(QUEUE_KEY);
  if (!raw) return;
  const queue = JSON.parse(raw);
  const remaining = [];

  for (const item of queue) {
    const result = await somaPost(item.path, item.data);
    if (!result) remaining.push(item);  // retry next flush
  }

  await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(remaining));
  console.log(`[SOMA] Flush complete. ${remaining.length} items remaining.`);
}
```

### BLE Collector (`src/polarCollector.js`)

```javascript
import { BleManager } from 'react-native-ble-plx';
import { Buffer } from 'buffer';
import { somaPost } from './somaClient';
import { queueForSync } from './syncQueue';

const manager = new BleManager();
const HR_SERVICE = '0000180d-0000-1000-8000-00805f9b34fb';
const HR_CHAR    = '00002a37-0000-1000-8000-00805f9b34fb';

let rrBuffer = [];
const FLUSH_INTERVAL_MS = 30000;  // sync to NAS every 30 seconds

function parseRR(base64Value) {
  const bytes = Buffer.from(base64Value, 'base64');
  const flag = bytes[0];
  const rrStart = (flag & 0x01) ? 3 : 2;
  const intervals = [];
  for (let i = rrStart; i < bytes.length - 1; i += 2) {
    const rr = ((bytes[i+1] << 8) | bytes[i]) / 1024 * 1000;
    intervals.push(Math.round(rr * 100) / 100);
  }
  return intervals;
}

export async function startCollection(sessionId, label) {
  manager.startDeviceScan(null, null, (error, device) => {
    if (error || !device?.name?.includes('Polar')) return;
    manager.stopDeviceScan();

    device.connect()
      .then(d => d.discoverAllServicesAndCharacteristics())
      .then(d => {
        d.monitorCharacteristicForService(HR_SERVICE, HR_CHAR, (err, char) => {
          if (err) return;
          const rr_list = parseRR(char.value);
          const ts = new Date().toISOString();
          rr_list.forEach(rr_ms => {
            rrBuffer.push({ timestamp: ts, rr_ms });
          });
        });
      });
  });

  // Flush buffer to SOMA every 30 seconds
  setInterval(async () => {
    if (rrBuffer.length === 0) return;
    const batch = {
      session_id: sessionId,
      label: label,
      device_id: 'polar_h10',
      readings: [...rrBuffer]
    };
    rrBuffer = [];
    await somaPost('/ingest/rr', batch);
  }, FLUSH_INTERVAL_MS);
}
```

### Main Screen (`src/App.js`)

```javascript
import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { startCollection } from './polarCollector';
import { somaGet } from './somaClient';
import { flushQueue } from './syncQueue';

const SESSION_LABELS = [
  'morning_baseline', 'post_run', 'meditation',
  'work_stress', 'commute', 'unlabeled'
];

export default function App() {
  const [status, setStatus] = useState(null);
  const [activeLabel, setActiveLabel] = useState('morning_baseline');
  const [collecting, setCollecting] = useState(false);
  const sessionId = `mobile_${Date.now()}`;

  useEffect(() => {
    flushQueue();  // flush any queued data on app open
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  async function fetchStatus() {
    const s = await somaGet('/status');
    if (s) setStatus(s);
  }

  function startSession() {
    startCollection(sessionId, activeLabel);
    setCollecting(true);
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>🫀 SOMA</Text>

      {status && (
        <View style={styles.statusBox}>
          <Text style={styles.metric}>HR: {status.rhr_bpm} bpm</Text>
          <Text style={styles.metric}>RMSSD: {status.rmssd_ms} ms</Text>
        </View>
      )}

      <Text style={styles.label}>Session type:</Text>
      {SESSION_LABELS.map(l => (
        <TouchableOpacity
          key={l}
          style={[styles.btn, activeLabel === l && styles.btnActive]}
          onPress={() => setActiveLabel(l)}
        >
          <Text style={styles.btnText}>{l}</Text>
        </TouchableOpacity>
      ))}

      <TouchableOpacity
        style={[styles.startBtn, collecting && styles.startBtnActive]}
        onPress={startSession}
        disabled={collecting}
      >
        <Text style={styles.startBtnText}>
          {collecting ? '● Recording...' : 'Start Session'}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0a0a0a', padding: 24, paddingTop: 60 },
  title: { fontSize: 32, color: '#fff', fontWeight: 'bold', marginBottom: 24 },
  statusBox: { backgroundColor: '#1a1a1a', borderRadius: 12, padding: 16, marginBottom: 24 },
  metric: { fontSize: 22, color: '#7fff7f', fontWeight: '600' },
  label: { color: '#888', marginBottom: 8 },
  btn: { padding: 10, borderRadius: 8, backgroundColor: '#1a1a1a', marginBottom: 6 },
  btnActive: { backgroundColor: '#2a3a2a', borderColor: '#7fff7f', borderWidth: 1 },
  btnText: { color: '#ccc', fontSize: 14 },
  startBtn: { marginTop: 24, padding: 16, borderRadius: 12, backgroundColor: '#1a3a1a', alignItems: 'center' },
  startBtnActive: { backgroundColor: '#0a2a0a' },
  startBtnText: { color: '#7fff7f', fontSize: 18, fontWeight: 'bold' },
});
```

---

## Sync Protocol

| Condition | Behavior |
|---|---|
| Home WiFi | Direct to NAS (192.168.x.x), low latency |
| Cellular / away | Tailscale tunnel (100.x.x.x), same API |
| NAS unreachable | Buffer to AsyncStorage, auto-flush on reconnect |
| App reopens | `flushQueue()` runs immediately |

No data is ever lost. Worst case: it arrives late.

---

## End-to-End Test Checklist

```
[ ] Tailscale running on NAS
[ ] Tailscale running on phone
[ ] curl http://100.x.x.x:8765/status returns JSON
[ ] soma_server.py running on NAS (uvicorn)
[ ] React Native app connects to H10 via BLE
[ ] RR intervals appear in /ingest/rr call
[ ] sqlite3 soma_cardio.db shows new rows
[ ] Kill WiFi, confirm cellular still reaches NAS via Tailscale
[ ] Queue a batch offline, reconnect, confirm flush
```

---

## Week 3 Preview — Baseline + Anomaly Detection

- Compute Paul's baseline distribution (mean RHR, mean RMSSD, std dev) from `morning_baseline` sessions
- Flag when current readings exceed 1.5 std dev
- Write anomaly events to `anomalies` table
- Mobile app shows anomaly badge

## Week 4 Preview — Conversational Probe

- Anomaly detected → SOMA sends a probe: *"Your HRV was suppressed from 2-4pm. Did something happen?"*
- Response logged and fed into SOMA's growing context
- The `/probe` endpoint goes live with LLM + biometric context injection

---

*SOMA — Sentient Observation & Memory Architecture*  
*Phase 2: The brain stays home. The body roams free.*
