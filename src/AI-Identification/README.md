# Project Butler — AI Core Engine (`src/`)

Welcome to the central intelligence hub of **Project Butler**. This repository contains the high-performance, asynchronous Python backend running on the **Raspberry Pi 5**. 

The system implements a radically decoupled, **Orchestration-based Microservices pattern** designed for secure entry identification, lookalike management, and human-in-the-loop smartphone verification. No sub-service communicates with another directly; every frame, crop, biometric score, and network notice passes exclusively through the central orchestrator (**The Runner**).

---

## 📂 System Directory Layout

Ensure your files are organized within the single, self-contained `AI-identification` folder on your persistent NVMe storage array exactly as shown below:

```text
src/
└── AI-identification/
    ├── main.py          # System master entry point and async event runtime loop
    ├── runner.py        # The Orchestrator: Repository of system state & file-grabbing engine
    ├── splicer.py       # The Cutter: Stateless face detection via YuNet (OpenCV)
    ├── identifier.py    # The Mathematician: Stateless SFace feature vector matching
    ├── db_manager.py    # The Accountant: SQLite gatekeeper & 5-photo LRU cache
    └── dispatcher.py    # The Bridge: Outbound HTTPS push engine & iPhone webhook link
```

> **Volatile Memory Note:** The system actively relies on a Linux RAM disk directory path located at `/dev/shm/butler_watch/` to ingest incoming camera streams. This completely bypasses flash drive wear-leveling constraints on your native NVMe SSD drive.

---

## 🧠 The Architecture Pipeline (Step-by-Step)

The engine functions as a stateful, event-driven loop following strict system parameters:

1. **The Accumulation Gate:** External systems continuously write captured visitor frames directly into the volatile RAM disk folder `/dev/shm/butler_watch/`. The `Runner` constantly checks this path, **holding execution until exactly 7 images accumulate**. It then loads the arrays into RAM matrix memory and purges the directory instantly to reclaim space.
2. **The Camera Exposure Cut:** The `Runner` performs a clean slice (`frame_stream[3:]`), **discarding the first 3 frames** to allow camera image sensor auto-exposure to stabilize. The remaining 4 high-quality frames form the analytical burst pool.
3. **Stateless Splicing (The Cutter):** The `Splicer` uses **YuNet** to scan the remaining 4 frames. It enforces strict structural thresholds: dropping frames where face scale is too small ($<40\text{px}$) or where two boxes clip shoulder-to-shoulder (Intersection over Union overlapping $>0.30$) to block anti-tailgating or enrollment poisoning risks.
4. **Biometric Fingerprinting (The Mathematician):** Valid face crops are leveled and fed into the **SFace** model inside the `Identifier` to extract a unique 128-float32 mathematical embedding vector. 
5. **Lookalike Sibling Lockout Override:** The `Identifier` runs a best-of-5 comparison loop against your whitelist cache using cosine similarity. To protect lookalike family members from permanently deadlocking the system, if a score clears an absolute certainty line ($\ge 0.55$), the system overrides the standard $0.10$ runner-up separation margin and authenticates the name immediately. Fuzzier scores must clear the margin check or default to `"Unknown"`.
6. **Burst Agreement Verification:** To protect against glitchy false-positives, **the exact same identity must be returned on at least 2 distinct frames** of the 4-frame pool to establish a final consensus name.
7. **Anti-Notification Suppression:** If the final consensus identity (Known or Unknown) matches the active porch session state recorded in memory within a rolling 2-minute window, **the alert is silently dropped**. A fresh notification is dispatched only if the cooldown expires or a completely *new* person enters the biometric loop.
8. **Unified Outbound Dispatching:** The system remains **locked by default** for everyone. Instead of unlocking automatically, the `Runner` calls the `Dispatcher` to compress the burst into the single highest-confidence JPEG crop, attach the identified name string or `"Unknown"` as metadata, and fire a secure outbound HTTPS request directly to your iPhone infrastructure to dynamically pre-fill your notification layouts.

---

## 🛑 The 30-Second Human-in-the-Loop Gate

Once an alert is dispatched to your iPhone, the local engine initializes a strict **30-second countdown gatekeeper task** (`asyncio.Future`). The gateway remains locked by default. The system pauses its path for this event, waiting for your input:

* 🔘 **Owner taps `[Unlock]`:** Your mobile app triggers a secure back-channel HTTP POST request to the local Pi server. The webhook hooks back into the pending task thread, and the Runner commands Node 3 to actuate the door lock relay.
* 🔘 **Owner taps `[Busy]` / `[Jokes]`:** The system commands your porch Audio Node to play a Text-to-Speech response or generate a joke audio track over the speaker.
* 🔘 **Owner ignores / 30 seconds expire:** The system drops seamlessly into **State 6C (Default Secure)**. The volatile memory arrays are immediately wiped out of memory, no database tables are polluted with unconfirmed metadata, and the front entryway remains safely locked down.

---

## 💾 SQLite Database Management (Capacity Rules)

The `DBManager` manages your user profiles using a strict **Least Recently Used (LRU) Cache Strategy** to balance automatic learning with high accuracy:
* **The Profile Balance:** Each enrolled profile can hold a maximum of 5 images. At most 3 can be tagged `manual` (ground-truth anchors verified by you via the app), leaving at least 2 slots open for system-collected `auto` variants to dynamically learn changing porch shadows or facial hair.
* **The Cache Loop:** When a new `auto` image clears the rigorous auto-save quality gate, the database selects the auto photo with the *oldest* matching timestamp, purges it from disk, and inserts the fresh crop. 
* **Write Ordering Rule:** To neutralize unexpected power outages on your porch, **the database row change commits first, and the file system delete follows.** A crash during execution leaves a harmless orphan file instead of a corrupted database pointer pointing at an empty path.

---

## 🚀 Quickstart System Deployment

### 1. Install System Dependencies
Ensure your Python environment contains the necessary computer vision and high-performance networking libraries:
```bash
pip install opencv-python numpy httpx
```

### 2. Download Core Weights
Place your trained open-source ONNX computer vision models straight into an accessible directory (e.g., `src/AI-identification/models/`):
* `face_detection_yunet.onnx` (OpenCV YuNet face tracker)
* `face_recognition_sface.onnx` (OpenCV SFace vector extractor)

### 3. Initialize the Master System
Kick off the automated folder watch loop natively via your command line terminal interface:
```bash
python AI-identification/main.py
```
