import cv2
import numpy as np
import time
import os
import asyncio
from core.splicer import Splicer, SpliceFailureException
from core.identifier import Identifier
from services.db_manager import DBManager
from services.dispatcher import Dispatcher

class Runner:
    def __init__(self, splicer: Splicer, identifier: Identifier, db_manager: DBManager, dispatcher: Dispatcher, watch_folder: str = "/dev/shm/butler_watch/", cooldown_seconds: int = 120):
        """
        Initializes the central system Orchestrator (The Runner).
        """
        self.splicer = splicer
        self.identifier = identifier
        self.db_manager = db_manager
        self.dispatcher = dispatcher
        self.watch_folder = watch_folder
        self.cooldown_seconds = cooldown_seconds
        
        os.makedirs(self.watch_folder, exist_ok=True)
        
        # Load Whitelist Cache directly into RAM for fast inference lookups
        self.whitelist_cache = self.db_manager.load_all_whitelist_embeddings()
        
        # Session State Parameters to eliminate duplicate notification spam
        self.active_session_identity = None
        self.last_session_timestamp = 0
        self.unknown_session_embeddings = []

    async def start_folder_monitor_loop(self):
        """
        Monitors the RAM disk watch folder, automatically grabs frames 
        when a full burst sequence arrives, and cleans up the paths natively.
        """
        print(f"[Runner Engine] Active folder-watch loop online. Monitoring: {self.watch_folder}")
        last_file_seen_time = time.time()
        
        while True:
            all_files = sorted([
                os.path.join(self.watch_folder, f) 
                for f in os.listdir(self.watch_folder) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            
            if all_files and (time.time() - last_file_seen_time > 2.0):
                print("[Runner Engine] Stream timeout detected (>2s gap). Purging stale folder fragments...")
                for file_path in all_files:
                    try: os.remove(file_path)
                    except OSError: pass
                all_files = []

            if len(all_files)  bool:
        """Evaluates whether the visitor matches the active porch session."""
        current_time = time.time()
        
        if current_time - self.last_session_timestamp > self.cooldown_seconds:
            self.active_session_identity = None
            self.unknown_session_embeddings = []
            return False

        if identity != "Unknown" and self.active_session_identity == identity:
            self.last_session_timestamp = current_time
            return True

        if identity == "Unknown" and self.active_session_identity == "Unknown":
            for live_emb in current_embeddings:
                for saved_emb in self.unknown_session_embeddings:
                    similarity = self.identifier._calculate_cosine_similarity(live_emb, saved_emb)
                    if similarity > 0.80:
                        self.last_session_timestamp = current_time
                        return True
        return False

    async def process_stream_burst(self, frame_stream: list):
        """
        Executes Section 4.1's complete Photo Pipeline across the 7-frame bundle.
        """
        # 1. Wakeup and burst: Discard first 3 camera-settling exposure frames
        burst_pool = frame_stream[3:]
        
        valid_crops = []
        burst_identities = []
        live_embeddings = []

        # 2. Splice: Pass stable frames through the Stateless Splicer
        for frame_idx, frame in enumerate(burst_pool):
            try:
                spliced_data_list = self.splicer.splice(frame)
                if spliced_data_list:
                    valid_crops.append(spliced_data_list)
            except SpliceFailureException as e:
                print(f"[Runner -> Splicer Notice] Frame {frame_idx + 1} validation failed: {e}")
                continue

        if not valid_crops:
            print("[Runner Error] Section 4.1 Step 2 Failure: Zero valid face crops extracted.")
            return

        # 3. Identify: Pass crops through the Stateless Identifier
        for crop_data in valid_crops:
            img_matrix = crop_data['crop']
            emb = self.identifier.extract_embedding(img_matrix)
            live_embeddings.append(emb)
            
            identity, score, _ = self.identifier.identify(img_matrix, self.whitelist_cache)
            burst_identities.append(identity)

        # 4. Burst Agreement Check: Verify identical match across >= 2 frames
        identity_counts = {}
        for name in burst_identities:
            if name != "Unknown":
                identity_counts[name] = identity_counts.get(name, 0) + 1

        final_consensus_identity = "Unknown"
        if identity_counts:
            best_match_name = max(identity_counts, key=identity_counts.get)
            if identity_counts[best_match_name] >= 2:
                final_consensus_identity = best_match_name

        # 5. Anti-Spam Evaluation
        if self._should_suppress_notification(final_consensus_identity, live_embeddings):
            print(f"[Runner] Suppression Active: Silently dropping duplicate loop for [{final_consensus_identity}].")
            return

        session_id = f"sess_{int(time.time())}"
        self.last_session_timestamp = time.time()
        self.active_session_identity = final_consensus_identity

        # =====================================================================
        # 6. UNIFIED ESCALATION HANDLER — ALL ROADS LEAD TO TRIAGE
        # =====================================================================
        print(f"[Runner] Processing Identification Result: [{final_consensus_identity}]")
        
        # Enforce Section 4 clustering to extract the single best JPEG crop from the burst matrix pool
        crop_bytes, winning_emb = self.dispatcher.cluster_unknown_burst(valid_crops, live_embeddings)
        
        if final_consensus_identity != "Unknown":
            # Update the existing profile reference timestamp record in SQLite 
            self.db_manager.update_timestamp(final_consensus_identity, winning_emb)
            print(f"[Runner] Whitelist Match verified. Updated last_matching_timestamp for {final_consensus_identity}.")
        else:
            # Anchor this unknown stranger's vector geometry into session memory to prevent notification spam
            self.unknown_session_embeddings = live_embeddings

        # Dispatch the alert payload to the iPhone, passing the identified name (or "Unknown") as metadata
        print(f"[Runner] Dispatching triage prompt to iPhone for session {session_id}...")
        user_action = await self.dispatcher.dispatch_escalated_alert(session_id, crop_bytes, identity=final_consensus_identity)
        
        print(f"[Runner] iPhone Action Response Hook Caught: '{user_action}'")
        
        # Process State 6 Final physical execution parameters strictly based on what you tap on your screen
        if user_action == "Unlock":
            print(f"[Runner] State 6A Authorized: Owner approved access for visitor [{final_consensus_identity}]. Triggering lock actuators...")
            # TODO: Dispatch the actual unlock packet command to Node 3 microcontroller
            
        elif user_action == "Busy":
            print("[Runner] State 6B: Active Deny. Sending signal to local audio module node...")
            # TODO: Trigger Audio Node 2 TTS stream playback
            
        elif user_action == "Jokes":
            print("[Runner] State 6B Bonus: Joke request triggered. Generating audio file data...")
            # TODO: Trigger Audio Node 2 Joke file playback
            
        else:
            print("[Runner] State 6C: Default Secure. Clearing volatile memory streams. Entryway remains locked.")
