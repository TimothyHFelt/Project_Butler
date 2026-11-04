import sqlite3
import numpy as np
import os

# 1. Get the absolute path of the directory containing THIS script
# (Points to: /home/timothyfelt/Project_Butler/src/AI-Identification/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Build the exact path to your data folder cleanly
# (Points to: /home/timothyfelt/Project_Butler/data/project_butler.db)
DB_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/project_butler.db"))

# 3. Ensure the 'data' directory actually exists before opening the DB
# (SQLite will create the file, but it WILL NOT create missing folders)
db = None 

def initialize_database():
    """
    Initializes the database by creating the necessary tables if they do not exist.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# 4. Safely connect using the absolute path
    global db
    db = sqlite3.connect(DB_PATH, check_same_thread=False)  # Allow access from multiple threads
    db.execute("PRAGMA foreign_keys = ON;")  # Ensure foreign key constraints are enforced

    db.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT NOT NULL,
            embedding_vector BLOB NOT NULL,
            image_slot INTEGER CHECK (image_slot BETWEEN 1 AND 5),
            last_used TEXT DEFAULT CURRENT_TIMESTAMP,
            is_user_added INTEGER CHECK (is_user_added IN (0, 1)),
            UNIQUE(subject_name, image_slot)
        )
    """)
    db.commit()  # Commit the transaction to save changes
    return db

def insert_embedding(subject_name: str, embedding_vector: np.ndarray, image_slot: int = None, is_user_added: int = 0):
    """
    Inserts a new embedding vector for a given subject into the database.
    
    Args:
        subject_name: The name of the subject (e.g., "Alice").
        embedding_vector: A 128-dimensional NumPy array representing the face embedding.
        image_slot: The slot number for the image (1-5).
        is_user_added: Flag indicating if the embedding was added by the user.
    """
    
    # Insert into the database
    if _check_vector_count(subject_name) >= 5:
        print(f"[DBManager] Subject '{subject_name}' already has 5 embeddings. Replacing oldest instead.")
        replace_oldest_embedding(subject_name, embedding_vector)
        return
    else:
        embedding_bytes = embedding_vector.astype(np.float32).tobytes()        
        cursor = db.execute("SELECT image_slot FROM embeddings WHERE subject_name = ?", (subject_name,))
        used_slots = [row[0] for row in cursor.fetchall()]
        image_slot = next((slot for slot in range(1, 6) if slot not in used_slots), 1)
        db.execute(
            "INSERT INTO embeddings (subject_name, embedding_vector, image_slot, is_user_added) VALUES (?, ?, ?, ?)",
            (subject_name, embedding_bytes, image_slot, is_user_added)
        )
        db.commit()  # Commit the transaction to save changes

def fetch_embeddings(subject_name: str) -> list:
    """
    Fetches all embedding vectors for a given subject from the database.
    
    Args:
        subject_name: The name of the subject (e.g., "Alice").
        
    Returns:
        A list of NumPy arrays, each representing a 128-dimensional embedding vector.
    """
    cursor = db.execute(
        "SELECT embedding_vector FROM embeddings WHERE subject_name = ?",
        (subject_name,)
    )
    
    embeddings = []
    for row in cursor.fetchall():
        embedding_bytes = row[0]
        # Convert bytes back to a NumPy array
        embedding_vector = np.frombuffer(embedding_bytes, dtype=np.float32)
        embeddings.append(embedding_vector)
    
    return embeddings

def remove_subject(subject_name: str):
    """
    Removes all embeddings associated with a given subject from the database.
    
    Args:
        subject_name: The name of the subject to remove (e.g., "Alice").
    """
    db.execute(
        "DELETE FROM embeddings WHERE subject_name = ?",
        (subject_name,)
    )
    db.commit()  # Commit the transaction to save changes

def remove_embedding(subject_name: str, embedding_vector: np.ndarray):
    """
    Removes a specific embedding vector for a given subject from the database.
    
    Args:
        subject_name: The name of the subject (e.g., "Alice").
        embedding_vector: The specific 128-dimensional NumPy array to remove.
    """
    # Convert the NumPy array to bytes for comparison
    embedding_bytes = embedding_vector.astype(np.float32).tobytes()
    
    db.execute(
        "DELETE FROM embeddings WHERE subject_name = ? AND embedding_vector = ?",
        (subject_name, embedding_bytes)
    )
    db.commit()  # Commit the transaction to save changes

def remove_all_embeddings():
    """
    Removes all embeddings from the database. Use with caution!
    """
    db.execute("DELETE FROM embeddings")
    db.commit()  # Commit the transaction to save changes

def _check_vector_count(subject_name: str) -> int:
    """
    Checks the number of embedding vectors stored for a given subject.
    
    Args:
        subject_name: The name of the subject (e.g., "Alice").
        
    Returns:
        The count of embedding vectors for the subject.
    """
    cursor = db.execute(
        "SELECT COUNT(*) FROM embeddings WHERE subject_name = ?",
        (subject_name,)
    )
    count = cursor.fetchone()[0]
    return count

def touch_embedding(subject_name: str, embedding_vector: np.ndarray):
    """
    Updates the last_used timestamp for a specific embedding vector of a given subject.
    
    Args:
        subject_name: The name of the subject (e.g., "Alice").
        embedding_vector: The specific 128-dimensional NumPy array to update.
    """
    # Convert the NumPy array to bytes for comparison
    embedding_bytes = embedding_vector.astype(np.float32).tobytes()
    
    db.execute(
        "UPDATE embeddings SET last_used = CURRENT_TIMESTAMP WHERE subject_name = ? AND embedding_vector = ?",
        (subject_name, embedding_bytes)
    )
    db.commit()  # Commit the transaction to save changes

def replace_oldest_embedding(subject_name: str, new_embedding_vector: np.ndarray):
    """
    Replaces the oldest embedding vector for a given subject with a new one.
    
    Args:
        subject_name: The name of the subject (e.g., "Alice").
        new_embedding_vector: The new 128-dimensional NumPy array to insert.
    """
    new_bytes = new_embedding_vector.astype(np.float32).tobytes()

    # Fetch the oldest embedding vector for the subject
    cursor = db.execute("""
        SELECT id, image_slot 
        FROM embeddings 
        WHERE subject_name = ? AND is_user_added = 0 
        ORDER BY last_used ASC 
        LIMIT 1
    """, (subject_name,))
    row = cursor.fetchone()
    
    if row:
        row_id, image_slot = row
        # Delete the oldest embedding
        db.execute("""
            UPDATE embeddings
            SET embedding_vector = ?, last_used = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (new_bytes, row_id))
        db.commit()
        print(f"[DBManager] Successfully replaced oldest system vector in slot {image_slot} for '{subject_name}'.")
    else:
        print(f"[DBManager] Replacement failed. All 5 images for '{subject_name}' are user-protected.")

def close_database():
    if db:
        db.close()
        
initialize_database()  # Ensure the database is initialized when the module is imported