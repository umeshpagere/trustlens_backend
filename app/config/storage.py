import os

# Render and most Linux environments use /tmp for ephemeral writable storage
TEMP_DIR = "/tmp/trustlens"

def init_temp_storage():
    """
    Ensures the temporary directory for media processing exists.
    Called during application startup.
    """
    if not os.path.exists(TEMP_DIR):
        print(f"📁 [Storage] Creating temporary directory: {TEMP_DIR}")
        os.makedirs(TEMP_DIR, exist_ok=True)
    else:
        print(f"📁 [Storage] Using existing temporary directory: {TEMP_DIR}")

def get_temp_path(filename: str) -> str:
    """Returns an absolute path within the temporary directory."""
    return os.path.join(TEMP_DIR, filename)
