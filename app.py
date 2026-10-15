"""
app.py — VisionInspect AI React Enterprise Dashboard Entrypoint

Launches the unified FastAPI backend serving the React SPA on port 8000.
"""

import uvicorn
from api import app

if __name__ == "__main__":
    print("=" * 60)
    print(" 🚀 Launching VisionInspect AI React Enterprise Dashboard...")
    print(" URL: http://127.0.0.1:8000")
    print("=" * 60)
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)