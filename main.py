import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "DDColor"))

if __name__ == "__main__":
    from backend.app import app
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Flask ColorizeAI UI on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)