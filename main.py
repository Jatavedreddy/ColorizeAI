import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "DDColor"))

if __name__ == "__main__":
    from backend.app import app
    print("Starting Flask ColorizeAI UI...")
    app.run(host="127.0.0.1", port=8080, debug=True)
