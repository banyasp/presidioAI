import requests
import time
import subprocess
import sys
import os

def test_backend():
    print("Starting backend server for testing...")
    # Start the server in the background
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd()
    )
    
    try:
        # Wait for server to start
        print("Waiting for server to start...")
        time.sleep(10) # Give it time to load the model and DB
        
        # Test Health
        try:
            resp = requests.get("http://localhost:8000/health")
            print(f"Health Check: {resp.status_code} - {resp.json()}")
        except Exception as e:
            print(f"Health check failed: {e}")
            return

        # Test Analyze
        print("Testing /analyze endpoint...")
        payload = {
            "text": "The defendant was denied counsel during the trial."
        }
        resp = requests.post("http://localhost:8000/analyze", json=payload)
        
        if resp.status_code == 200:
            data = resp.json()
            print("Analyze Success!")
            print(f"Found {len(data['results'])} results.")
            if data['results']:
                print("Top result case:", data['results'][0]['case_name'])
        else:
            print(f"Analyze Failed: {resp.status_code} - {resp.text}")

    finally:
        print("Stopping server...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    test_backend()
