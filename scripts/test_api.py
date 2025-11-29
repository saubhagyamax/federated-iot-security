import requests
from datetime import datetime

BASE_URL = "http://localhost:8000"

# Test 1: Register a device
print("1. Registering device...")
device_data = {
    "device_id": "test-device-001",
    "device_type": "camera",
    "name": "Test Smart Camera",
    "organization_id": "org-test-001",
    "baseline_metrics": {
        "cpu": 15.0,
        "network": 50.0,
        "requests": 20.0
    }
}
response = requests.post(f"{BASE_URL}/api/v1/devices/register", json=device_data)
print(f"   Status: {response.status_code}")
print(f"   Response: {response.json()}\n")

# Test 2: Submit metrics
print("2. Submitting metrics...")
metrics_data = {
    "device_id": "test-device-001",
    "timestamp": datetime.utcnow().isoformat(),
    "cpu_usage": 16.5,
    "network_traffic": 52.0,
    "request_count": 22,
    "connection_attempts": 5,
    "memory_usage": 45.0
}
response = requests.post(f"{BASE_URL}/api/v1/metrics", json=metrics_data)
print(f"   Status: {response.status_code}")
print(f"   Response: {response.json()}\n")

# Test 3: Get stats
print("3. Getting platform statistics...")
response = requests.get(f"{BASE_URL}/api/v1/stats")
print(f"   Status: {response.status_code}")
print(f"   Response: {response.json()}\n")

print("✅ All tests completed!")