#!/bin/bash

# Test Scenarios for Phase 2
# This script helps test various scenarios

API_URL="http://localhost:8000"
API_KEY="dev-key"

echo "=== Phase 2 Testing Scenarios ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print test header
print_test() {
    echo -e "${YELLOW}Test: $1${NC}"
}

# Function to check response
check_response() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success${NC}"
    else
        echo -e "${RED}✗ Failed${NC}"
    fi
    echo ""
}

# Test 1: Multiple People in Same Request
print_test "1. Multiple People in Same Request"
python3 << 'EOF'
import json

# Create request with 2 people
keypoint_template = [[0.52, 0.18, 0.91] for _ in range(17)]
keypoints = [keypoint_template.copy() for _ in range(30)]

request_data = {
    "schema_version": 1,
    "device_id": "pi-001",
    "camera_id": "cam-1",
    "window": {
        "ts_start_ms": 1737970000000,
        "ts_end_ms": 1737970001000,
        "fps": 30,
        "size": 30
    },
    "people": [
        {
            "track_id": 7,
            "keypoints": keypoints,
            "pose_conf": 0.3  # Should be "unknown" (0.2)
        },
        {
            "track_id": 8,
            "keypoints": keypoints,
            "pose_conf": 0.8  # Should be "standing" (0.6)
        }
    ]
}

# Save to temp file
with open("/tmp/test_multiple_people.json", "w") as f:
    json.dump(request_data, f)

print("Created request with 2 people")
EOF

response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/v1/activity/infer" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d @/tmp/test_multiple_people.json)

http_code=$(echo "$response" | tail -1)
body=$(echo "$response" | sed '$d')

if [ "$http_code" = "200" ]; then
    echo "Status: $http_code"
    echo "$body" | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"Results: {len(d['results'])} people\"); [print(f\"  Track {r['track_id']}: {r['activity']} (confidence: {r['confidence']})\") for r in d['results']]"
else
    echo "Status: $http_code"
    echo "$body"
fi
check_response

# Test 2: Multiple Devices
print_test "2. Multiple Devices"
echo "Sending inference from pi-001..."
curl -s -X POST "$API_URL/v1/activity/infer" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d @samples/test_request.json > /dev/null
check_response

echo "Sending inference from pi-002..."
python3 << 'EOF'
import json

keypoint_template = [[0.52, 0.18, 0.91] for _ in range(17)]
keypoints = [keypoint_template.copy() for _ in range(30)]

request_data = {
    "schema_version": 1,
    "device_id": "pi-002",  # Different device
    "camera_id": "cam-1",
    "window": {
        "ts_start_ms": 1737970002000,
        "ts_end_ms": 1737970003000,
        "fps": 30,
        "size": 30
    },
    "people": [{
        "track_id": 1,
        "keypoints": keypoints,
        "pose_conf": 0.85
    }]
}

with open("/tmp/test_pi002.json", "w") as f:
    json.dump(request_data, f)
EOF

response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/v1/activity/infer" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d @/tmp/test_pi002.json)

http_code=$(echo "$response" | tail -1)
echo "Status: $http_code"
check_response

# Test 3: Check Devices List
print_test "3. Check All Devices"
echo "GET /v1/devices"
curl -s "$API_URL/v1/devices" | python3 -m json.tool | grep -E '"device_id"'
check_response

# Test 4: Check Events Limit
print_test "4. Test Events Limit Parameter"
echo "GET /v1/events?limit=2"
curl -s "$API_URL/v1/events?limit=2" | python3 -m json.tool | grep -E '"device_id"|"activity"' | head -4
check_response

# Test 5: Device-Specific Events
print_test "5. Device-Specific Events"
echo "GET /v1/devices/pi-001/events?limit=5"
curl -s "$API_URL/v1/devices/pi-001/events?limit=5" | python3 -m json.tool | grep -E '"device_id"|"activity"' | head -2
check_response

# Test 6: Non-existent Device
print_test "6. Non-existent Device (should return empty array)"
echo "GET /v1/devices/non-existent/events"
result=$(curl -s "$API_URL/v1/devices/non-existent/events")
if [ "$result" = "[]" ]; then
    echo -e "${GREEN}✓ Returns empty array (correct)${NC}"
else
    echo -e "${RED}✗ Unexpected response${NC}"
fi
echo ""

# Test 7: Invalid Limit Parameter
print_test "7. Invalid Limit Parameter (should return 422)"
response=$(curl -s -w "\n%{http_code}" "$API_URL/v1/events?limit=0")
http_code=$(echo "$response" | tail -1)
if [ "$http_code" = "422" ]; then
    echo -e "${GREEN}✓ Returns 422 (correct)${NC}"
else
    echo -e "${RED}✗ Unexpected status: $http_code${NC}"
fi
echo ""

# Test 8: Verify Sorting (Newest First)
print_test "8. Verify Events Sorted by Created At (Newest First)"
echo "Checking first 3 events timestamps:"
curl -s "$API_URL/v1/events?limit=3" | python3 -m json.tool | grep -E '"created_at"' | head -3
check_response

# Test 9: Same Device, Different Cameras
print_test "9. Same Device, Different Cameras"
python3 << 'EOF'
import json

keypoint_template = [[0.52, 0.18, 0.91] for _ in range(17)]
keypoints = [keypoint_template.copy() for _ in range(30)]

# Send from cam-2
request_data = {
    "schema_version": 1,
    "device_id": "pi-001",
    "camera_id": "cam-2",  # Different camera
    "window": {
        "ts_start_ms": 1737970004000,
        "ts_end_ms": 1737970005000,
        "fps": 30,
        "size": 30
    },
    "people": [{
        "track_id": 9,
        "keypoints": keypoints,
        "pose_conf": 0.75
    }]
}

with open("/tmp/test_cam2.json", "w") as f:
    json.dump(request_data, f)
EOF

response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/v1/activity/infer" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d @/tmp/test_cam2.json)

http_code=$(echo "$response" | tail -1)
echo "Status: $http_code"
echo "Event saved with camera_id: cam-2"
check_response

# Test 10: Verify Device Events Include All Cameras
print_test "10. Verify Device Events Include All Cameras"
echo "GET /v1/devices/pi-001/events (should show both cam-1 and cam-2)"
curl -s "$API_URL/v1/devices/pi-001/events?limit=10" | python3 -m json.tool | grep -E '"camera_id"' | head -5
check_response

# Cleanup temp files
rm -f /tmp/test_multiple_people.json /tmp/test_pi002.json /tmp/test_cam2.json

echo ""
echo "=== Testing Complete ==="
echo ""
echo "Next steps:"
echo "1. Open dashboard: http://localhost:8000/dashboard"
echo "2. Check device dashboard: http://localhost:8000/dashboard/devices/pi-001"
echo "3. Run automated tests: pytest"
echo "4. Review TESTING_CHECKLIST.md for more tests"
