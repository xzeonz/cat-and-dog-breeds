#!/bin/bash

# Test root endpoint
echo "Testing root endpoint..."
curl -X GET http://localhost:8000/
echo -e "\n"

# Test prediction endpoint (adjust data as needed)
echo "Testing prediction endpoint..."
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"input_data": "sample"}'
echo -e "\n"

echo "API endpoint tests completed."
