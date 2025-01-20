#!/bin/bash

while true; do
  curl -X POST http://localhost:35001/predict \
       -H "Content-Type: application/json" \
       -d '{"latitude": 33.87, "longitude": -98.59, "elevation": 307, "season": "Summer", "weather": "Clear"}'
  echo "Query sent. Waiting 5 seconds..."
  sleep 5
done
