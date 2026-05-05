#!/bin/bash
# Kill any processes listening on port 8000
PID=$(netstat -tlnp | grep ':8000' | awk '{print $7}' | cut -d'/' -f1)
if [ ! -z "$PID" ]; then
    echo "Killing process on 8000: $PID"
    kill -9 $PID
fi
pkill -9 -f uvicorn
pkill -9 -f spawn_main
pkill -9 -f multiprocessing
pkill -9 -f test_chat
sleep 2

# Clean the env variable and start uvicorn without --reload to avoid multiprocessing leaks
unset OPENAI_BASE_URL
cd /data
rm -f /tmp/uvicorn.log
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/uvicorn.log 2>&1 &
echo "Uvicorn started."
