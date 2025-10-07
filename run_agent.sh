#!/bin/bash

set -e

pwd=`pwd`
if [ ! -d $(pwd)/db ]; then
    python download_sample_db.py
fi

gunicorn app.gradio.app:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
