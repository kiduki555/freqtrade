#!/usr/bin/env bash
# Download candle data for OHIO pipeline warmup.
# Requires: freqtrade installed, config file with exchange settings.
#
# Usage: ./scripts/download_ohio_data.sh [--days DAYS] [--config CONFIG]

set -euo pipefail

DAYS="${1:-220}"  # 180 days warmup + 40 days test buffer
CONFIG="${2:-user_data/config_download.json}"

echo "=== OHIO Data Download ==="
echo "Days: $DAYS"
echo "Config: $CONFIG"

freqtrade download-data \
    --config "$CONFIG" \
    --timeframes 1h 4h \
    --days "$DAYS" \
    --erase

echo "=== Download complete ==="
echo "Data location: user_data/data/"

# Verify minimum row count for 1h data
echo "=== Verifying data ==="
python -c "
import pathlib, sys
data_dir = pathlib.Path('user_data/data')
if not data_dir.exists():
    print('ERROR: data directory not found'); sys.exit(1)
feather_files = list(data_dir.rglob('*-1h.feather'))
if not feather_files:
    print('WARNING: No 1h feather files found')
    sys.exit(0)
import pandas as pd
for f in feather_files:
    df = pd.read_feather(f)
    print(f'{f.name}: {len(df)} rows ({len(df)/24:.0f} days)')
    if len(df) < 4320:
        print(f'  WARNING: {f.name} has < 4320 rows (180 days warmup)')
"
