#!/bin/zsh
# Keep the long-running experiment daemons alive.
#
# queue_runner and harvest both self-terminate on a time budget (they are launched
# as background tasks that get reaped roughly hourly). Rather than re-arming each
# one by hand every time, this restarts whichever is missing. Logs append so the
# history survives restarts.
cd "$(dirname "$0")/.." || exit 1
while true; do
  pgrep -f "polychart.queue_runner" >/dev/null 2>&1 || \
    nohup python3 -u -m polychart.queue_runner 55 >> data/runs/queue.log 2>&1 &
  pgrep -f "polychart.harvest" >/dev/null 2>&1 || \
    nohup python3 -u -m polychart.harvest 50 >> data/runs/harvest.log 2>&1 &
  pgrep -f "polychart.watch_jobs" >/dev/null 2>&1 || \
    nohup python3 -u -m polychart.watch_jobs >> data/runs/watch.log 2>&1 &
  # Refresh the offline analysis every cycle (no API calls, so it never competes
  # with the runners for the rate limiter).
  python3 -u -m polychart.report > /dev/null 2>&1
  sleep 180
done
