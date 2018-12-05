import json
from baselines.acer.runner import HaliteRunner
import subprocess
import time

with open("params.json") as f:
    runs = json.load(f)["actor_runs_per_upload"]
runner = HaliteRunner()
current_proc = None
while True:
    for i in range(runs):
        runner.run()
    while current_proc is not None and current_proc.poll() is None:
        # not done yet!
        print("waiting on previous upload")
        # wait n seconds
        time.sleep(1)
    current_proc = subprocess.Popen(['sh', 'actor_upload.sh'])

