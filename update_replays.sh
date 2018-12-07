if [ `python -c 'import os, json, time; print(int(time.time() - os.stat(".git/FETCH_HEAD").st_mtime > json.load(open("params.json"))["pull_frequency"]))'` ]; then
    mv sync/replays/*.phlt replays/
    ls -1 replays/*.phlt | tail -n +`python3 -c 'import json; print(json.load(open("params.json"))["disk_buffer_size"] + 1)'` | xargs rm
    cd sync
    git add .
    git commit -m `python3 -c "import time; print(int(time.time()*1e9))"`
    git pull --no-edit && git pull --no-edit && git pull --no-edit
    git push
fi
