cd sync
git pull
ls -1 replays/*.phlt | tail -n +`python3 -c 'import json; print(json.load(open("params.json"))["disk_buffer_size"] + 1)'` | xargs rm
cp ../actor.ckpt .
git add .
git commit -m `python3 -c "import time; print(int(time.time()*1e9))"`
git push

