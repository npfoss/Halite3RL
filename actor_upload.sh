cd sync
git add .
git commit -m `python3 -c "import time; print(int(time.time()*1e9))"`
git pull --no-edit && git pull --no-edit && git pull --no-edit
git push
cp actor.ckpt ..
