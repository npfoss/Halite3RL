cd sync
git pull
git add .
git commit -m `python -c "import time; print(int(time.time()*1e9))"`
git push
cp actor.ckpt ..
