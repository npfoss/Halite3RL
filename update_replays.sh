param () { python3 -c "import json; print(json.load(open('$1'))['$2'] + 1)" }
url () { python3 -c "import json; print((lambda x: x['base_url']+'/$2?secret='+x['secret'])(json.load(open('$1'))))"; }
if [ `python -c 'import os, json, time; print(int(time.time() - os.stat(".git/FETCH_HEAD").st_mtime > json.load(open("params.json"))["pull_frequency"]))'` ]; then
	curl -F @weights=actor.ckpt `url params.json upload/weights`
	cd replays
	curl `url ../params.json list/replays` | while read line; do
		curl -OJ `url ../params.json "download/replay/$line"`
	done
	ls -1 *.phlt | tail -n +`url ../params.json disk_buffer_size` | xargs rm
fi
