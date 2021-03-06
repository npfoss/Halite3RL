param () { python3 -c "import json; print(json.load(open('$1'))['$2'] + 1)"; }
url () { python3 -c "import json; print((lambda x: x['base_url']+'/$2?secret='+x['secret'])(json.load(open('$1'))))"; }
curl -X POST -F weights=@actor.ckpt `url params.json upload/weights`
cd replays
ls -1r *.phlt | tail -n +`param ../params.json disk_buffer_size` | xargs rm
curl --silent `url ../params.json list/replays` | while read line; do
	curl --silent -OJ `url ../params.json "download/replay/$line"`
done
