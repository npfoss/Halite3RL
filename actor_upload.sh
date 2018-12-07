param () { python3 -c "import json; print(json.load(open('$1'))['$2'] + 1)"; }
url () { python3 -c "import json; print((lambda x: x['base_url']+'/$2?secret='+x['secret'])(json.load(open('$1'))))"; }
curl `url params.json download/weights` > actor.ckpt
cd sync/replays
ls *.phlt | while read line; do
	curl -X POST -F "file=@$line" `url ../../params.json upload/replay`
	rm $line
done
