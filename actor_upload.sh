param () { python3 -c "import json; print(json.load(open('$1'))['$2'] + 1)"; }
url () { python3 -c "import json; print((lambda x: x['base_url']+'/$2?secret='+x['secret'])(json.load(open('$1'))))"; }
curl -OJ `url params.json download/weights`
cd sync/replays
ls *.phlt | while read line; do
	curl -X POST -F "file=@$line" `url ../../params.json upload/replay`
done
