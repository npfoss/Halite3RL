from flask import Flask, request, abort, send_from_directory, send_file
import os, json, functools
app = Flask(__name__)

relpath = lambda path: os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
replays_folder = relpath("replays")
weights_file = relpath("actor.ckpt")

with open(relpath("../params.json")) as f:
	params = json.load(f)

def secret(f):
	@functools.wraps(f)
	def wrapper(*args, **kwargs):
		print('check')
		if request.args.get('secret') == params["secret"]:
			return f(*args, **kwargs)
		else:
			abort(401)
	return wrapper

@app.route('/upload/replay', methods = ["POST"])
@secret
def upload_replay():
	upload(False, "phlt", "replays")
	return ""

@app.route('/upload/weights', methods = ["POST"])
@secret
def upload_weights():
	upload("actor.ckpt")
	return ""

def upload(fname = False, ext = False, path = False):
	for f in request.files.values():
		if f == None or f.filename == "":
			abort(400)
		fname = fname or ((f.filename.rsplit(".", 1)[0] if "." in f.filename else "") + "." + ext if ext else f.filename)
		f.save(os.path.join(replays_folder, fname) if path else relpath(fname))

@app.route('/download/replay/<filename>')
@secret
def download_replay(filename):
	res = send_from_directory(replays_folder, filename + ".phlt", as_attachment=True)
	os.remove(os.path.join(replays_folder, filename + ".phlt"))
	return res

@app.route('/download/weights')
@secret
def download_weights():
	return send_file(relpath('actor.ckpt'), as_attachment=True)

@app.route('/list/replays')
@secret
def list():
	return "\n".join([i.rsplit('.', 1)[0] for i in os.listdir(replays_folder)])

if __name__ == "__main__":
	if not os.path.exists(replays_folder):
		os.makedirs(replays_folder)
	app.run(port = 8000, debug = False, threaded = True)
