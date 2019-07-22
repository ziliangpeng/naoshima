from flask import Flask, escape, request
import gflags
import glog
import os

FLAGS = gflags.FLAGS

gflags.DEFINE_string('dir', 'links', 'path to store links')

app = Flask(__name__)


@app.route('/set')
def put():
    return f'setting'


@app.route('/list')
def ls():
    abs_path = os.path.abspath(FLAGS.dir)

    html = f"<html><body>"
    for fname in os.listdir(abs_path):
        with open(os.path.join(abs_path, fname)) as f:
            content = f.read()
            html += fname + '<br>'
            html += content + '<br>'
    html += "</body></html>"
    return html


@app.route('/', defaults={'name': ''})
@app.route('/<path:name>')
def get(name):
    glog.info('fetching ' + name)
    abs_path = os.path.abspath(FLAGS.dir)
    file_path = os.path.join(abs_path, name)
    glog.info('abs path is ' + file_path)

    try:
        with open(file_path) as f:
            content = f.read()
        return f'path: {content}'
    except:
        return f'path: not found'
