from flask import Flask, escape, request, abort, redirect
import gflags
import glog
import os
import sys

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


@app.route('/rlink/', defaults={'name': ''})
@app.route('/rlink/<path:name>')
def get_proxy(name):
    if name == 'list':
        return ls()
    else:
        glog.info('getting proxied request ' + name)
        return get(name)


@app.route('/', defaults={'name': ''})
@app.route('/<path:name>')
def get(name):
    glog.info('fetching ' + name)
    abs_path = os.path.abspath(FLAGS.dir)
    file_path = os.path.join(abs_path, name)

    try:
        with open(file_path) as f:
            content = f.read().strip()
        glog.info('redirecting to ' + content)
        return redirect(content)
    except Exception as e:
        glog.error(e)

    glog.info('not found')
    abort(404)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
