from flask import Flask, request, jsonify
from flask_caching import Cache

import statsd

sd = statsd.StatsClient('localhost', 8125)

# curl -X POST -H "Content-Type: application/json" -d '{"input_integer": 42}' http://127.0.0.1:4200/echo

# Run with gunicorn:
# gunicorn --workers 4 --bind 0.0.0.0:4200 app:app

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/echo', methods=['POST'])
# @cache.cached(timeout=60)
@cache.cached()
def echo_integer():
    data = request.get_json()
    if 'input_integer' not in data:
        return jsonify({"error": "No input_integer provided"}), 400
    
    input_integer = data['input_integer']
    
    if not isinstance(input_integer, int):
        return jsonify({"error": "input_integer must be an integer"}), 400

    sd.incr('web.test.echo_integer.py')
    
    return jsonify({"input_integer": input_integer})

if __name__ == '__main__':
    app.run(debug=True, port=4200)