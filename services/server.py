from flask import Flask, request, jsonify

# curl -X POST -H "Content-Type: application/json" -d '{"input_integer": 42}' http://127.0.0.1:4200/echo

app = Flask(__name__)

@app.route('/echo', methods=['POST'])
def echo_integer():
    data = request.get_json()
    if 'input_integer' not in data:
        return jsonify({"error": "No input_integer provided"}), 400
    
    input_integer = data['input_integer']
    
    if not isinstance(input_integer, int):
        return jsonify({"error": "input_integer must be an integer"}), 400
    
    return jsonify({"input_integer": input_integer})

if __name__ == '__main__':
    app.run(debug=True, port=4200)