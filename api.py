
## for the Flask Rest API

from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data
tasks = [{"id": 1, "title": "Learn Flask", "done": False}]

@app.route('/tasks', methods=['GET'])
def get_tasks():
    return jsonify({"tasks": tasks})

@app.route('/tasks', methods=['POST'])
def add_task():
    new_task = request.get_json()
    tasks.append(new_task)
    return jsonify(new_task), 201

if __name__ == '__main__':
    app.run(debug=True)
