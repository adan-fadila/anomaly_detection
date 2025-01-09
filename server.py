# from flask import Flask
# from app import create_app

# # Initialize Flask app
# app = create_app()

# if __name__ == '__main__':
#     app.run(debug=False)

# from flask import Flask, jsonify

# app = Flask(__name__)

# @app.route('/receive-data', methods=['GET'])
# def receive_data():
#     return jsonify({'message': 'Data received successfully from Raspberry Pi!'})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


from flask import Flask, request, jsonify
import csv
import os

app = Flask(__name__)

# Define the path to the CSV file in the root directory
CSV_FILE_PATH = os.path.join(os.getcwd(), "logs.csv")

# Ensure the CSV file exists and has a header row
if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "temperature"])  # Add header row

@app.route('/log-data', methods=['POST'])
def log_data():
    try:
        # Parse the incoming JSON data
        data = request.json
        timestamp = data.get("timestamp")
        temperature = data.get("temperature")

        # Append the data to the CSV file
        with open(CSV_FILE_PATH, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, temperature])

        return jsonify({"status": "success", "message": "Data logged successfully."}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

