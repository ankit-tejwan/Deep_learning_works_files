from flask import Flask, Response, jsonify, render_template
import mysql.connector
from mysql.connector import Error
import base64

app = Flask(__name__)

# MySQL connection function
from flask import jsonify

def create_connection():
    """Create a connection to the MySQL database."""
    connection = None
    try:
        connection = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='12345',
            database='parleg'  # The database where your images table is stored
        )
        if connection.is_connected():
            print("Successfully connected to the MySQL database")
    except Error as e:
        print(f"Error: {e}")
        raise  # Re-raise the error after logging it
    return connection

@app.route('/')
def home():
    try:
        connection = create_connection()
        if connection:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM images ORDER BY timestamp DESC LIMIT 1")
            image_document = cursor.fetchone()

            if image_document:
                image_data = image_document.get('image')
                if image_data:
                    image_data = base64.b64encode(image_data).decode('utf-8')  # Convert binary to base64 string

                image_id = image_document.get('id')
                status = image_document.get('status', 'No status available')
                count = image_document.get('count', 0)
                lab_list = image_document.get('lab_list', [])

                return render_template('index.html', image_id=image_id, 
                                       image_data=image_data, status=status, 
                                       count=count, lab_list=lab_list)
            else:
                return jsonify({"error": "No image found in the database."}), 404
    except Error as e:
        print(f"Error: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
    finally:
        if connection:
            connection.close()


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5001)
