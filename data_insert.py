import mysql.connector
from mysql.connector import Error

def create_connection():
    """Create a database connection."""
    connection = None
    try:
        connection = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            port='3306',
            password='12345',
            database='parleg'
        )
        if connection.is_connected():
            print("MYSQL Database Connecting .......!")
            print("Successfully connected to the database")
    except Error as e:
        print(f"Error: {e}")
    return connection

def create_table():
    """Create the images table if it doesn't exist."""
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        try:
            query = """
            CREATE TABLE IF NOT EXISTS images (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image LONGBLOB,
                status VARCHAR(255),
                count INT,
                lab_list TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(query)
            connection.commit()
            print("Table 'images' is ready.")
        except Error as e:
            print(f"Error: {e}")
        finally:
            cursor.close()
            connection.close()

def insert_image(image_data, status, packet_count, lab_list):
    """Insert an image with additional data into the table."""
    connection = create_connection()
    if connection:
        cursor = connection.cursor()

        try:
            query = """
            INSERT INTO images (image, status, count, lab_list)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (image_data, status, packet_count, lab_list))
            connection.commit()
            print("Image inserted successfully.")
        except Error as e:
            print(f"Error: {e}")
        finally:
            cursor.close()
            connection.close()
    else: 
        print("Unable to connect to the database.")

# Create table if it does not exist
create_table()
