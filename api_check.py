import requests
import base64


# Define the URL for the prediction endpoint
url = 'http://127.0.0.1:5000/predict'

# Read the image file and encode it to base64
with open(r'ng.bmp', 'rb') as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

# Create the JSON payload with the base64 encoded image
payload = {
    'image': image_base64
}

# Send the POST request with the JSON payload
response = requests.post(url, json=payload)

# Print the response
if response.status_code == 200:
    #print(response.json())
     print(" Inspection successfull...!")
else:
    print(f"Error: {response.status_code}")

