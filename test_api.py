import requests
import cv2
import numpy as np
import io

def test_api():
    # Create a dummy black image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    url = "http://127.0.0.1:8000/measure"
    files = {'image': ('test.jpg', img_bytes, 'image/jpeg')}
    data = {'height': '170'}

    try:
        response = requests.post(url, files=files, data=data)
        print(f"Status Code: {response.status_code}")
        json_resp = response.json()
        if "image_base64" in json_resp:
            print("Success: image_base64 field present.")
        else:
            print("Response:", json_resp)
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    test_api()
