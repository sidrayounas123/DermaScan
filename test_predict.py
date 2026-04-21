import urllib.request
import json
import io
from PIL import Image

# Create a simple test image
img = Image.new('RGB', (224, 224), color='red')
img_bytes = io.BytesIO()
img.save(img_bytes, format='PNG')
img_bytes = img_bytes.getvalue()

# Create multipart form data
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = '--' + boundary + '\r\n'
body += 'Content-Disposition: form-data; name="file"; filename="test.png"\r\n'
body += 'Content-Type: image/png\r\n\r\n'
body = body.encode('utf-8')
body += img_bytes
body += ('\r\n--' + boundary + '--\r\n').encode('utf-8')

# Make POST request
req = urllib.request.Request(
    'http://localhost:8000/predict/dataset1',
    data=body,
    headers={
        'Content-Type': 'multipart/form-data; boundary=' + boundary,
        'Content-Length': str(len(body))
    }
)

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        print('Prediction response:')
        print(json.dumps(result, indent=2))
except Exception as e:
    print('Error making prediction:', e)
