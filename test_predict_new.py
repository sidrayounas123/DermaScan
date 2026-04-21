import urllib.request
import json
import io
from PIL import Image

# Create a simple test image
img = Image.new('RGB', (224, 224), color='red')
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

# Create multipart form data
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
data = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
    f'Content-Type: image/jpeg\r\n\r\n'
).encode('utf-8')
data += img_bytes.read()
data += f'\r\n--{boundary}--\r\n'.encode('utf-8')

# Make request
req = urllib.request.Request(
    'http://localhost:8000/predict/dataset1',
    data=data,
    headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'Content-Length': str(len(data))
    }
)

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        print('Prediction Response:')
        print(json.dumps(result, indent=2))
except Exception as e:
    print('Error:', e)
