import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import os
import json

# Initialize Firebase
if os.environ.get("FIREBASE_CREDENTIALS"):
    # Use environment variable for Render deployment
    firebase_creds = json.loads(os.environ.get("FIREBASE_CREDENTIALS"))
    cred = credentials.Certificate(firebase_creds)
else:
    # Use local file for development
    cred = credentials.Certificate("serviceAccountKey.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

def save_scan(user_id, disease, confidence, severity, see_doctor, dataset):
    doc = {
        "user_id": user_id,
        "disease": disease,
        "confidence": confidence,
        "severity": severity,
        "see_doctor": see_doctor,
        "dataset": dataset,
        "timestamp": datetime.datetime.now()
    }
    db.collection("scans").add(doc)
    return True

def get_scans(user_id):
    scans = db.collection("scans")\
              .where("user_id", "==", user_id)\
              .order_by("timestamp", direction=firestore.Query.DESCENDING)\
              .limit(20)\
              .stream()
    result = []
    for scan in scans:
        data = scan.to_dict()
        data["id"] = scan.id
        if "timestamp" in data:
            data["timestamp"] = data["timestamp"].isoformat()
        result.append(data)
    return result
