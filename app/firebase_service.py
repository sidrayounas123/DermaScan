import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import os
import json

# Initialize Firebase
if os.environ.get("FIREBASE_CREDENTIALS"):
    # Use environment variable for deployment
    firebase_creds = json.loads(os.environ.get("FIREBASE_CREDENTIALS"))
    cred = credentials.Certificate(firebase_creds)
else:
    # No Firebase credentials available - skip Firebase initialization
    print("Firebase credentials not found. Firebase features will be disabled.")
    cred = None

if cred and not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Initialize database only if Firebase is available
db = firestore.client() if cred else None

def save_scan(user_id, disease, confidence, severity, see_doctor, dataset):
    if not db:
        print("Firebase not available - scan not saved")
        return False
    
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

def get_scans(user_id, filter_type=None):
    try:
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        this_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        this_week_start = now - datetime.timedelta(days=7)
        
        scans_ref = db.collection("scans").where("user_id", "==", user_id)
        scans = scans_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        
        result = []
        total = 0
        this_month = 0
        this_week = 0
        
        for scan in scans:
            data = scan.to_dict()
            data["id"] = scan.id
            timestamp = data.get("timestamp")
            if timestamp:
    try:
        if hasattr(timestamp, 'tzinfo'):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)
        data["timestamp"] = timestamp.isoformat()
        if timestamp >= this_month_start:
            this_month += 1
        if timestamp >= this_week_start:
            this_week += 1
    except Exception as ts_error:
        print(f"Timestamp error: {ts_error}")
        data["timestamp"] = str(timestamp)
            is_high_risk = data.get("severity") == "Severe" or data.get("see_doctor") == True
            data["is_high_risk"] = is_high_risk
            total += 1
            result.append(data)
        
        if filter_type == "high_risk":
            result = [s for s in result if s.get("is_high_risk") == True]
        elif filter_type == "low_risk":
            result = [s for s in result if s.get("is_high_risk") == False]
        
        return {"total": total, "this_month": this_month, "this_week": this_week, "scans": result}
    except Exception as e:
        print(f"Error retrieving scans: {str(e)}")
        return {"total": 0, "this_month": 0, "this_week": 0, "scans": []}
