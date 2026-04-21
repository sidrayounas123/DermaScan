import firebase_admin
from firebase_admin import credentials, firestore, auth
import requests
import datetime

# Firebase Web API Key
FIREBASE_WEB_API_KEY = "AIzaSyCqC7fkcWhLLdMlb080P7vQHwe3t-1YSSs"

# Initialize Firebase (using same initialization from firebase_service)
db = firestore.client()

def register_user(name, email, password):
    """
    Creates user in Firebase Auth and saves name+email to Firestore
    Returns user uid on success
    """
    try:
        # Create user in Firebase Auth
        user = auth.create_user(
            email=email,
            password=password
        )
        
        # Save user data to Firestore
        user_data = {
            "name": name,
            "email": email,
            "created_at": datetime.datetime.now()
        }
        
        db.collection("users").document(user.uid).set(user_data)
        
        return user.uid
        
    except Exception as e:
        raise Exception(f"Registration failed: {str(e)}")

def login_user(email, password):
    """
    Verifies user credentials using Firebase Auth REST API
    Returns user token and uid on success
    """
    try:
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
        
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "token": data["idToken"],
                "uid": data["localId"]
            }
        else:
            error_data = response.json()
            raise Exception(f"Login failed: {error_data.get('error', {}).get('message', 'Unknown error')}")
            
    except Exception as e:
        raise Exception(f"Login failed: {str(e)}")

def get_user_profile(uid):
    """
    Gets user data from Firestore "users" collection
    Returns {name, email}
    """
    try:
        doc_ref = db.collection("users").document(uid)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            return {
                "name": data.get("name"),
                "email": data.get("email")
            }
        else:
            raise Exception("User not found")
            
    except Exception as e:
        raise Exception(f"Failed to get user profile: {str(e)}")
