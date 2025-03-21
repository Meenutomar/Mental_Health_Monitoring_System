from fastapi import FastAPI, HTTPException, UploadFile, File, APIRouter, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import io
import tempfile

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL or Key not found in environment variables")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Define FastAPI router
router = APIRouter()

# Define Pydantic Model for User Profile
class UserProfile(BaseModel):
    email: str
    name: str
    age: int
    phone: str
    address: str
    city: str
    state: str
    country: str
    zip_code: str
    profile_pic_url: str = None

# 🚀 1. Get User Profile
@router.get("/profile/{email}")
async def get_profile(email: str, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")

    token = authorization.split("Bearer ")[-1]

    # Set the Supabase session BEFORE any action
    supabase.auth.set_session(access_token=token, refresh_token=token)

    # Optional: Validate the token really belongs to this email
    user = supabase.auth.get_user()
    if not user or user.user.email != email:
        raise HTTPException(status_code=403, detail="Access denied")

    # Proceed to fetch profile from Supabase DB
    try:
        result = supabase.table("UserProfile").select("*").eq("email", email).execute()
        if not result.data:
            return {"status": "not_found", "message": "Profile not found"}
        
        profile = result.data[0]
        print('Profile:', profile)
        image_path = f"/{email}.png"
        # Generate signed URL valid for, say, 1 day
        signed_url_response = supabase.storage.from_("profilepictures").create_signed_url(image_path, expires_in=86400)
        print('Signed URL', signed_url_response)
        profile["profile_pic_url"] = signed_url_response.get("signedURL")
        print("Fetched Profile::" , profile)
        return profile
    except Exception as e:
        print('Ëxception::',e)
        raise HTTPException(status_code=500, detail=str(e))

# 🚀 2. Save or Update User Profile
@router.post("/profile")
def save_profile(profile: UserProfile, authorization: str = Header(None)):
    try:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization header missing or invalid")

        token = authorization.split("Bearer ")[-1]

        # Set session for auth (note: Supabase Python SDK only stores this for future use)
        supabase.auth.set_session(access_token=token, refresh_token=token)

        # Check if profile exists
        existing = supabase.table("UserProfile").select("*").eq("email", profile.email).execute()

        profile_data = profile.dict()

        if existing.data:
            response = supabase.table("UserProfile").update(profile_data).eq("email", profile.email).execute()
        else:
            response = supabase.table("UserProfile").insert(profile_data).execute()

        return {"status": "success", "message": "Profile saved successfully", "profile": response.data[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save Profile failed: {str(e)}")

# 🚀 3. Upload Profile Picture


@router.post("/upload_image/{file_name}")
async def upload_image(file_name: str, file: UploadFile = File(...), authorization: str = Header(None)):
    try:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization header missing or invalid")

        token = authorization.split("Bearer ")[-1]

        # ✅ Set Supabase session before performing auth-related actions
        supabase.auth.set_session(access_token=token, refresh_token=token)

        # ✅ (Optional Debug) Print current user after setting session
        print("Current Supabase user:", supabase.auth.get_user())

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Upload to Supabase Storage
        supabase.storage.from_("profilepictures").upload(file_name, tmp_path, {
            "content-type": file.content_type,
            "x-upsert": "true"
        })

        # Public URL
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/profilepictures/{file_name}"
        return {"status": "success", "url": public_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")



