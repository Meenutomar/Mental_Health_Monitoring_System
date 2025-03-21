from fastapi import FastAPI, HTTPException, UploadFile, File, APIRouter, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import io

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
def get_profile(email: str):
    print('Get Profile::', email);
    response = supabase.table("UserProfile").select("*").eq("email", email).execute()
    print('Response:', response)
    if response.data:
        return {"status": "success", "profile": response.data[0]}
    raise HTTPException(status_code=404, detail="Profile not found")

# 🚀 2. Save or Update User Profile
@router.post("/profile")
def save_profile(profile: UserProfile, authorization: str = Header(None)):
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

# 🚀 3. Upload Profile Picture
@router.post("/upload_image/{file_name}")
async def upload_image(file_name: str, file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        file_io = io.BytesIO(file_bytes)

        # Upload to Supabase Storage
        supabase.storage.from_("profilepictures").upload(file_name, file_io, {"content-type": file.content_type})

        # Public URL
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/profilepictures/{file_name}"
        return {"status": "success", "url": public_url}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


