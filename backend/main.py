from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chatbot
import videostream
import audiostream
import audiojs

# Initialize FastAPI app
app = FastAPI()

# Allow requests from any origin (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Allow requests from all origins (can be restricted later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(chatbot.router)
app.include_router(videostream.router)
app.include_router(audiostream.router)
app.include_router(audiojs.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
