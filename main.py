from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
import httpx
import asyncio
import json
from datetime import datetime
import os
from typing import Optional, Dict, Any
import firebase_admin
from firebase_admin import credentials, firestore
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()
# Initialize FastAPI app
app = FastAPI(title="Financial Assistant API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    message: str
    uid: str

class ChatResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    success: bool
    response: str
    timestamp: str
    processing_time: Optional[float] = None

# Initialize Firebase (do this once at startup)
db = None
try:
    # Method 1: Using base64 encoded service account (for production/Render)
    if os.getenv('FIREBASE_SERVICE_ACCOUNT'):
        import base64
        service_account_info = json.loads(base64.b64decode(os.getenv('FIREBASE_SERVICE_ACCOUNT')).decode())
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase initialized with environment variable")
    
    # Method 2: Using service account file (for local development)
    elif os.path.exists("serviceAccountKey.json"):
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase initialized with local service account file")
    
    # Method 3: Using Google Application Default Credentials
    elif os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        firebase_admin.initialize_app()
        db = firestore.client()
        print("✅ Firebase initialized with default credentials")
    
    else:
        print("❌ No Firebase credentials found. Please set up one of the following:")
        print("   1. FIREBASE_SERVICE_ACCOUNT environment variable (base64 encoded)")
        print("   2. serviceAccountKey.json file in the project root")
        print("   3. GOOGLE_APPLICATION_CREDENTIALS environment variable")
        
except Exception as e:
    print(f"❌ Firebase initialization error: {e}")
    print("The API will run without Firebase. User data requests will fail.")
    db = None

# Global HTTP client for reuse (important for performance)
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(30.0),
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
)

# Cache for user data (expires after 5 minutes)
@lru_cache(maxsize=1000)
def get_cached_user_data(uid: str, cache_key: str):
    """Cache user data with timestamp-based invalidation"""
    return None  # This will be implemented with actual caching logic

# Ultra-concise prompt to minimize token usage
PROMPT_TEMPLATE = """
You are a smart financial assistant.

User's financial summary:
{user_data}

Question:
{message}

Based on the financial summary, answer the question clearly with useful insights. Keep it concise but informative. and user currency is INR
"""

async def fetch_user_data_async(uid: str) -> Dict[str, Any]:
    """Async function to fetch user data from Firestore"""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        print(f"🔍 Fetching data for user ID: {uid}")
        # Use async Firestore operations if available
        user_ref = db.collection('users').document(uid)
        doc = user_ref.get()
        
        if not doc.exists:
            print(f"❌ User {uid} not found in Firestore")
            raise HTTPException(status_code=404, detail=f"User {uid} not found")
        
        user_data = doc.to_dict()
        user_bills = user_data.get('user_bills', [])
        print(f"✅ Found user data with {len(user_bills)} bills")
        return user_bills
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Firebase error: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Error fetching user data: {str(e)}")

async def call_gemini_async(prompt: str) -> str:
    """Async function to call Gemini API with optimizations"""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    print("api key:", gemini_api_key)
    print("api_key:", gemini_api_key)
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_api_key}"
    
    # Optimized payload - increase token limit further
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 2000,  # Increased from 1000 to 2000
            "temperature": 0.7,
            "topP": 0.8,
            "topK": 40
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
    }
    
    try:
        print(f"🌐 Sending request to Gemini API...")
        print(f"📏 Prompt length: {len(prompt)} characters")
        print(f"🔑 API Key present: {'Yes' if gemini_api_key else 'No'}")
        
        response = await http_client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code != 200:
            error_text = response.text
            print(f"❌ Gemini API Error Response: {error_text}")
            raise HTTPException(status_code=500, detail=f"Gemini API error: {error_text}")
        
        data = response.json()
        print(f"📦 Full Gemini response: {json.dumps(data, indent=2)}")
        
        # Check if response was blocked
        if 'candidates' not in data or not data['candidates']:
            print("❌ No candidates in response - possibly blocked by safety filters")
            return "I apologize, but I cannot process this request. Please try rephrasing your question about your spending data."
        
        candidate = data['candidates'][0]
        
        # Check for safety blocking
        if candidate.get('finishReason') == 'SAFETY':
            print("❌ Response blocked by safety filters")
            return "I apologize, but I cannot process this request due to safety restrictions. Please try asking about your spending in a different way."
        
        # Check for MAX_TOKENS finish reason and handle gracefully
        if candidate.get('finishReason') == 'MAX_TOKENS':
            print("⚠️ Response was truncated due to token limit")
            # Get whatever text we have and add a continuation note
            content = candidate.get('content', {})
            parts = content.get('parts', [])
            partial_text = parts[0].get('text', '') if parts else ''
            
            if partial_text.strip():
                return partial_text.strip() + "... [Response truncated - please ask a more specific question for a complete answer]"
            else:
                return "Response was truncated. Please ask a more specific question about your spending (e.g., 'How much did I spend yesterday?' or 'What did I buy at V MART?')"
        
        # Extract the actual response text
        content = candidate.get('content', {})
        parts = content.get('parts', [])
        
        if not parts or not parts[0].get('text'):
            print("❌ No text content in response parts")
            print(f"📦 Parts structure: {json.dumps(parts, indent=2)}")
            return "I received an empty response. Please try asking about your spending data again."
        
        reply = parts[0]['text'].strip()
        print(f"✅ Extracted reply: {len(reply)} characters")
        print(f"💬 Reply preview: {reply[:100]}...")
        
        return reply
        
    except httpx.TimeoutException:
        print("⏰ Request timed out")
        raise HTTPException(status_code=408, detail="Request timeout - try again")
    except httpx.HTTPStatusError as e:
        print(f"🌐 HTTP error: {e}")
        raise HTTPException(status_code=500, detail=f"HTTP error: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error calling Gemini: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error calling Gemini: {str(e)}")

def process_user_bills(user_bills: list) -> str:
    """Optimized processing of user bills data - minimal data for token efficiency"""
    print(f"🔍 Processing {len(user_bills)} user bills...")
    
    if not user_bills:
        print("❌ No user bills provided")
        return "[]"
    
    # Extract only essential data for AI processing
    processed_bills = []
    seen_bills = set()
    
    for i, bill in enumerate(user_bills):
        if not bill or not bill.get('json'):
            continue
            
        bill_json = bill['json']
        
        # Extract minimal essential data
        merchant = bill_json.get('merchant', '')[:30]  # Truncate long names
        total_amount = bill_json.get('total_amount', bill_json.get('tender_amount', ''))
        date = bill_json.get('time_stamp', '')[:10]  # Just date, not time
        
        # Simplified category inference
        items = bill_json.get('items', [])
        if any('food' in item.get('name', '').lower() or 'chocolate' in item.get('name', '').lower() 
               or 'ice' in item.get('name', '').lower() for item in items):
            category = 'Food'
        elif any('comb' in item.get('name', '').lower() or 'cosmetic' in item.get('name', '').lower()
                 for item in items):
            category = 'Personal Care'
        else:
            category = 'Shopping'
        
        # Create simple hash for duplicates
        bill_hash = f"{merchant}_{total_amount}_{date}"
        
        if bill_hash not in seen_bills:
            seen_bills.add(bill_hash)
            
            # Minimal data structure
            essential_data = {
                'merchant': merchant,
                'amount': float(total_amount) if total_amount else 0,
                'date': date,
                'category': category,
                'items': len(items)
            }
            
            processed_bills.append(essential_data)
            print(f"✅ Added: ₹{total_amount} at {merchant[:20]}")
    
    # Create compact JSON
    result = json.dumps(processed_bills, separators=(',', ':'))  # No spaces for compactness
    print(f"📋 Processed {len(processed_bills)} bills, {len(result)} chars")
    
    return result

@app.post("/test-gemini")
async def test_gemini():
    """Test endpoint to check if Gemini API is working"""
    try:
        simple_prompt = "Hello, please respond with 'API is working' if you can see this message."
        response = await call_gemini_async(simple_prompt)
        return {
            "success": True,
            "response": response,
            "message": "Gemini API test successful"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Gemini API test failed"
        }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Optimized chat endpoint with parallel processing"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        print(f"🔍 Processing request for user: {request.uid}")
        print(f"💬 Message: {request.message}")
        
        # Validate input
        if not request.message or not request.uid:
            raise HTTPException(status_code=400, detail="Message and UID are required")
        
        # Check if database is available
        if not db:
            raise HTTPException(status_code=500, detail="Database not initialized. Check Firebase configuration.")
        
        print("📊 Fetching user data from Firebase...")
        # Fetch user data asynchronously
        user_bills = await fetch_user_data_async(request.uid)
        print(f"✅ Found {len(user_bills)} bills for user")
        
        # Process bills data efficiently - limit to recent bills to save tokens
        print("⚙️ Processing user bills...")
        processed_bills = process_user_bills(user_bills)
        
        # If data is too large, limit to most recent bills
        if len(processed_bills) > 500:  # If JSON is > 500 chars
            print("⚠️ Data too large, limiting to recent bills...")
            bills_data = json.loads(processed_bills)
            # Sort by date and take only the 3 most recent
            bills_data.sort(key=lambda x: x.get('date', ''), reverse=True)
            limited_bills = bills_data[:3]
            processed_bills = json.dumps(limited_bills, separators=(',', ':'))
            print(f"✅ Limited to {len(limited_bills)} recent bills: {len(processed_bills)} chars")
        
        print(f"✅ Processed bills data: {len(processed_bills)} characters")
        
        # Create optimized prompt
        prompt = PROMPT_TEMPLATE.format(
            user_data=processed_bills,
            message=request.message
        )
        print(f"📝 Created prompt with {len(prompt)} characters")
        
        # Call Gemini API asynchronously
        print("🤖 Calling Gemini API...")
        ai_response = await call_gemini_async(prompt)
        print(f"✅ Received Gemini response: {len(ai_response)} characters")
        
        # Calculate processing time
        end_time = asyncio.get_event_loop().time()
        processing_time = round((end_time - start_time) * 1000, 2)  # Convert to milliseconds
        
        print(f"⚡ Total processing time: {processing_time}ms")
        
        return ChatResponse(
            success=True,
            response=ai_response,
            timestamp=datetime.utcnow().isoformat(),
            processing_time=processing_time
        )
        
    except HTTPException as e:
        print(f"❌ HTTP Exception: {e.detail}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "OK",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Financial Assistant API"
    }

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    print("🚀 FastAPI Financial Assistant started")
    print("✅ HTTP client initialized")
    if db:
        print("✅ Firebase connected")
    else:
        print("❌ Firebase connection failed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    await http_client.aclose()
    print("🛑 FastAPI Financial Assistant stopped")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Financial Assistant API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,  # Disable reload in production
        workers=1  # Single worker for free tier
    )
