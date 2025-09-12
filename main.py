import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import hashlib
import uuid

# Core libraries
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# AI and Vector Database
import openai
from pinecone import Pinecone, ServerlessSpec
import tiktoken

# Database for conversation history
import sqlite3
from contextlib import contextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cancer_chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # API Keys - Set these as environment variables
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_zRyjS_2FyS6uk3NsKW9AHPzDvvQPzANF2S3B67MS6UZ7ax6tnJfmCbLiYXrEcBJFHzcHg")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB3N9BHeIWs_8sdFK76PU-v9N6prcIq2Hw")
    GREEN_API_ID_INSTANCE = os.getenv("GREEN_API_ID_INSTANCE", "7105287498")
    GREEN_API_TOKEN = os.getenv("GREEN_API_TOKEN", "0017430b3b204cf28ac14a41cc5ede0ce8e5a68d91134d5fbe")

    #OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    #PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    #GREEN_API_INSTANCE_ID = os.getenv("GREEN_API_INSTANCE_ID")
    #GREEN_API_ACCESS_TOKEN = os.getenv("GREEN_API_ACCESS_TOKEN")
    
    # Pinecone settings
    PINECONE_INDEX_NAME = "cancel"
    PINECONE_DIMENSION = 1536  # OpenAI embedding dimension
    
    # OpenAI settings
    OPENAI_MODEL = "gpt-4"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    
    # Conversation settings
    MAX_CONVERSATION_LENGTH = 10  # Keep last 10 exchanges
    MAX_CONTEXT_TOKENS = 3000
    
    # Database
    DB_PATH = "conversations.db"

# Pydantic models
class WhatsAppMessage(BaseModel):
    messageData: dict
    idMessage: str
    timestamp: int
    typeMessage: str
    chatId: str
    senderId: str
    senderName: str

class ConversationEntry(BaseModel):
    user_id: str
    message: str
    response: str
    timestamp: datetime

# Database manager
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize the SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    session_id TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    phone_number TEXT,
                    first_interaction DATETIME,
                    last_interaction DATETIME,
                    total_messages INTEGER DEFAULT 0
                )
            ''')
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def save_conversation(self, user_id: str, message: str, response: str, session_id: str = None):
        """Save a conversation exchange"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations (user_id, message, response, timestamp, session_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, message, response, datetime.now(), session_id))
            conn.commit()
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation history for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT message, response, timestamp
                FROM conversations
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
            
            rows = cursor.fetchall()
            return [
                {
                    'user_message': row[0],
                    'bot_response': row[1],
                    'timestamp': row[2]
                }
                for row in reversed(rows)  # Reverse to get chronological order
            ]
    
    def update_user_profile(self, user_id: str, phone_number: str):
        """Update user profile information"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO user_profiles 
                (user_id, phone_number, first_interaction, last_interaction, total_messages)
                VALUES (
                    ?, ?, 
                    COALESCE((SELECT first_interaction FROM user_profiles WHERE user_id = ?), ?),
                    ?, 
                    COALESCE((SELECT total_messages FROM user_profiles WHERE user_id = ?), 0) + 1
                )
            ''', (user_id, phone_number, user_id, datetime.now(), datetime.now(), user_id))
            conn.commit()

# Vector Database Manager
class VectorDBManager:
    def __init__(self):
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index_name = Config.PINECONE_INDEX_NAME
        self.dimension = Config.PINECONE_DIMENSION
        self.init_index()
        
    def init_index(self):
        """Initialize Pinecone index"""
        try:
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate OpenAI embedding for text"""
        try:
            response = openai.embeddings.create(
                input=text,
                model=Config.EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def upsert_document(self, doc_id: str, text: str, metadata: Dict):
        """Add or update a document in the vector database"""
        try:
            embedding = self.generate_embedding(text)
            self.index.upsert(
                vectors=[(doc_id, embedding, metadata)]
            )
            logger.info(f"Upserted document {doc_id}")
        except Exception as e:
            logger.error(f"Error upserting document {doc_id}: {e}")
            raise
    
    def search_similar(self, query: str, top_k: int = 5, filter_dict: Dict = None) -> List[Dict]:
        """Search for similar documents"""
        try:
            query_embedding = self.generate_embedding(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            return [
                {
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                }
                for match in results.matches
            ]
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []

# WhatsApp Integration
class WhatsAppManager:
    def __init__(self):
        self.instance_id = Config.GREEN_API_INSTANCE_ID
        self.access_token = Config.GREEN_API_ACCESS_TOKEN
        self.base_url = f"https://api.green-api.com/waInstance{self.instance_id}"
    
    def send_message(self, chat_id: str, message: str) -> bool:
        """Send a message via WhatsApp"""
        try:
            url = f"{self.base_url}/sendMessage/{self.access_token}"
            payload = {
                "chatId": chat_id,
                "message": message
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"Message sent successfully to {chat_id}")
                return True
            else:
                logger.error(f"Failed to send message: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {e}")
            return False
    
    def send_typing(self, chat_id: str):
        """Send typing indicator"""
        try:
            url = f"{self.base_url}/sendChatStateTyping/{self.access_token}"
            payload = {"chatId": chat_id}
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            logger.error(f"Error sending typing indicator: {e}")

# AI Response Generator
class AIResponseGenerator:
    def __init__(self, vector_db: VectorDBManager, db_manager: DatabaseManager):
        self.vector_db = vector_db
        self.db_manager = db_manager
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # System prompt for the cancer support chatbot
        self.system_prompt = """You are a compassionate AI assistant specifically designed to support cancer patients and their families. Your role is to provide helpful, accurate, and empathetic responses while being mindful of the emotional and physical challenges they face.

IMPORTANT GUIDELINES:
1. You are NOT a replacement for medical professionals
2. Always encourage users to consult with their healthcare team for medical decisions
3. Provide emotional support and practical information
4. Be empathetic, patient, and understanding
5. If you don't know something, admit it and suggest consulting medical professionals
6. Never provide specific medical diagnoses or treatment recommendations
7. Focus on general education, support, and guidance

You can help with:
- General information about cancer types and treatments
- Side effect management tips
- Emotional support and coping strategies
- Nutrition and lifestyle guidance
- Healthcare navigation
- Questions about what to expect during treatment

Always maintain a warm, supportive tone while being informative and helpful."""
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def build_context_from_history(self, conversation_history: List[Dict], max_tokens: int = 1000) -> str:
        """Build context from conversation history within token limit"""
        if not conversation_history:
            return ""
        
        context_parts = []
        current_tokens = 0
        
        # Start from most recent conversations
        for conv in reversed(conversation_history):
            conv_text = f"User: {conv['user_message']}\nAssistant: {conv['bot_response']}\n"
            conv_tokens = self.count_tokens(conv_text)
            
            if current_tokens + conv_tokens > max_tokens:
                break
                
            context_parts.insert(0, conv_text)
            current_tokens += conv_tokens
        
        if context_parts:
            return "Previous conversation:\n" + "\n".join(context_parts) + "\n---\n"
        return ""
    
    def generate_response(self, user_message: str, user_id: str) -> str:
        """Generate AI response using RAG and conversation history"""
        try:
            # Get conversation history
            conversation_history = self.db_manager.get_conversation_history(
                user_id, limit=Config.MAX_CONVERSATION_LENGTH
            )
            
            # Build context from history
            history_context = self.build_context_from_history(conversation_history, max_tokens=800)
            
            # Search for relevant documents in vector database
            relevant_docs = self.vector_db.search_similar(user_message, top_k=3)
            
            # Build RAG context
            rag_context = ""
            if relevant_docs:
                rag_context = "Relevant information from medical resources:\n"
                for doc in relevant_docs:
                    if doc['score'] > 0.7:  # Only include highly relevant documents
                        content = doc['metadata'].get('content', '')
                        source = doc['metadata'].get('source', 'Medical Resource')
                        rag_context += f"From {source}: {content[:500]}...\n\n"
            
            # Construct the full prompt
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{history_context}{rag_context}Current question: {user_message}"}
            ]
            
            # Generate response using OpenAI
            response = openai.chat.completions.create(
                model=Config.GOOGLE_API_KEY,
                messages=messages,
                max_tokens=800,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Add disclaimer for medical advice
            if any(keyword in user_message.lower() for keyword in ['diagnosis', 'treatment', 'medication', 'dosage', 'should i']):
                ai_response += "\n\n⚠️ Please remember to discuss any medical concerns with your healthcare team. This information is for educational purposes only."
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return ("I apologize, but I'm having trouble processing your message right now. "
                   "Please try again in a moment, or contact your healthcare provider if this is urgent.")

# Main Chatbot Class
class CancerSupportChatbot:
    def __init__(self):
        # Initialize components
        self.db_manager = DatabaseManager(Config.DB_PATH)
        self.vector_db = VectorDBManager()
        self.whatsapp = WhatsAppManager()
        self.ai_generator = AIResponseGenerator(self.vector_db, self.db_manager)
        
        # Initialize OpenAI
        openai.api_key = Config.GOOGLE_API_KEY
        
        logger.info("Cancer Support Chatbot initialized successfully")
    
    async def process_message(self, chat_id: str, user_id: str, user_message: str, sender_name: str = ""):
        """Process incoming message and generate response"""
        try:
            logger.info(f"Processing message from {user_id}: {user_message[:100]}...")
            
            # Send typing indicator
            self.whatsapp.send_typing(chat_id)
            
            # Update user profile
            self.db_manager.update_user_profile(user_id, chat_id)
            
            # Handle special commands
            if user_message.lower().strip() in ['/start', 'hello', 'hi', 'start']:
                response = self.get_welcome_message(sender_name)
            elif user_message.lower().strip() in ['/help', 'help']:
                response = self.get_help_message()
            elif user_message.lower().strip() in ['/emergency', 'emergency']:
                response = self.get_emergency_message()
            else:
                # Generate AI response
                response = self.ai_generator.generate_response(user_message, user_id)
            
            # Send response
            success = self.whatsapp.send_message(chat_id, response)
            
            if success:
                # Save conversation to database
                self.db_manager.save_conversation(user_id, user_message, response)
                logger.info(f"Successfully processed message for {user_id}")
            else:
                logger.error(f"Failed to send response to {user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Send error message
            error_response = ("I apologize, but I encountered an error processing your message. "
                            "Please try again or contact your healthcare provider if this is urgent.")
            self.whatsapp.send_message(chat_id, error_response)
            return False
    
    def get_welcome_message(self, sender_name: str = "") -> str:
        """Get welcome message for new users"""
        name_part = f"Hello {sender_name}! " if sender_name else "Hello! "
        return f"""{name_part}I'm your Cancer Support Assistant. I'm here to help answer questions about cancer treatment, side effects, and provide emotional support.

I can help with:
🩺 General cancer information
💊 Treatment side effects
🥗 Nutrition guidance
💪 Coping strategies
📋 Healthcare navigation

Type 'help' for more information or 'emergency' for urgent resources.

⚠️ Important: I'm not a replacement for medical care. Always consult your healthcare team for medical decisions."""

    def get_help_message(self) -> str:
        """Get help message"""
        return """Here's how I can help you:

📚 **Ask me about:**
• Cancer types and treatments
• Managing side effects (nausea, fatigue, pain)
• Nutrition during treatment
• Emotional support and coping
• What to expect during procedures
• Healthcare communication tips

🆘 **Special Commands:**
• Type 'emergency' for urgent resources
• Type 'help' to see this message again

💡 **Tips for better responses:**
• Be specific about your concerns
• Mention your treatment stage if relevant
• Ask follow-up questions

Remember: Always consult your healthcare team for medical decisions. I'm here to support and inform, not to replace professional medical care."""

    def get_emergency_message(self) -> str:
        """Get emergency contact information"""
        return """🚨 **EMERGENCY RESOURCES**

**Immediate Medical Emergency:**
• Call emergency services (911, 999, etc.)
• Go to your nearest emergency room

**Cancer-Specific Urgent Concerns:**
• Contact your oncology team immediately
• Call your hospital's 24/7 cancer helpline
• Visit your cancer center's emergency department

**Mental Health Crisis:**
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741

**24/7 Cancer Support:**
• American Cancer Society: 1-800-227-2345
• CancerCare Helpline: 1-800-813-4673

⚠️ If you're experiencing severe symptoms like difficulty breathing, chest pain, severe bleeding, or thoughts of self-harm, seek immediate emergency medical care."""

    def add_document_to_db(self, content: str, metadata: Dict):
        """Add a document to the vector database"""
        try:
            # Generate unique ID
            doc_id = hashlib.md5(content.encode()).hexdigest()
            
            # Add content to metadata
            metadata['content'] = content
            metadata['added_date'] = datetime.now().isoformat()
            
            self.vector_db.upsert_document(doc_id, content, metadata)
            logger.info(f"Added document to database: {metadata.get('title', doc_id)}")
            
        except Exception as e:
            logger.error(f"Error adding document to database: {e}")

# FastAPI Application
app = FastAPI(title="Cancer Support Chatbot", version="1.0.0")

# Initialize chatbot
chatbot = CancerSupportChatbot()

@app.post("/webhook")
async def webhook(message: WhatsAppMessage, background_tasks: BackgroundTasks):
    """Webhook endpoint for receiving WhatsApp messages"""
    try:
        # Extract message data
        message_data = message.messageData
        
        # Skip if not a text message
        if message.typeMessage != "textMessage":
            return {"status": "ignored", "reason": "Not a text message"}
        
        text_message = message_data.get("textMessageData", {}).get("textMessage", "")
        chat_id = message.chatId
        user_id = message.senderId
        sender_name = message.senderName
        
        if not text_message.strip():
            return {"status": "ignored", "reason": "Empty message"}
        
        # Process message in background
        background_tasks.add_task(
            chatbot.process_message,
            chat_id, user_id, text_message, sender_name
        )
        
        return {"status": "processing"}
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/add-document")
async def add_document(content: str, title: str, source: str = "Manual Upload", category: str = "general"):
    """Endpoint to add documents to the vector database"""
    try:
        metadata = {
            "title": title,
            "source": source,
            "category": category
        }
        chatbot.add_document_to_db(content, metadata)
        return {"status": "success", "message": f"Document '{title}' added successfully"}
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail="Failed to add document")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/stats")
async def get_stats():
    """Get chatbot statistics"""
    try:
        with chatbot.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get total conversations
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_conversations = cursor.fetchone()[0]
            
            # Get unique users
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
            unique_users = cursor.fetchone()[0]
            
            # Get conversations today
            today = datetime.now().date()
            cursor.execute("SELECT COUNT(*) FROM conversations WHERE DATE(timestamp) = ?", (today,))
            conversations_today = cursor.fetchone()[0]
            
        return {
            "total_conversations": total_conversations,
            "unique_users": unique_users,
            "conversations_today": conversations_today,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

# Sample function to populate initial documents
def populate_sample_documents():
    """Populate the database with sample cancer support documents"""
    sample_docs = [
        {
            "content": """Chemotherapy side effects can vary greatly between patients and treatments. Common side effects include fatigue, nausea, hair loss, and increased infection risk. Fatigue is often the most challenging side effect, typically lasting throughout treatment and sometimes beyond. To manage fatigue: get adequate rest, maintain light physical activity as tolerated, eat nutritious meals, and don't hesitate to ask for help with daily tasks. Nausea can often be prevented or minimized with anti-nausea medications prescribed by your doctor. Take these as directed, even before you feel nauseous.""",
            "metadata": {
                "title": "Managing Chemotherapy Side Effects",
                "source": "Cancer Treatment Guide",
                "category": "treatment"
            }
        },
        {
            "content": """Nutrition during cancer treatment is crucial for maintaining strength and supporting recovery. Focus on eating small, frequent meals throughout the day rather than three large meals. Include protein-rich foods like lean meats, fish, eggs, beans, and nuts to help maintain muscle mass. Stay hydrated by drinking plenty of water, and consider electrolyte drinks if recommended by your healthcare team. If you're experiencing taste changes, try using different seasonings, eating foods at room temperature, or using plastic utensils if food tastes metallic.""",
            "metadata": {
                "title": "Nutrition Guidelines During Cancer Treatment",
                "source": "Oncology Nutrition Guide",
                "category": "nutrition"
            }
        },
        {
            "content": """Coping with a cancer diagnosis involves both practical and emotional adjustments. It's normal to experience a wide range of emotions including fear, anger, sadness, and anxiety. Building a strong support network is essential - this may include family, friends, support groups, counselors, or spiritual advisors. Maintain open communication with your healthcare team about both physical symptoms and emotional concerns. Consider keeping a journal to track your thoughts and feelings. Gentle activities like meditation, deep breathing, or light yoga can help manage stress and anxiety.""",
            "metadata": {
                "title": "Emotional Support and Coping Strategies",
                "source": "Cancer Support Resources",
                "category": "emotional_support"
            }
        }
    ]
    
    for doc in sample_docs:
        try:
            chatbot.add_document_to_db(doc["content"], doc["metadata"])
        except Exception as e:
            logger.error(f"Error adding sample document: {e}")

if __name__ == "__main__":
    # Populate sample documents (run this once)
    populate_sample_documents()
    
    # Run the FastAPI application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
