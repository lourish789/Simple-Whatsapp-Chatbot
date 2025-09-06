#main.py

import os
import asyncio
import nest_asyncio
from datetime import datetime
import re
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from pinecone import Pinecone
from whatsapp_chatbot_python import GreenAPIBot, Notification
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
nest_asyncio.apply()

class CancerCoachAI:
    """
    Intelligent Cancer Coach AI that provides personalized support
    using RAG technology and conversation memory
    """
    
    def __init__(self, llm, embed_model, pinecone_index):
        self.llm = llm
        self.embed_model = embed_model
        self.pinecone_index = pinecone_index
        
        # Enhanced system prompt for cancer coaching
        self.system_prompt_template = """
        You are CancerCare Coach, a compassionate and knowledgeable AI assistant specifically designed to support cancer patients and their families. 

        Your core principles:
        1. **Empathy First**: Always acknowledge the emotional weight of cancer journey
        2. **Evidence-Based**: Use provided medical information while encouraging professional consultation
        3. **Personalized Support**: Tailor responses to individual patient circumstances
        4. **Hope & Encouragement**: Maintain a balance of realism and optimism
        5. **Clear Communication**: Explain complex medical concepts in understandable terms

        Important Guidelines:
        - NEVER provide specific medical diagnoses or treatment recommendations
        - Always encourage patients to consult healthcare professionals for medical decisions
        - Focus on emotional support, general information, and practical guidance
        - Be sensitive to the patient's stage of treatment and emotional state
        - Acknowledge uncertainty when information is not available

        Patient Context: {patient_name} - {cancer_type} - {treatment_stage}

        Relevant Medical Information:
        {doc_content}

        Respond with compassion, accuracy, and practical support while maintaining appropriate medical boundaries.
        """

        # Create conversation prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        # Create conversation chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=False
        )

    def extract_patient_info(self, message):
        """Extract patient information from initial messages"""
        message_lower = message.lower()
        
        # Common cancer types
        cancer_types = [
            'breast cancer', 'lung cancer', 'colorectal cancer', 'prostate cancer',
            'skin cancer', 'melanoma', 'leukemia', 'lymphoma', 'brain cancer',
            'pancreatic cancer', 'liver cancer', 'kidney cancer', 'bladder cancer'
        ]
        
        # Treatment stages
        treatment_stages = [
            'newly diagnosed', 'in treatment', 'chemotherapy', 'radiation',
            'surgery', 'remission', 'survivor', 'palliative care', 'post-treatment'
        ]
        
        detected_cancer = None
        detected_stage = None
        
        for cancer in cancer_types:
            if cancer in message_lower:
                detected_cancer = cancer
                break
        
        for stage in treatment_stages:
            if stage in message_lower:
                detected_stage = stage
                break
                
        return detected_cancer, detected_stage

    def retrieve_relevant_documents(self, user_input, top_k=5):
        """Retrieve most relevant documents from Pinecone"""
        try:
            # Create embedding for user query
            query_embed = self.embed_model.embed_query(user_input)
            query_embed = [float(val) for val in query_embed]
            
            # Query Pinecone
            results = self.pinecone_index.query(
                vector=query_embed,
                top_k=top_k,
                include_values=False,
                include_metadata=True
            )
            
            # Extract document contents
            doc_contents = []
            for match in results.get('matches', []):
                text = match['metadata'].get('text', '')
                relevance_score = match.get('score', 0)
                if text and relevance_score > 0.7:  # Only include highly relevant content
                    doc_contents.append(text)
            
            return "\n\n".join(doc_contents) if doc_contents else "No specific medical information found."
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return "Unable to retrieve relevant medical information at this time."

    def generate_coach_response(self, user_input, patient_context, conversation_history):
        """Generate personalized coaching response using RAG"""
        try:
            # Retrieve relevant medical documents
            doc_content = self.retrieve_relevant_documents(user_input)
            
            # Convert conversation history to LangChain format
            chat_history = []
            for entry in conversation_history[-8:]:  # Last 8 messages for context
                if entry["role"] == "user":
                    chat_history.append(("human", entry["message"]))
                elif entry["role"] == "assistant":
                    chat_history.append(("ai", entry["message"]))

            # Generate response using the chain
            response = self.chain.run(
                input=user_input,
                chat_history=chat_history,
                doc_content=doc_content,
                patient_name=patient_context.get('name', 'friend'),
                cancer_type=patient_context.get('cancer_type', 'your condition'),
                treatment_stage=patient_context.get('treatment_stage', 'your journey')
            )

            return response.strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm having some technical difficulties right now, but I'm here for you. Could you please try asking your question again?"

def initialize_ai_services():
    """Initialize all AI services and return configured instances"""
    
    # Get API keys
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    if not PINECONE_API_KEY or not GOOGLE_API_KEY:
        raise ValueError("Missing required API keys. Please check your environment variables.")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("cancer-coach")  # Use the index we created
    
    # Initialize embedding model
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=GOOGLE_API_KEY
    )
    
    # Initialize language model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Updated model name
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,  # Balance between creativity and consistency
        max_tokens=1000
    )
    
    # Create cancer coach AI
    cancer_coach = CancerCoachAI(llm, embed_model, pinecone_index)
    
    return cancer_coach

# Initialize the AI coach
print("🤖 Initializing Cancer Coach AI...")
cancer_coach = initialize_ai_services()
print("✅ AI Coach ready!")

# Initialize WhatsApp bot
GREEN_API_ID = os.getenv("GREEN_API_ID", "your_green_api_id")
GREEN_API_TOKEN = os.getenv("GREEN_API_TOKEN", "your_green_api_token")

bot = GreenAPIBot(
    GREEN_API_ID, 
    GREEN_API_TOKEN,
    debug_mode=True,
    bot_debug_mode=True
)

# Store patient profiles and conversation history
patient_profiles = {}  # chat_id: {"name": str, "cancer_type": str, "treatment_stage": str, "history": [...]}

@bot.router.message(command="start")
def welcome_handler(notification: Notification) -> None:
    """Handle new patients and welcome messages"""
    chat_id = notification.chat

    # Initialize patient profile
    patient_profiles[chat_id] = {
        "name": None,
        "cancer_type": None,
        "treatment_stage": None,
        "history": [],
        "last_interaction": datetime.now().isoformat()
    }

    welcome_message = (
        "🌟 **Welcome to CancerCare Coach** 🌟\n\n"
        "I'm here to provide support, information, and encouragement throughout your cancer journey. "
        "While I can't replace your medical team, I can offer:\n\n"
        "• 📚 Information about cancer and treatments\n"
        "• 💪 Emotional support and encouragement\n"
        "• 🏥 Guidance on questions to ask your healthcare team\n"
        "• 🤝 Connection to support resources\n\n"
        "To get started, could you share:\n"
        "• Your name (how you'd like me to address you)\n"
        "• Your cancer type (if you're comfortable sharing)\n"
        "• Where you are in your treatment journey\n\n"
        "For example: *\"Hi, I'm Sarah. I was recently diagnosed with breast cancer and starting chemotherapy next week.\"*\n\n"
        "Remember: I encourage you to always consult with your healthcare team for medical decisions. 💙"
    )
    
    notification.answer(welcome_message)

@bot.router.message()
def coach_message_handler(notification: Notification) -> None:
    """Main message handler for cancer coaching conversations"""
    try:
        # Extract message content
        message_data = notification.event.get("messageData", {})
        text_data = message_data.get("textMessageData", {})
        user_message = text_data.get("textMessage", "").strip()
        chat_id = notification.chat

        if not user_message:
            notification.answer("I didn't receive your message clearly. Could you please send it again? 🤗")
            return

        # Initialize profile if new user
        if chat_id not in patient_profiles:
            patient_profiles[chat_id] = {
                "name": None,
                "cancer_type": None,
                "treatment_stage": None,
                "history": [],
                "last_interaction": datetime.now().isoformat()
            }

        patient = patient_profiles[chat_id]

        # Extract patient info from message if not already set
        if not patient["name"] or not patient["cancer_type"]:
            cancer_type, treatment_stage = cancer_coach.extract_patient_info(user_message)
            
            # Simple name extraction
            name_patterns = [
                r"i[''']?m ([a-zA-Z]+)",
                r"my name is ([a-zA-Z]+)",
                r"call me ([a-zA-Z]+)",
                r"i am ([a-zA-Z]+)"
            ]
            
            detected_name = None
            for pattern in name_patterns:
                match = re.search(pattern, user_message.lower())
                if match:
                    detected_name = match.group(1).capitalize()
                    break
            
            # Update patient profile
            if detected_name and not patient["name"]:
                patient["name"] = detected_name
            if cancer_type and not patient["cancer_type"]:
                patient["cancer_type"] = cancer_type
            if treatment_stage and not patient["treatment_stage"]:
                patient["treatment_stage"] = treatment_stage

        # Add user message to history
        patient["history"].append({
            "role": "user",
            "message": user_message,
            "timestamp": datetime.now().isoformat()
        })

        # Trim history to last 20 messages
        if len(patient["history"]) > 20:
            patient["history"] = patient["history"][-20:]

        # Generate AI response
        ai_response = cancer_coach.generate_coach_response(
            user_message, 
            patient, 
            patient["history"]
        )

        # Add AI response to history
        patient["history"].append({
            "role": "assistant",
            "message": ai_response,
            "timestamp": datetime.now().isoformat()
        })

        # Update last interaction
        patient["last_interaction"] = datetime.now().isoformat()

        # Send response
        notification.answer(ai_response)

    except Exception as e:
        print(f"Error in coach handler: {e}")
        notification.answer(
            "I apologize, but I'm experiencing some technical difficulties. "
            "Please try sending your message again, and I'll do my best to help you. 💙"
        )

if __name__ == "__main__":
    print("🚀 Starting CancerCare Coach WhatsApp Bot...")
    print("📱 Ready to support cancer patients via WhatsApp")
    bot.run_forever()
