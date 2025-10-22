"""
MongoDB Database Configuration and Connection
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection string
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'hospital_readmission_db')

def get_database_connection():
    """
    Create and return MongoDB database connection.
    Uses lazy loading to avoid slowing down app startup.
    """
    import streamlit as st
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    
    # Cache the connection in session state
    if 'db_connection' not in st.session_state:
        try:
            # Create MongoDB client with short timeout
            client = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=2000,  # 2 second timeout
                connectTimeoutMS=2000
            )
            
            # Test connection (non-blocking)
            client.admin.command('ping')
            
            # Get database
            db = client[DATABASE_NAME]
            
            st.session_state.db_connection = db
            st.session_state.db_status = "connected"
            print(f"✅ Connected to MongoDB: {DATABASE_NAME}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"⚠️ MongoDB connection failed: {str(e)}")
            st.session_state.db_connection = None
            st.session_state.db_status = "failed"
            
        except Exception as e:
            print(f"⚠️ Unexpected MongoDB error: {str(e)}")
            st.session_state.db_connection = None
            st.session_state.db_status = "error"
    
    return st.session_state.get('db_connection', None)

def get_patient_collection():
    """Get the patient records collection"""
    db = get_database_connection()
    if db is not None:
        return db['patient_records']
    return None

def test_connection():
    """Test if MongoDB connection is working"""
    try:
        db = get_database_connection()
        if db is not None:
            # Try to perform a simple operation
            db.list_collection_names()
            return True, "Connected successfully"
        return False, "Database unavailable (app will work without MongoDB)"
    except Exception as e:
        return False, f"Connection test failed: {str(e)}"
