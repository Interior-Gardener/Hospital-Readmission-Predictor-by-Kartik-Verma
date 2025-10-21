"""
Patient Record Schema and Database Operations
This module defines the schema for patient records and provides CRUD operations
"""
from datetime import datetime
from typing import Dict, List, Optional
from database import get_patient_collection
import streamlit as st

class PatientRecord:
    """
    Patient Record Schema (similar to Mongoose schema)
    
    Fields:
    - patient_id: Unique identifier
    - timestamp: When the prediction was made
    - demographics: Age, gender
    - admission_details: Type, source, disposition
    - clinical_metrics: Hospital stay, lab procedures, etc.
    - diagnostic_info: Diagnoses, glucose, A1C results
    - treatment_info: Medication changes
    - prediction_result: Model prediction and probability
    """
    
    @staticmethod
    def create_schema(patient_data: Dict, prediction: int, probability: float) -> Dict:
        """
        Create a patient record following the schema
        
        Args:
            patient_data: Dictionary containing patient information
            prediction: Model prediction (0 or 1)
            probability: Prediction probability
            
        Returns:
            Dictionary following the patient record schema
        """
        record = {
            'timestamp': datetime.now(),
            'demographics': {
                'age': patient_data.get('age'),
                'gender': patient_data.get('gender')
            },
            'admission_details': {
                'admission_type': patient_data.get('admission_type_id'),
                'discharge_disposition': patient_data.get('discharge_disposition_id'),
                'admission_source': patient_data.get('admission_source_id')
            },
            'clinical_metrics': {
                'time_in_hospital': patient_data.get('time_in_hospital'),
                'num_lab_procedures': patient_data.get('num_lab_procedures'),
                'number_inpatient': patient_data.get('number_inpatient')
            },
            'diagnostic_info': {
                'number_diagnoses': patient_data.get('number_diagnoses'),
                'max_glu_serum': patient_data.get('max_glu_serum'),
                'A1Cresult': patient_data.get('A1Cresult')
            },
            'treatment_info': {
                'medication_change': patient_data.get('change'),
                'diabetes_medication': patient_data.get('diabetesMed')
            },
            'prediction_result': {
                'prediction': prediction,
                'probability': float(probability),
                'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
                'risk_percentage': float(probability * 100)
            }
        }
        
        return record
    
    @staticmethod
    def save_record(patient_data: Dict, prediction: int, probability: float) -> Optional[str]:
        """
        Save a patient record to MongoDB
        
        Args:
            patient_data: Patient information
            prediction: Model prediction
            probability: Prediction probability
            
        Returns:
            Record ID if successful, None otherwise
        """
        try:
            collection = get_patient_collection()
            
            if collection is None:
                st.warning("⚠️ Database not available. Record not saved.")
                return None
            
            # Create record following schema
            record = PatientRecord.create_schema(patient_data, prediction, probability)
            
            # Insert into MongoDB
            result = collection.insert_one(record)
            
            print(f"✅ Record saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            st.error(f"❌ Error saving record: {str(e)}")
            print(f"Error details: {str(e)}")
            return None
    
    @staticmethod
    def get_all_records(limit: int = 100, sort_by: str = 'timestamp', 
                       sort_order: int = -1) -> List[Dict]:
        """
        Retrieve all patient records from MongoDB
        
        Args:
            limit: Maximum number of records to retrieve
            sort_by: Field to sort by
            sort_order: 1 for ascending, -1 for descending
            
        Returns:
            List of patient records
        """
        try:
            collection = get_patient_collection()
            
            if collection is None:
                return []
            
            # Query all records with sorting and limit
            records = list(
                collection.find()
                .sort(sort_by, sort_order)
                .limit(limit)
            )
            
            return records
            
        except Exception as e:
            st.error(f"❌ Error retrieving records: {str(e)}")
            return []
    
    @staticmethod
    def get_records_by_risk(risk_level: str, limit: int = 50) -> List[Dict]:
        """
        Retrieve records filtered by risk level
        
        Args:
            risk_level: 'High Risk' or 'Low Risk'
            limit: Maximum number of records
            
        Returns:
            List of filtered patient records
        """
        try:
            collection = get_patient_collection()
            
            if collection is None:
                return []
            
            # Query with filter
            records = list(
                collection.find({'prediction_result.risk_level': risk_level})
                .sort('timestamp', -1)
                .limit(limit)
            )
            
            return records
            
        except Exception as e:
            st.error(f"❌ Error retrieving filtered records: {str(e)}")
            return []
    
    @staticmethod
    def get_statistics() -> Dict:
        """
        Get statistics about stored records
        
        Returns:
            Dictionary with statistics
        """
        try:
            collection = get_patient_collection()
            
            if collection is None:
                return {}
            
            total_records = collection.count_documents({})
            high_risk_count = collection.count_documents({'prediction_result.risk_level': 'High Risk'})
            low_risk_count = collection.count_documents({'prediction_result.risk_level': 'Low Risk'})
            
            # Calculate average risk probability
            pipeline = [
                {
                    '$group': {
                        '_id': None,
                        'avg_probability': {'$avg': '$prediction_result.probability'}
                    }
                }
            ]
            
            avg_result = list(collection.aggregate(pipeline))
            avg_probability = avg_result[0]['avg_probability'] if avg_result else 0
            
            return {
                'total_records': total_records,
                'high_risk_count': high_risk_count,
                'low_risk_count': low_risk_count,
                'high_risk_percentage': (high_risk_count / total_records * 100) if total_records > 0 else 0,
                'average_risk_probability': avg_probability
            }
            
        except Exception as e:
            st.error(f"❌ Error calculating statistics: {str(e)}")
            return {}
    
    @staticmethod
    def delete_record(record_id: str) -> bool:
        """
        Delete a specific record
        
        Args:
            record_id: MongoDB ObjectId as string
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from bson.objectid import ObjectId
            collection = get_patient_collection()
            
            if collection is None:
                return False
            
            result = collection.delete_one({'_id': ObjectId(record_id)})
            return result.deleted_count > 0
            
        except Exception as e:
            st.error(f"❌ Error deleting record: {str(e)}")
            return False
    
    @staticmethod
    def clear_all_records() -> bool:
        """
        Clear all records from the collection (use with caution!)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = get_patient_collection()
            
            if collection is None:
                return False
            
            result = collection.delete_many({})
            print(f"🗑️ Deleted {result.deleted_count} records")
            return True
            
        except Exception as e:
            st.error(f"❌ Error clearing records: {str(e)}")
            return False
