"""
Test script to verify MongoDB setup and functionality
"""
import sys
from database import test_connection, get_patient_collection
from models import PatientRecord

def main():
    print("=" * 60)
    print("Hospital Readmission Predictor - MongoDB Test")
    print("=" * 60)
    print()
    
    # Test 1: Database Connection
    print("Test 1: Testing MongoDB Connection...")
    is_connected, message = test_connection()
    if is_connected:
        print(f"✅ SUCCESS: {message}")
    else:
        print(f"❌ FAILED: {message}")
        print("\nPlease ensure:")
        print("  1. MongoDB is installed and running")
        print("  2. .env file exists with correct MONGODB_URI")
        print("  3. Connection string is valid")
        print("\nSee MONGODB_SETUP.md for detailed setup instructions")
        return False
    print()
    
    # Test 2: Collection Access
    print("Test 2: Testing Collection Access...")
    collection = get_patient_collection()
    if collection is not None:
        print(f"✅ SUCCESS: Connected to collection 'patient_records'")
    else:
        print("❌ FAILED: Could not access collection")
        return False
    print()
    
    # Test 3: Statistics
    print("Test 3: Getting Database Statistics...")
    stats = PatientRecord.get_statistics()
    if stats:
        print(f"✅ SUCCESS: Retrieved statistics")
        print(f"   - Total Records: {stats.get('total_records', 0)}")
        print(f"   - High Risk: {stats.get('high_risk_count', 0)}")
        print(f"   - Low Risk: {stats.get('low_risk_count', 0)}")
    else:
        print("⚠️  WARNING: No statistics available (database might be empty)")
    print()
    
    # Test 4: Sample Record Creation (optional)
    print("Test 4: Testing Record Creation...")
    sample_data = {
        'age': '[50-60)',
        'gender': 'Male',
        'admission_type_id': 'Emergency',
        'discharge_disposition_id': 'Home',
        'admission_source_id': 'Emergency Room',
        'time_in_hospital': 5,
        'num_lab_procedures': 45,
        'number_inpatient': 1,
        'number_diagnoses': 9,
        'max_glu_serum': 'None',
        'A1Cresult': 'None',
        'change': 'No',
        'diabetesMed': 'Yes'
    }
    
    # Ask user if they want to create a test record
    create_test = input("Create a test record in database? (y/n): ").lower().strip()
    
    if create_test == 'y':
        record_id = PatientRecord.save_record(sample_data, prediction=0, probability=0.35)
        if record_id:
            print(f"✅ SUCCESS: Test record created with ID: {record_id}")
            print("   You can view this record in the 'Patient Records Database' page")
        else:
            print("❌ FAILED: Could not create test record")
    else:
        print("⏭️  Skipped test record creation")
    print()
    
    print("=" * 60)
    print("✅ MongoDB Setup Complete!")
    print("=" * 60)
    print("\nYou can now run the application:")
    print("  streamlit run app.py")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
