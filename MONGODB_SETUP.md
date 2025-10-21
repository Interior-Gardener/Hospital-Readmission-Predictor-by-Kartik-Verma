# MongoDB Setup Guide for Hospital Readmission Predictor

This guide will help you set up MongoDB to store patient records.

## Option 1: Local MongoDB Installation

### Windows:
1. Download MongoDB Community Server from: https://www.mongodb.com/try/download/community
2. Run the installer and choose "Complete" installation
3. Install MongoDB as a Windows Service (check this option during installation)
4. MongoDB will start automatically on `mongodb://localhost:27017/`

### Verify Installation:
Open PowerShell and run:
```powershell
mongod --version
```

## Option 2: MongoDB Atlas (Cloud - Free Tier Available)

1. Go to https://www.mongodb.com/cloud/atlas/register
2. Create a free account
3. Create a new cluster (select Free Tier - M0)
4. Wait for cluster to be created (2-5 minutes)
5. Click "Connect" on your cluster
6. Add your IP address to the whitelist (or use 0.0.0.0/0 for testing)
7. Create a database user with username and password
8. Choose "Connect your application"
9. Copy the connection string (looks like: `mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/`)
10. Update your `.env` file with this connection string

## Setup Steps

1. **Install Python dependencies:**
```powershell
pip install -r requirements.txt
```

2. **Configure MongoDB connection:**
   - Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
   - Edit `.env` and update:
     - For local MongoDB: `MONGODB_URI=mongodb://localhost:27017/`
     - For Atlas: `MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/`

3. **Test the connection:**
```powershell
python -c "from database import test_connection; print(test_connection())"
```

## Running the Application

```powershell
streamlit run app.py
```

The app will:
- Automatically connect to MongoDB
- Create the database and collection if they don't exist
- Save each prediction to the database
- Display all previous records in the "Patient Records Database" page

## Database Schema

Each patient record contains:

```python
{
    'timestamp': datetime,
    'demographics': {
        'age': str,
        'gender': str
    },
    'admission_details': {
        'admission_type': str,
        'discharge_disposition': str,
        'admission_source': str
    },
    'clinical_metrics': {
        'time_in_hospital': int,
        'num_lab_procedures': int,
        'number_inpatient': int
    },
    'diagnostic_info': {
        'number_diagnoses': int,
        'max_glu_serum': str,
        'A1Cresult': str
    },
    'treatment_info': {
        'medication_change': str,
        'diabetes_medication': str
    },
    'prediction_result': {
        'prediction': int,
        'probability': float,
        'risk_level': str,
        'risk_percentage': float
    }
}
```

## Features

1. **Automatic Record Storage**: Every prediction is automatically saved to MongoDB
2. **Records Database Page**: View all stored records in a searchable table
3. **Filtering & Sorting**: Filter by risk level, sort by date or risk score
4. **Export Options**: Download records as CSV, Excel, or JSON
5. **Detailed View**: Expand any record to see complete patient information
6. **Analytics**: Visualize database statistics with charts
7. **Database Management**: Clear all records if needed

## Troubleshooting

### "MongoDB Connection Failed" Error:
- **Local MongoDB**: Make sure MongoDB service is running
  ```powershell
  net start MongoDB
  ```
- **MongoDB Atlas**: Check your connection string, username, password, and IP whitelist

### Cannot import 'pymongo':
```powershell
pip install pymongo python-dotenv
```

### Application works but records not saving:
- Check the sidebar "Database Status" indicator
- The app will show a warning if MongoDB is not connected
- Records will still be stored in session memory even if MongoDB is offline

## MongoDB Commands (Optional)

Connect to MongoDB shell:
```powershell
mongosh
```

View databases:
```javascript
show dbs
```

Use your database:
```javascript
use hospital_readmission_db
```

View collections:
```javascript
show collections
```

Count records:
```javascript
db.patient_records.countDocuments()
```

View all records:
```javascript
db.patient_records.find().pretty()
```

Delete all records:
```javascript
db.patient_records.deleteMany({})
```

## Support

For issues, please check:
1. MongoDB is running and accessible
2. Connection string in `.env` is correct
3. Python dependencies are installed
4. Firewall/network settings allow MongoDB connection
