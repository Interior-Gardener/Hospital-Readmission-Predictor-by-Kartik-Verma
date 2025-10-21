# 🚀 Quick Start Guide - MongoDB Integration

## What's New?

Your Hospital Readmission Predictor now stores all patient records in MongoDB! Every prediction is automatically saved and can be viewed in a beautiful table interface.

## 📋 Prerequisites

You need either:
- **Option A**: MongoDB installed locally (recommended for development)
- **Option B**: MongoDB Atlas account (free cloud database)

## 🛠️ Setup (5 minutes)

### Step 1: Install MongoDB Dependencies

```powershell
pip install pymongo python-dotenv
```

### Step 2: Set Up MongoDB

#### Option A: Local MongoDB (Recommended)
1. Download: https://www.mongodb.com/try/download/community
2. Install with default settings (install as Windows Service)
3. MongoDB will run automatically on `mongodb://localhost:27017/`

#### Option B: MongoDB Atlas (Cloud)
1. Create free account at https://www.mongodb.com/cloud/atlas
2. Create a free cluster (M0)
3. Get your connection string
4. Update `.env` file with your connection string

### Step 3: Configure Connection

The `.env` file is already created with default local settings:
```
MONGODB_URI=mongodb://localhost:27017/
DATABASE_NAME=hospital_readmission_db
```

**For MongoDB Atlas**, update the `MONGODB_URI` in `.env`:
```
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/
```

### Step 4: Test Connection

```powershell
python test_mongodb.py
```

This will verify your MongoDB connection and optionally create a test record.

### Step 5: Run the Application

```powershell
streamlit run app.py
```

## ✨ New Features

### 1. **Automatic Record Saving**
Every prediction is now automatically saved to MongoDB with full patient details and results.

### 2. **Patient Records Database Page** 
Access via the navigation menu to see:
- ✅ All previous patient records in a beautiful table
- 📊 Filter by risk level (High/Low/All)
- 📅 Sort by date or risk score
- 🔍 Detailed view of any record
- 📥 Export as CSV, Excel, or JSON
- 📈 Visual analytics and charts

### 3. **Database Status Indicator**
Check the sidebar to see if MongoDB is connected (green checkmark) or offline (orange warning).

### 4. **Persistent Storage**
Unlike before where data was lost when you closed the browser, now:
- Records persist across sessions
- View history from any time
- Track patient outcomes over time
- Generate reports from historical data

## 📊 Using the Patient Records Database

1. **Make predictions** on the Home page - they auto-save to MongoDB
2. **Navigate to "Patient Records Database"** in the sidebar
3. **View all records** in an interactive table
4. **Filter and sort** records as needed
5. **Click on any record** to see complete details
6. **Export data** for external analysis

## 🔧 Database Management

From the Patient Records Database page, you can:
- View total records and statistics
- Filter by high/low risk
- Sort by various criteria
- Export data in multiple formats
- Clear all records (with confirmation)

## 📁 File Structure

New files added:
```
database.py           # MongoDB connection configuration
models.py             # Patient record schema and CRUD operations
test_mongodb.py       # Test script for MongoDB setup
.env                  # Environment variables (MongoDB connection)
.env.example          # Example environment file
.gitignore            # Protects sensitive files
MONGODB_SETUP.md      # Detailed MongoDB setup guide
```

## 🐛 Troubleshooting

### "MongoDB Connection Failed"

**For Local MongoDB:**
```powershell
# Check if MongoDB is running
net start MongoDB

# If not installed as a service, start manually:
mongod
```

**For MongoDB Atlas:**
- Verify your connection string in `.env`
- Check username and password
- Ensure your IP is whitelisted in Atlas

### "Import pymongo could not be resolved"
```powershell
pip install pymongo python-dotenv
```

### App runs but records don't save
- Check sidebar "Database Status" - should show green checkmark
- Run `python test_mongodb.py` to diagnose
- Records still save in session memory even if DB is offline

## 📖 Schema Overview

Each patient record includes:
- **Demographics**: Age, gender
- **Admission Details**: Type, source, disposition
- **Clinical Metrics**: Hospital stay, lab procedures, visits
- **Diagnostic Info**: Diagnoses count, glucose, A1C
- **Treatment Info**: Medication changes
- **Prediction Result**: Risk level, score, probability

## 🎯 Next Steps

1. ✅ Install dependencies: `pip install pymongo python-dotenv`
2. ✅ Set up MongoDB (local or Atlas)
3. ✅ Test connection: `python test_mongodb.py`
4. ✅ Run app: `streamlit run app.py`
5. ✅ Make predictions and see them saved!
6. ✅ View records in "Patient Records Database" page

## 💡 Tips

- **First Time**: Run `test_mongodb.py` to create a test record
- **Local Dev**: Use local MongoDB for faster performance
- **Production**: Use MongoDB Atlas for reliability and backups
- **Backup**: Export data regularly using the export feature
- **Performance**: Database queries are cached for speed

## 🆘 Need Help?

1. Check `MONGODB_SETUP.md` for detailed setup instructions
2. Run `python test_mongodb.py` to diagnose issues
3. Check sidebar "Database Status" indicator
4. Verify `.env` file configuration

Enjoy your new persistent patient record storage! 🎉
