# 📋 Patient Record Schema - Visual Guide

## Complete Schema Structure

```javascript
{
  // Auto-generated unique identifier
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  
  // When the prediction was made
  "timestamp": ISODate("2025-10-21T10:30:00Z"),
  
  // Patient Demographics
  "demographics": {
    "age": "[50-60)",           // Age range
    "gender": "Male"            // Male or Female
  },
  
  // Hospital Admission Information
  "admission_details": {
    "admission_type": "Emergency",              // Emergency, Urgent, Elective
    "discharge_disposition": "Home",            // Home, Transferred, Expired
    "admission_source": "Emergency Room"        // How patient was admitted
  },
  
  // Clinical Measurements
  "clinical_metrics": {
    "time_in_hospital": 5,          // Days (1-14)
    "num_lab_procedures": 45,       // Count (0-100)
    "number_inpatient": 1           // Prior visits (0-20)
  },
  
  // Diagnostic Information
  "diagnostic_info": {
    "number_diagnoses": 9,          // Count (1-16)
    "max_glu_serum": "None",        // None, Norm, >200, >300
    "A1Cresult": "None"             // None, Norm, >7, >8
  },
  
  // Treatment Details
  "treatment_info": {
    "medication_change": "No",      // No, Ch (Changed)
    "diabetes_medication": "Yes"    // Yes, No
  },
  
  // AI Prediction Results
  "prediction_result": {
    "prediction": 0,                // 0 = Low Risk, 1 = High Risk
    "probability": 0.35,            // Raw probability (0.0 - 1.0)
    "risk_level": "Low Risk",       // Human-readable risk level
    "risk_percentage": 35.0         // Percentage format (0-100)
  }
}
```

## Field Types & Validation

### Demographics
| Field | Type | Values | Required |
|-------|------|--------|----------|
| age | String | Age ranges: [20-30), [30-40), [40-50), [50-60), [60-70), [70-80), [80-90), [90-100) | ✅ Yes |
| gender | String | "Male" or "Female" | ✅ Yes |

### Admission Details
| Field | Type | Values | Required |
|-------|------|--------|----------|
| admission_type | String | "Emergency", "Urgent", "Elective" | ✅ Yes |
| discharge_disposition | String | "Home", "Transferred", "Expired" | ✅ Yes |
| admission_source | String | "Physician Referral", "Emergency Room", "Transfer" | ✅ Yes |

### Clinical Metrics
| Field | Type | Range | Required |
|-------|------|-------|----------|
| time_in_hospital | Integer | 1-14 days | ✅ Yes |
| num_lab_procedures | Integer | 0-100 | ✅ Yes |
| number_inpatient | Integer | 0-20 | ✅ Yes |

### Diagnostic Info
| Field | Type | Values | Required |
|-------|------|--------|----------|
| number_diagnoses | Integer | 1-16 | ✅ Yes |
| max_glu_serum | String | "None", "Norm", ">200", ">300" | ✅ Yes |
| A1Cresult | String | "None", "Norm", ">7", ">8" | ✅ Yes |

### Treatment Info
| Field | Type | Values | Required |
|-------|------|--------|----------|
| medication_change | String | "No", "Ch" | ✅ Yes |
| diabetes_medication | String | "Yes", "No" | ✅ Yes |

### Prediction Result
| Field | Type | Range/Values | Auto-Generated |
|-------|------|--------------|----------------|
| prediction | Integer | 0 (Low) or 1 (High) | ✅ Yes |
| probability | Float | 0.0 - 1.0 | ✅ Yes |
| risk_level | String | "Low Risk" or "High Risk" | ✅ Yes |
| risk_percentage | Float | 0.0 - 100.0 | ✅ Yes |

## Example Records

### Example 1: High Risk Patient

```json
{
  "_id": ObjectId("671234567890abcdef123456"),
  "timestamp": ISODate("2025-10-21T14:30:00Z"),
  "demographics": {
    "age": "[70-80)",
    "gender": "Female"
  },
  "admission_details": {
    "admission_type": "Emergency",
    "discharge_disposition": "Home",
    "admission_source": "Emergency Room"
  },
  "clinical_metrics": {
    "time_in_hospital": 10,
    "num_lab_procedures": 85,
    "number_inpatient": 3
  },
  "diagnostic_info": {
    "number_diagnoses": 14,
    "max_glu_serum": ">300",
    "A1Cresult": ">8"
  },
  "treatment_info": {
    "medication_change": "Ch",
    "diabetes_medication": "Yes"
  },
  "prediction_result": {
    "prediction": 1,
    "probability": 0.78,
    "risk_level": "High Risk",
    "risk_percentage": 78.0
  }
}
```

**Risk Factors Present:**
- ⚠️ Extended hospital stay (10 days)
- ⚠️ High number of lab procedures (85)
- ⚠️ Multiple prior visits (3)
- ⚠️ Many diagnoses (14)
- ⚠️ Very high glucose (>300)
- ⚠️ High A1C (>8)
- ⚠️ Medication changed

### Example 2: Low Risk Patient

```json
{
  "_id": ObjectId("671234567890abcdef123457"),
  "timestamp": ISODate("2025-10-21T15:45:00Z"),
  "demographics": {
    "age": "[40-50)",
    "gender": "Male"
  },
  "admission_details": {
    "admission_type": "Elective",
    "discharge_disposition": "Home",
    "admission_source": "Physician Referral"
  },
  "clinical_metrics": {
    "time_in_hospital": 3,
    "num_lab_procedures": 25,
    "number_inpatient": 0
  },
  "diagnostic_info": {
    "number_diagnoses": 5,
    "max_glu_serum": "Norm",
    "A1Cresult": "Norm"
  },
  "treatment_info": {
    "medication_change": "No",
    "diabetes_medication": "Yes"
  },
  "prediction_result": {
    "prediction": 0,
    "probability": 0.25,
    "risk_level": "Low Risk",
    "risk_percentage": 25.0
  }
}
```

**Positive Indicators:**
- ✅ Short hospital stay (3 days)
- ✅ Moderate lab procedures (25)
- ✅ No prior visits (0)
- ✅ Few diagnoses (5)
- ✅ Normal glucose
- ✅ Normal A1C
- ✅ No medication changes
- ✅ Elective admission (planned)

## Schema Benefits

### 1. **Organized Structure**
Data is logically grouped into categories for easy access

### 2. **Complete Information**
Every important aspect of patient care is captured

### 3. **Searchable**
MongoDB allows fast queries on any field:
```javascript
// Find all high-risk patients
db.patient_records.find({ "prediction_result.risk_level": "High Risk" })

// Find elderly patients
db.patient_records.find({ "demographics.age": "[70-80)" })

// Find patients with long stays
db.patient_records.find({ "clinical_metrics.time_in_hospital": { $gt: 7 } })
```

### 4. **Analytics Ready**
Structured format enables easy aggregation:
```javascript
// Average risk by age group
db.patient_records.aggregate([
  {
    $group: {
      _id: "$demographics.age",
      avgRisk: { $avg: "$prediction_result.risk_percentage" }
    }
  }
])
```

### 5. **Export Friendly**
Clean structure exports well to CSV/Excel/JSON

## Index Recommendations

For better performance with large datasets:

```javascript
// Index on timestamp for chronological queries
db.patient_records.createIndex({ "timestamp": -1 })

// Index on risk level for filtering
db.patient_records.createIndex({ "prediction_result.risk_level": 1 })

// Compound index for risk analysis
db.patient_records.createIndex({ 
  "prediction_result.risk_level": 1, 
  "timestamp": -1 
})
```

## Schema Validation (Optional)

You can add MongoDB validation for data integrity:

```javascript
db.createCollection("patient_records", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["timestamp", "demographics", "prediction_result"],
      properties: {
        timestamp: {
          bsonType: "date"
        },
        demographics: {
          bsonType: "object",
          required: ["age", "gender"],
          properties: {
            age: { bsonType: "string" },
            gender: { enum: ["Male", "Female"] }
          }
        },
        prediction_result: {
          bsonType: "object",
          required: ["prediction", "probability"],
          properties: {
            prediction: { bsonType: "int", minimum: 0, maximum: 1 },
            probability: { bsonType: "double", minimum: 0, maximum: 1 }
          }
        }
      }
    }
  }
})
```

## Summary

This schema provides:
- ✅ **Complete patient history** in one document
- ✅ **Logical organization** for easy understanding
- ✅ **Fast queries** with proper indexing
- ✅ **Data integrity** through validation
- ✅ **Analytics capabilities** for insights
- ✅ **Export compatibility** for external tools

---

**Schema Version**: 1.0
**Last Updated**: October 21, 2025
