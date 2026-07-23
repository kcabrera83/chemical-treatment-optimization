# API Documentation - Chemical Treatment Optimization

## Base URL

```
http://localhost:5009
```

## Endpoints

### GET /

Serve the main web dashboard UI.

**Response:** HTML page

---

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "chemical-treatment-optimization"
}
```

---

### GET /api/models

Return information about loaded models and treatment types.

**Response:**
```json
{
  "dosage_optimizer": {
    "type": "GradientBoostingRegressor",
    "features": ["treatment_type", "temperature_c", "ph", "water_hardness"],
    "target": "dosage_ppm"
  },
  "effectiveness_predictor": {
    "type": "RandomForestClassifier",
    "features": ["dosage_ppm", "temperature_c", "ph", "water_hardness", "treatment_type"],
    "target": "effectiveness_category (poor/fair/good/excellent)"
  },
  "treatment_types": ["scale_inhibitor", "corrosion_inhibitor", "demulsifier", "paraffin_inhibitor"]
}
```

---

### POST /api/optimize

Optimize chemical treatment dosage based on operating conditions.

**Request:**
```json
{
  "treatment_type": "scale_inhibitor",
  "temperature_c": 65,
  "ph": 7.2,
  "water_hardness": 350
}
```

**Required Fields:**

| Field | Type | Description | Range |
|-------|------|-------------|-------|
| treatment_type | string | Type of chemical treatment | One of: scale_inhibitor, corrosion_inhibitor, demulsifier, paraffin_inhibitor |
| temperature_c | float | Operating temperature | 20 - 120 C |
| ph | float | Water pH level | 4.0 - 10.0 |
| water_hardness | float | Water hardness (CaCO3) | 50 - 800 ppm |

**Response:**
```json
{
  "optimal_dosage_ppm": 85.23,
  "treatment_type": "scale_inhibitor",
  "conditions": {
    "temperature_c": 65.0,
    "ph": 7.2,
    "water_hardness": 350.0
  }
}
```

**Error Responses:**
| Status | Condition | Body |
|--------|-----------|------|
| 400 | No JSON data | `{"error": "No JSON data provided"}` |
| 400 | Missing fields | `{"error": "Missing fields: [...]"}"}` |
| 400 | Invalid treatment type | `{"error": "Invalid treatment_type. Must be one of: [...]"}` |

---

### POST /api/predict

Predict treatment effectiveness for given conditions and dosage.

**Request:**
```json
{
  "treatment_type": "corrosion_inhibitor",
  "dosage_ppm": 100,
  "temperature_c": 50,
  "ph": 6.5,
  "water_hardness": 200
}
```

**Required Fields:**

| Field | Type | Description |
|-------|------|-------------|
| treatment_type | string | Type of chemical treatment |
| dosage_ppm | float | Chemical dosage in parts per million |
| temperature_c | float | Operating temperature (C) |
| ph | float | Water pH level |
| water_hardness | float | Water hardness (CaCO3 ppm) |

**Response:**
```json
{
  "predicted_effectiveness": "good",
  "treatment_type": "corrosion_inhibitor",
  "dosage_ppm": 100.0,
  "conditions": {
    "temperature_c": 50.0,
    "ph": 6.5,
    "water_hardness": 200.0
  }
}
```

**Effectiveness Categories:**
| Category | Score Range |
|----------|-------------|
| poor | < 0.25 |
| fair | 0.25 - 0.50 |
| good | 0.50 - 0.75 |
| excellent | >= 0.75 |

**Error Responses:**
| Status | Condition | Body |
|--------|-----------|------|
| 400 | No JSON data | `{"error": "No JSON data provided"}` |
| 400 | Missing fields | `{"error": "Missing fields: [...]"}"}` |
| 400 | Invalid treatment type | `{"error": "Invalid treatment_type. Must be one of: [...]"}` |

---

### GET /api/docs

Return OpenAPI 3.0 specification.

---

## Treatment Types

| Type | Description | Dosage Range (ppm) |
|------|-------------|-------------------|
| scale_inhibitor | Prevents mineral scale buildup | 10 - 150 |
| corrosion_inhibitor | Prevents metal corrosion | 20 - 200 |
| demulsifier | Breaks oil-water emulsions | 5 - 100 |
| paraffin_inhibitor | Prevents wax/paraffin deposits | 15 - 180 |

## Error Codes

- **200**: Success
- **400**: Bad request (missing or invalid parameters)
- **500**: Internal server error

---

*Elaborado por Ing. Kelvin Cabrera*
