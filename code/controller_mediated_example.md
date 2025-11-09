# Controller-Mediated Architecture Example: Student Retention ML Pipeline

## Scenario: User Requests ML Pipeline for Student Retention Prediction

**User Query:**
> "I need to build a machine learning model to predict which students are at risk of dropping out. I have student enrollment data, grades, and demographic information."

---

## Step-by-Step Execution Flow

### **STEP 1: User Query Reception & HCI Processing**

**Controller receives:** Raw user query string

**Action:** Controller routes to **HCI SLM (3.8B)** - specialized for understanding user intent and generating responses

**HCI SLM Processing:**
```
Input: "I need to build a machine learning model to predict which students are at risk of dropping out. I have student enrollment data, grades, and demographic information."

HCI SLM Output (Structured Intent):
{
  "task_type": "classification",
  "target": "student_retention",
  "data_sources": ["enrollment", "grades", "demographics"],
  "user_goal": "predict_risk_of_dropout",
  "context": "educational_institution"
}
```

**Controller Action:** Stores intent, proceeds to workflow initialization

---

### **STEP 2: Controller Workflow Coordination**

**Controller analyzes intent and creates workflow graph:**

```
Workflow Dependencies:
1. Data Agent → (output: cleaned_data)
2. Feature Agent → (requires: cleaned_data, output: engineered_features)
3. Case Retriever → (parallel with Data Agent, output: similar_cases)
4. Planning Agent → (requires: cleaned_data, similar_cases, output: pipeline_plan)
5. Coding Agent → (requires: pipeline_plan, engineered_features, output: pipeline_code)
6. Model Agent → (requires: pipeline_code, output: model_selection)
7. Hyperparameter Agent → (requires: model_selection, output: tuned_model)
8. Evaluator Agent → (requires: tuned_model, output: performance_report)
```

**Controller Decision:** Execute Data Agent and Case Retriever in parallel (no dependencies)

---

### **STEP 3: Parallel Execution - Data Agent & Case Retrieval**

#### **3A. Data Agent (1.5B SLM) - Data Quality Assessment**

**Controller sends structured message to Data Agent:**

```json
{
  "agent": "data_agent",
  "task": "assess_data_quality",
  "inputs": {
    "data_sources": ["enrollment.csv", "grades.csv", "demographics.csv"],
    "task_type": "classification"
  }
}
```

**Data Agent Processing (focused context, ~200 tokens):**
- Analyzes schema: `student_id, enrollment_date, gpa, major, ethnicity, first_gen_status, ...`
- Detects missing values: `gpa: 5% missing, ethnicity: 2% missing`
- Identifies data types: `categorical: major, ethnicity, first_gen_status`
- Generates cleaning code:

```python
# Data Agent Generated Code
def clean_data(df_enrollment, df_grades, df_demographics):
    # Merge datasets
    df = df_enrollment.merge(df_grades, on='student_id')
    df = df.merge(df_demographics, on='student_id')
    
    # Handle missing values
    df['gpa'] = df['gpa'].fillna(df.groupby('major')['gpa'].transform('median'))
    df['ethnicity'] = df['ethnicity'].fillna('Unknown')
    
    # Type conversions
    df['enrollment_date'] = pd.to_datetime(df['enrollment_date'])
    df['first_gen_status'] = df['first_gen_status'].astype('category')
    
    return df
```

**Data Agent Response to Controller:**
```json
{
  "status": "success",
  "cleaned_data_path": "/sandbox/cleaned_student_data.csv",
  "data_profile": {
    "rows": 50000,
    "features": 15,
    "missing_values_handled": true,
    "data_types_validated": true
  }
}
```

#### **3B. Case Retriever (1.5B SLM) - Similar Case Search**

**Controller sends structured message to Case Retriever:**

```json
{
  "agent": "case_retriever",
  "task": "retrieve_similar_cases",
  "inputs": {
    "query_description": "student retention prediction classification",
    "data_characteristics": {
      "modality": "tabular",
      "problem_type": "classification",
      "domain": "education"
    }
  }
}
```

**Case Retriever Processing (semantic similarity, ~150 tokens):**
- Generates embedding for query: `[0.23, -0.45, 0.67, ...]` (384-dim vector)
- Compares with case bank (10,000 cases) using cosine similarity
- Returns top-3 similar cases:

```json
{
  "status": "success",
  "retrieved_cases": [
    {
      "case_id": "kaggle_2023_student_success",
      "similarity": 0.89,
      "description": "Student success prediction using XGBoost with feature engineering",
      "performance": {"AUC": 0.87, "accuracy": 0.82}
    },
    {
      "case_id": "openml_university_retention",
      "similarity": 0.85,
      "description": "University student retention with temporal features",
      "performance": {"AUC": 0.85, "accuracy": 0.80}
    },
    {
      "case_id": "competition_2024_education",
      "similarity": 0.82,
      "description": "Educational outcome prediction with demographic features",
      "performance": {"AUC": 0.83, "accuracy": 0.79}
    }
  ]
}
```

**Controller Action:** Receives both responses, stores in state manager, proceeds to next stage

---

### **STEP 4: Feature Engineering (Feature Agent 2B SLM)**

**Controller sends structured message to Feature Agent:**

```json
{
  "agent": "feature_agent",
  "task": "generate_features",
  "inputs": {
    "cleaned_data_path": "/sandbox/cleaned_student_data.csv",
    "data_profile": {
      "temporal_features": ["enrollment_date"],
      "categorical_features": ["major", "ethnicity", "first_gen_status"],
      "numerical_features": ["gpa", "credit_hours"]
    },
    "target": "retention_status"
  }
}
```

**Feature Agent Processing (CAAFE-style, ~300 tokens):**
- Routes to specialized sub-agents:
  - **Temporal Feature Agent (2B):** Generates `days_since_enrollment`, `semester_progress`, `academic_year`
  - **Categorical Feature Agent (2B):** Generates `major_difficulty_encoding`, `first_gen_interaction`
  - **Numerical Feature Agent (2.4B):** Generates `gpa_trend`, `credit_load_score`

**Feature Agent Generated Code:**
```python
# Temporal Features
df['days_since_enrollment'] = (pd.Timestamp.now() - df['enrollment_date']).dt.days
df['semester_progress'] = df['days_since_enrollment'] / 120  # 4-month semester
df['academic_year'] = df['enrollment_date'].dt.year

# Categorical Features
major_difficulty = {'Engineering': 0.8, 'Business': 0.5, 'Arts': 0.4, ...}
df['major_difficulty'] = df['major'].map(major_difficulty)
df['first_gen_major_interaction'] = df['first_gen_status'].astype(int) * df['major_difficulty']

# Numerical Features
df['gpa_trend'] = df.groupby('student_id')['gpa'].transform(lambda x: x.diff().mean())
df['credit_load_score'] = df['credit_hours'] / df['credit_hours'].max()
```

**Feature Agent Response:**
```json
{
  "status": "success",
  "features_generated": 8,
  "feature_engineering_code": "...",
  "engineered_data_path": "/sandbox/engineered_features.csv",
  "performance_improvement": {"baseline_auc": 0.75, "with_features_auc": 0.82}
}
```

---

### **STEP 5: Pipeline Planning (Planning Agent 2B SLM)**

**Controller sends structured message to Planning Agent:**

```json
{
  "agent": "planning_agent",
  "task": "generate_pipeline_plan",
  "inputs": {
    "cleaned_data": "/sandbox/cleaned_student_data.csv",
    "engineered_features": "/sandbox/engineered_features.csv",
    "retrieved_cases": [...],  // from Step 3B
    "task_type": "classification",
    "target": "retention_status"
  }
}
```

**Planning Agent Processing (pipeline structure, ~250 tokens):**
- Analyzes retrieved cases for successful patterns
- Creates pipeline plan:

```json
{
  "status": "success",
  "pipeline_plan": {
    "stages": [
      {
        "stage": 1,
        "operation": "data_loading",
        "dependencies": []
      },
      {
        "stage": 2,
        "operation": "feature_selection",
        "dependencies": [1],
        "method": "mutual_information"
      },
      {
        "stage": 3,
        "operation": "train_test_split",
        "dependencies": [2],
        "split_ratio": 0.8
      },
      {
        "stage": 4,
        "operation": "model_training",
        "dependencies": [3],
        "candidates": ["XGBoost", "LightGBM", "RandomForest"]
      },
      {
        "stage": 5,
        "operation": "hyperparameter_tuning",
        "dependencies": [4]
      },
      {
        "stage": 6,
        "operation": "evaluation",
        "dependencies": [5],
        "metrics": ["AUC", "accuracy", "precision", "recall"]
      }
    ]
  }
}
```

---

### **STEP 6: Code Generation (Coding Agent 3.8B SLM)**

**Controller sends structured message to Coding Agent:**

```json
{
  "agent": "coding_agent",
  "task": "generate_pipeline_code",
  "inputs": {
    "pipeline_plan": {...},  // from Planning Agent
    "retrieved_cases": [...],  // similar code patterns
    "engineered_features": "/sandbox/engineered_features.csv"
  }
}
```

**Coding Agent Processing (code generation, ~400 tokens):**
- Adapts code from retrieved cases to current dataset
- Generates complete pipeline:

```python
# Coding Agent Generated Pipeline Code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

# Load data
df = pd.read_csv('/sandbox/engineered_features.csv')

# Feature selection
X = df.drop('retention_status', axis=1)
y = df['retention_status']
mi_scores = mutual_info_classif(X, y)
selected_features = X.columns[mi_scores > 0.01].tolist()
X_selected = X[selected_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# Model training (XGBoost based on case similarity)
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

print(f"AUC: {auc:.3f}, Accuracy: {accuracy:.3f}")
```

**Coding Agent Response:**
```json
{
  "status": "success",
  "pipeline_code": "...",
  "code_path": "/sandbox/generated_pipeline.py",
  "estimated_runtime": "2.3 minutes"
}
```

---

### **STEP 7: Code Execution & Validation**

**Controller Action:** Executes code in sandboxed environment

**Execution Result:**
```json
{
  "status": "success",
  "execution_time": "2.1 minutes",
  "initial_metrics": {
    "AUC": 0.84,
    "accuracy": 0.79
  }
}
```

**Controller Decision:** Performance meets threshold (AUC > 0.80), proceeds to hyperparameter tuning

---

### **STEP 8: Hyperparameter Tuning (Hyperparameter Agent 2B SLM)**

**Controller sends structured message to Hyperparameter Agent:**

```json
{
  "agent": "hyperparameter_agent",
  "task": "optimize_hyperparameters",
  "inputs": {
    "model_type": "XGBoost",
    "current_performance": {"AUC": 0.84},
    "search_budget": 50
  }
}
```

**Hyperparameter Agent Processing (optimization, ~200 tokens):**
- Generates hyperparameter search space
- Uses Bayesian optimization (learned from training data)

**Hyperparameter Agent Generated Code:**
```python
from skopt import gp_minimize
from skopt.space import Real, Integer

space = [
    Integer(50, 300, name='n_estimators'),
    Real(0.01, 0.3, name='learning_rate'),
    Integer(3, 10, name='max_depth'),
    Real(0.5, 1.0, name='subsample')
]

def objective(params):
    model = xgb.XGBClassifier(
        n_estimators=params[0],
        learning_rate=params[1],
        max_depth=params[2],
        subsample=params[3],
        objective='binary:logistic',
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return -roc_auc_score(y_test, y_pred_proba)  # Minimize negative AUC

result = gp_minimize(objective, space, n_calls=50, random_state=42)
best_params = result.x
```

**Execution Result:**
```json
{
  "status": "success",
  "best_hyperparameters": {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8
  },
  "improved_metrics": {
    "AUC": 0.87,
    "accuracy": 0.82
  }
}
```

---

### **STEP 9: Evaluation & Report Generation (Evaluator Agent 1.5B SLM)**

**Controller sends structured message to Evaluator Agent:**

```json
{
  "agent": "evaluator_agent",
  "task": "generate_evaluation_report",
  "inputs": {
    "model_performance": {
      "AUC": 0.87,
      "accuracy": 0.82,
      "precision": 0.85,
      "recall": 0.78
    },
    "pipeline_stages": [...],
    "feature_importance": {...}
  }
}
```

**Evaluator Agent Processing (report generation, ~300 tokens):**
- Analyzes performance metrics
- Identifies key features
- Generates interpretable report

**Evaluator Agent Output:**
```json
{
  "status": "success",
  "evaluation_report": {
    "performance_summary": {
      "AUC": 0.87,
      "accuracy": 0.82,
      "precision": 0.85,
      "recall": 0.78,
      "f1_score": 0.81
    },
    "top_features": [
      {"feature": "gpa", "importance": 0.32},
      {"feature": "days_since_enrollment", "importance": 0.18},
      {"feature": "first_gen_major_interaction", "importance": 0.15}
    ],
    "recommendations": [
      "Model shows strong performance (AUC: 0.87)",
      "GPA is the most predictive feature",
      "Consider collecting more temporal data for improved predictions"
    ]
  }
}
```

---

### **STEP 10: Response Generation (HCI SLM 3.8B)**

**Controller sends structured message to HCI SLM for final response:**

```json
{
  "agent": "hci_agent",
  "task": "generate_user_response",
  "inputs": {
    "user_query": "I need to build a machine learning model to predict which students are at risk of dropping out...",
    "workflow_results": {
      "pipeline_created": true,
      "performance_metrics": {
        "AUC": 0.87,
        "accuracy": 0.82
      },
      "top_features": ["gpa", "days_since_enrollment", "first_gen_major_interaction"],
      "pipeline_code": "/sandbox/generated_pipeline.py",
      "execution_time": "4.2 minutes"
    }
  }
}
```

**HCI SLM Processing (natural language generation, ~400 tokens):**
- Synthesizes all workflow results
- Generates human-readable response
- Formats in conversational style

**HCI SLM Final Response to User:**
```
✅ **ML Pipeline Successfully Created!**

I've built a machine learning pipeline to predict student retention risk. Here's what was accomplished:

**Performance Results:**
- AUC Score: 0.87 (excellent predictive power)
- Accuracy: 82%
- The model correctly identifies 85% of at-risk students (precision: 0.85)

**Key Insights:**
The most important factors for predicting dropout risk are:
1. **GPA** (32% importance) - Students with lower GPAs are at higher risk
2. **Days Since Enrollment** (18% importance) - Early intervention is critical
3. **First-Generation Major Interaction** (15% importance) - First-gen students in difficult majors face higher risk

**Pipeline Details:**
- Used XGBoost classifier with optimized hyperparameters
- Engineered 8 new features including temporal and interaction features
- Trained on 40,000 students, tested on 10,000 students
- Total execution time: 4.2 minutes

**Next Steps:**
The pipeline code is saved at `/sandbox/generated_pipeline.py`. You can:
- Deploy this model for real-time predictions
- Retrain periodically with new data
- Fine-tune hyperparameters for your specific institution

Would you like me to explain any specific part of the pipeline or help you deploy it?
```

---

## Architecture Benefits Demonstrated

### **1. Separation of Concerns**
- **HCI SLM:** Only handles conversation (understanding queries, generating responses)
- **Controller:** Manages workflow coordination, state, and routing
- **Specialized SLMs:** Focus on specific tasks (data cleaning, feature engineering, code generation)

### **2. Cost Efficiency**
- **HCI SLM (3.8B):** ~400 tokens for conversation
- **Specialized Agents (1.5B-3.8B):** Focused context (150-400 tokens each)
- **Total Cost:** ~10-30× lower than single LLM processing entire workflow

### **3. Parallel Execution**
- Data Agent and Case Retriever run simultaneously (no dependencies)
- Controller manages parallelism efficiently

### **4. Transparency & Debuggability**
- Each agent maintains focused decision logs
- Controller tracks all agent interactions
- Easy to identify which agent caused issues

### **5. Modularity**
- Each agent can be updated independently
- New feature engineering technique → update Feature Agent only
- No need to retrain entire system

---

## Controller State Management

Throughout the workflow, the Controller maintains:

```json
{
  "workflow_id": "retention_pipeline_2025_01_15_001",
  "user_query": "I need to build a machine learning model...",
  "intent": {
    "task_type": "classification",
    "target": "student_retention"
  },
  "execution_state": {
    "current_stage": "evaluation",
    "completed_stages": [
      "data_cleaning",
      "case_retrieval",
      "feature_engineering",
      "pipeline_planning",
      "code_generation",
      "hyperparameter_tuning"
    ],
    "agent_results": {
      "data_agent": {...},
      "case_retriever": {...},
      "feature_agent": {...},
      ...
    }
  },
  "performance_metrics": {
    "total_latency": "4.2 minutes",
    "cost_estimate": "$0.05",
    "agent_calls": 8
  }
}
```

---

## Key Takeaways

1. **HCI SLM** handles all human communication - understands queries, generates friendly responses
2. **Controller** manages workflow - coordinates agents, maintains state, handles dependencies
3. **Specialized SLMs** execute tasks - each focused on specific domain expertise
4. **Separation** enables efficiency - conversation separate from execution logic
5. **Transparency** - each agent's role is clear, easy to debug and improve

This architecture demonstrates how controller-mediated systems with specialized SLMs can handle complex workflows efficiently while maintaining clear separation between human interaction and task execution.

