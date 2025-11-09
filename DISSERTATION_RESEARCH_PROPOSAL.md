# Dissertation Research Proposal: SLM-First Agentic Systems for Data Science and Institutional Analytics

**Author:** Isaac Kobby Anni  
**Institution:** Department of Computer Science  
**Degree:** Ph.D. in Data Science  
**Committee:** Dr. Md Main Uddin Rony (Chair), Prof. Dasigi, Prof. Robert Gree, Dr. Umar Islambekov, Dr. Sara Slitter

---

## Executive Summary

This dissertation proposes the development and evaluation of Small Language Model (SLM)-first agentic systems for data science and institutional analytics applications. Building upon recent advances in SLM capabilities (Belcak et al., 2025; Hu et al., 2024) and automated feature engineering (Hollmann et al., 2023), this research addresses critical gaps in deploying efficient, privacy-preserving, and cost-effective AI systems for educational institutions. The four interconnected projects demonstrate SLMs' viability for complex agentic tasks traditionally requiring large language models, while maintaining performance standards and addressing real-world constraints in data privacy, computational resources, and operational costs.

---

## Project 1: SLM-First Automated Feature Engineering (CAAFE-Style)

### Problem Setup and Motivation

Automated machine learning (AutoML) has made significant strides in model selection and hyperparameter optimization, yet data engineering—the most time-consuming aspect of data science—remains largely manual. Hollmann et al. (2023) demonstrated that large language models can automate feature engineering through code generation, but LLM-based approaches face critical limitations:

- **High operational costs**: LLM API calls are expensive for iterative feature engineering workflows
- **Privacy concerns**: Sensitive institutional data must be transmitted to external LLM services
- **Latency issues**: Real-time feature engineering suffers from API response delays
- **Complexity overhead**: Full LLM capabilities exceed the narrow scope of feature generation

We hypothesize that **SLM-first heterogeneous agentic systems** can match or exceed LLM performance for feature engineering tasks while offering significant advantages in privacy, cost, and latency.

### Methodology

#### System Architecture

Our proposed system employs a **controller-mediated heterogeneous architecture** (Belcak et al., 2025) where specialized SLMs handle different aspects of feature engineering:

```
Controller → Feature Type Detection → Task-Specific SLM Router
                                   ↓
    ┌──────────────────────────────┼──────────────────────────────┐
    │                              │                              │
  Date/Time                     Categorical                Numerical
  Feature SLM                    Feature SLM              Feature SLM
    │                              │                              │
    └──────────────────────────────┼──────────────────────────────┘
                                   ↓
                          Code Generation & Execution
                                   ↓
                         Performance Validation
                                   ↓
                          Accept/Reject Decision
```

#### Feature Engineering Specialists

1. **Temporal Features (1.5B SLM)**
   - Extract day-of-week, month, seasonality
   - Create time differences, duration calculations
   - Generate lag features, rolling statistics
   - **Example**: "days_since_enrollment", "semester_progress_pct"

2. **Categorical Encodings (1.5B SLM)**
   - Target encoding, frequency encoding
   - Interaction terms between categories
   - Rare category grouping
   - **Example**: "major_difficulty_interaction", "dept_popularity_score"

3. **Numerical Transformations (2B SLM)**
   - Polynomial features, binning operations
   - Statistical aggregations
   - Ratio and combination features
   - **Example**: "credit_hours_per_gpa", "course_load_difficulty"

4. **Escalation Gate (7B SLM or LLM)**
   - Handles complex multi-modal features
   - Handles ambiguous feature requests
   - Applies only when SLMs fail validation threshold (3 consecutive rejections)

#### Workflow

1. **Input Processing**
   - Dataset schema: column names, types, missing values
   - Domain context: "student retention prediction"
   - Sample rows for scale and distribution information

2. **Feature Generation Iteration**
   - Controller selects appropriate SLM based on feature type
   - SLM generates Python code with pandas operations
   - Safe execution in sandboxed environment
   - Cross-validation performance evaluation

3. **Acceptance Criteria**
   - Performance improvement: ROC AUC increase ≥ 0.02 OR
   - Interpretability enhancement: New feature provides domain insight
   - Ensemble contribution: Feature improves diverse models

4. **Escalation Strategy**
   - Track consecutive failures per feature type
   - Escalate to larger model after 3 failures
   - Cache successful patterns for future reuse

### Experimental Plan

#### Datasets

**Primary Institutional Data (Synthetic for Privacy)**
- Student retention dataset: 10,000 students, 50 features
- Course performance dataset: 50,000 enrollments, 30 features  
- Financial aid dataset: 5,000 applicants, 25 features

**Benchmark Datasets**
- OpenML: 20 tabular datasets (100-20,000 samples)
- UCI ML Repository: 15 classification datasets
- Kaggle competitions: 10 tabular datasets

#### Baseline Methods

1. **Traditional AutoFE**
   - Deep Feature Synthesis (DFS)
   - AutoFeat
   - No feature engineering

2. **LLM-Based**
   - CAAFE with GPT-4
   - CAAFE with GPT-3.5
   - OpenAI Function Calling API

3. **Our Method**
   - SLM-first heterogeneous system
   - Phi-3-mini (3.8B) as escalation gate
   - Specialized models for each task type

#### Evaluation Metrics

**Performance Metrics**
- ROC AUC improvement vs. baseline
- Feature count comparison
- Training/inference time (ms)
- Success rate (acceptances / total proposals)

**Economic Metrics**
- Cost per dataset (API calls + compute)
- Latency reduction vs. LLM baseline
- Energy consumption (FLOPs per feature)

**Qualitative Metrics**
- Feature interpretability scores
- Code quality (execution success rate)
- Domain expert validation

### Technology Stack

**SLM Models**
- Phi-3-mini (3.8B parameters) - Microsoft
- Qwen2-1.5B - Alibaba
- SmolLM2-1.7B - Hugging Face
- Custom fine-tuned variants on domain-specific data

**Infrastructure**
- **Training**: NVIDIA A100 (40GB) or H100 (80GB)
- **Inference**: Consumer GPUs (RTX 4090 24GB) for deployment
- **Fine-tuning**: LoRA/QLoRA for parameter-efficient adaptation
- **Deployment**: Edge devices for on-campus privacy-preserving execution

**Software Tools**
- PyTorch / Hugging Face Transformers
- Pandas / Polars for feature execution
- MLflow for experiment tracking
- FastAPI for REST API deployment
- Docker containers for reproducibility

---

## Project 2: SLM-First Knowledge Graph RAG for Student Retention

### Problem Setup and Motivation

Educational institutions generate massive amounts of structured and unstructured data across student records, course enrollments, academic performance, financial aid, housing, and support services. While traditional databases excel at transactional queries, they struggle with **complex analytical questions** that require traversing multiple relationships:

- "Which student populations show highest retention risk?"
- "What resource allocation strategies correlate with retention success?"
- "How do first-generation students' support usage patterns affect outcomes?"

Knowledge graphs (KGs) excel at representing such relationships, but existing RAG systems rely heavily on LLMs for both retrieval and generation, creating:

- **Privacy risks**: Institutional data transmitted to external APIs
- **Cost concerns**: Frequent LLM calls for routine queries
- **Complexity overkill**: LLMs' vast knowledge may interfere with institution-specific context

We propose an **SLM-first KG-RAG system** where knowledge graph traversal provides sufficient context that small models can generate accurate, verifiable insights.

### Methodology

#### System Architecture

```
                    Institutional RDBMS
                           ↓
                     ETL Pipeline
                           ↓
                    Knowledge Graph (Neo4j)
                           ↓
              Query Interface → SLM Router
                                ↓
                    ┌───────────┼───────────┐
                    │           │           │
            Simple Queries  Complex Queries  Causal Analysis
            (1.5B SLM)      (3.8B SLM)      (7B LLM)
                    │           │           │
                    └───────────┼───────────┘
                                ↓
                       KG-Traversed Context
                                ↓
                      Structured Insights
                                ↓
                     Human-Readable Reports
```

#### Knowledge Graph Construction

**Entities**
- Students: demographics, enrollment status, academic standing
- Courses: difficulty, prerequisites, department
- Resources: tutoring, counseling, financial aid
- Faculty: teaching load, student ratings, support availability
- Events: academic calendar, interventions, support services usage

**Relationships**
- ENROLLED_IN: student-course connections
- PREREQUISITE: course dependencies
- UTILIZED: student-resource usage
- TAUGHT_BY: course-faculty assignments
- PREDICTS: statistical correlations (e.g., "early warning indicators")

**Example Cypher Query for Retention Risk:**
```cypher
MATCH (s:Student)-[:ENROLLED_IN]->(c:Course)
       -[:PREREQUISITE*1..2]->(p:Course)
WHERE p.difficulty_score > 0.8
  AND s.credit_hours > 15
  AND s.generation_status = "First_Generation"
WITH s, count(c) as hard_courses
WHERE hard_courses >= 2
RETURN s, hard_courses
ORDER BY s.retention_probability ASC
LIMIT 50
```

#### Query Categories and SLM Routing

**Category 1: Pattern Discovery (1.5B SLM)**
- Count and aggregate queries
- Simple relationship traversals
- Pre-defined templates
- **Example**: "How many first-gen students take >18 credits?"

**Category 2: Descriptive Analysis (3.8B SLM)**
- Multi-hop relationship queries
- Contextual summaries with graphs
- Statistical comparisons
- **Example**: "Compare retention rates between housing types, accounting for major"

**Category 3: Causal Discovery (LLM escalation)**
- Complex reasoning about interventions
- Counterfactual analysis
- Requires external knowledge integration
- **Example**: "Would increasing early intervention improve retention for at-risk populations?"

#### RAG Process with SLMs

1. **User Query**: Natural language question
2. **Intent Classification**: Determine query category
3. **KG Traversal**: Execute Cypher query, retrieve relevant subgraph
4. **Context Assembly**:
   - Subgraph entities and relationships
   - Pre-computed statistics
   - Historical patterns and baselines
5. **SLM Generation**:
   - Prompts designed with KG structure in mind
   - Structured output formats (JSON, Markdown)
   - Citation back to source entities
6. **Verification**: 
   - Check claims against KG
   - Flag uncertainties for human review
7. **Escalation**: Route to LLM if confidence < threshold

### Experimental Plan

#### Data Sources (Synthetic Institutional Data)

**Student Dataset** (50,000 records)
- Demographics: age, gender, ethnicity, generation status
- Academic: GPA, credits, major, graduation status
- Financial: aid received, family income
- Behavioral: attendance, LMS activity, support utilization

**Course Dataset** (10,000 courses over 10 years)
- Course metadata: title, difficulty, prerequisites
- Enrollment: capacity, actual enrollment, waitlists
- Outcomes: pass rates, average grades by student type

**Resource Dataset** (1000+ resource types)
- Support services: tutoring, counseling, career services
- Utilization patterns: timing, frequency, student demographics
- Effectiveness metrics: follow-up success rates

#### Knowledge Graph Schema

**Nodes**
```python
class StudentNode:
    student_id: str
    name: str
    gpa: float
    credit_hours: int
    generation_status: str
    retention_risk: float

class CourseNode:
    course_id: str
    title: str
    difficulty_score: float
    credits: int
    department: str

class ResourceNode:
    resource_id: str
    type: str  # tutoring, counseling, etc.
    capacity: int
    availability: str
```

**Relationships**
- `ENROLLED_IN`: (Student, Course) with grade, semester
- `UTILIZED`: (Student, Resource) with frequency, outcome
- `CORRELATED_WITH`: (Resource, Retention) with evidence strength
- `PREREQUISITE`: (Course, Course) with requirement level

#### Baseline Comparisons

1. **LLM-Only RAG**
   - GPT-4 with vector embeddings
   - No knowledge graph structure
   - Generic retrieval-augmented generation

2. **Traditional KG Query System**
   - Cypher queries only
   - No NL generation
   - Requires technical users

3. **Hybrid LLM-KG**
   - GPT-4 with KG traversal
   - Full LLM reasoning

4. **Our Method**
   - SLM-first with KG traversal
   - Heterogeneous routing

#### Evaluation Metrics

**Retrieval Accuracy**
- Precision@K for relevant entities
- Recall of KG traversal paths
- Query success rate

**Generation Quality**
- ROUGE-L, BLEU vs. ground truth reports
- Factual accuracy vs. KG ground truth
- Citation correctness

**System Performance**
- Query latency (ms)
- Cost per query ($)
- Privacy preservation (local processing %)

### Technology Stack

**Knowledge Graph Platform**
- **Neo4j** (primary): Native graph database, Cypher queries
- **Alternative**: ArangoDB (multi-model with document support)
- **Visualization**: Neo4j Bloom for interactive exploration

**SLM Models**
- **Small (1.5B)**: SmolLM2-1.5B for simple queries
- **Medium (3.8B)**: Phi-3-mini for complex queries  
- **Large (7B)**: Mistral-7B as escalation gate
- **Fine-tuning**: LoRA on institutional query logs

**ETL and Integration**
- **Apache Airflow**: Scheduled ETL pipelines
- **DuckDB**: Fast analytical queries for KG population
- **Python**: pandas, networkx for graph construction

**Deployment**
- **Neo4j Aura**: Managed cloud (for development)
- **Self-hosted**: On-premise for production (privacy)
- **API**: FastAPI with GraphQL for flexible queries
- **Cache**: Redis for frequent query results

**References**
- [Neo4j Official Documentation](https://neo4j.com/docs/)
- [Neo4j + LLM Patterns (Community)](https://neo4j.com/developer-blog/integrating-large-language-models-with-neo4j/)
- [Graph RAG Architecture (Microsoft)](https://github.com/microsoft/graphrag)

---

## Project 3: Multi-Modal Institutional Analytics Agent

### Problem Setup and Motivation

Real-world institutional analytics requires integrating multiple data modalities: structured tabular data (grades, enrollment), unstructured text (student feedback, course descriptions), and temporal sequences (attendance patterns, engagement history). Current approaches either:

1. Process each modality in isolation, missing cross-modal insights
2. Use expensive LLMs for all tasks, creating privacy and cost issues
3. Require manual feature engineering for each data type

We propose an **SLM-first multi-modal agent** that orchestrates specialized models for different data types while maintaining privacy and cost efficiency.

### Methodology

#### System Architecture

```
Institutional Data Sources
        ↓
   Multi-Modal Router
        ↓
┌───────┼───────┬─────────────────┐
│       │       │                 │
Tabular Text  Temporal  Knowledge Graph
SLM     SLM    SLM      Traversal
│       │       │                 │
└───────┼───────┴─────────────────┘
        ↓
   Feature Fusion
        ↓
   Decision SLM
        ↓
   Actionable Insights
```

#### Modality-Specific Processors

**1. Tabular Data Agent (1.5B SLM)**
- Feature engineering from structured data
- Statistical summaries
- Relationship discovery
- **Input**: CSV/Parquet files (grades, enrollment)
- **Output**: Engineered features (e.g., "gpa_trend_slope")

**2. Text Analytics Agent (3.8B SLM)**
- Sentiment analysis from student feedback
- Topic modeling from course evaluations
- Named entity recognition (student names, course codes)
- **Input**: Text documents (reviews, surveys, emails)
- **Output**: Sentiment scores, extracted themes

**3. Temporal Sequence Agent (2B SLM)**
- Time-series analysis of attendance
- Pattern detection in engagement history
- Forecasting student trajectories
- **Input**: Time-stamped events (login, assignment submission)
- **Output**: Trajectory predictions, anomaly detection

**4. Knowledge Graph Agent**
- Cross-modal relationship discovery
- Graph-based insights from Project 2
- Entity resolution across modalities
- **Input**: All processed features
- **Output**: Unified entity view with relationships

**5. Orchestrator (3.8B SLM)**
- Coordinates all processors
- Fuses multi-modal features
- Generates final insights
- Escalates to LLM only for novel/unstructured queries

#### Integration Workflow

1. **Input Processing**
   - Parse multi-modal inputs
   - Assign to appropriate SLM based on data type
   - Extract metadata (timestamps, schemas)

2. **Parallel Processing**
   - Each SLM processes its modality independently
   - Results cached for reuse
   - Error handling: escalate to larger model if SLM fails

3. **Feature Fusion**
   - Combine outputs from all modalities
   - KG provides cross-modal relationship context
   - Temporal features align with structured data

4. **Decision Generation**
   - Orchestrator SLM synthesizes insights
   - Citation to source modalities
   - Uncertainty quantification
   - Actionable recommendations

### Experimental Plan

#### Dataset Construction

**Modality 1: Structured Tabular**
- Student grade records: 100,000 rows × 50 columns
- Enrollment history: 200,000 rows × 30 columns
- Financial aid: 50,000 rows × 40 columns

**Modality 2: Unstructured Text**
- Course evaluations: 10,000 reviews (avg 200 words)
- Student feedback forms: 15,000 responses
- Advisor meeting notes: 5,000 notes (semantic summaries)

**Modality 3: Temporal Sequences**
- Canvas LMS activity logs: 500,000 events
- Attendance records: 100,000 sessions
- Library usage: 50,000 check-in/out events

**Integration Point: Knowledge Graph**
- 50,000 students as central entities
- Relationships to all modalities
- Pre-computed cross-modal correlations

#### Query Types and Expected Outcomes

**Type 1: Student Success Prediction**
- Input: Grade history, course evaluations mentioning student, attendance patterns
- Output: Retention risk score with explanations
- Evaluation: Accuracy vs. actual retention outcomes

**Type 2: Resource Effectiveness Analysis**
- Input: Tutoring usage, student performance, support service logs
- Output: Effectiveness ranking with usage patterns
- Evaluation: Correlation with actual outcome improvements

**Type 3: Intervention Recommendation**
- Input: At-risk student profile (all modalities)
- Output: Specific support strategies with KG-based evidence
- Evaluation: Expert validation of recommendations

### Technology Stack

**SLM Models**
- **Tabular**: Phi-3-mini (3.8B) fine-tuned on structured data
- **Text**: Mistral-7B-Instruct for sentiment/topic modeling
- **Temporal**: Time-series specific model (optional pre-training)
- **Orchestrator**: Phi-3.5-mini (3.5B) for coordination

**Data Processing**
- **Tabular**: Polars (fast dataframe operations)
- **Text**: spaCy for NLP, sentence-transformers for embeddings
- **Temporal**: Prophet for forecasting, LSTM variants
- **Graph**: Neo4j from Project 2

**Infrastructure**
- **Training**: Multi-GPU setup (8× A100 40GB)
- **Inference**: Edge deployment with RTX 4090 (24GB)
- **Orchestration**: Kubernetes for multi-modal processing
- **Caching**: Redis for cross-modal feature caching

---

## Project 4: SLM-Based Causal Discovery for Institutional Interventions

### Problem Setup and Motivation

Determining causality in institutional settings is challenging due to complex relationships, hidden confounders, and ethical constraints on experimentation. Traditional causal inference requires expert knowledge to specify causal graphs (DAGs), yet domain experts may miss important connections, and LLMs may hallucinate relationships.

**The Challenge**:
- Observational data only (no random assignment)
- Complex confounders (socioeconomic status, student background)
- Need for explainable causal claims
- Resource constraints on data collection

**Our Approach**: 
Use SLMs to assist in **causal discovery and hypothesis generation** while leveraging knowledge graphs for structural constraints and interpretability.

### Methodology

#### Causal Discovery Pipeline

```
Observational Data → SLM-Assisted DAG Generation
                          ↓
                   KG Structure Priors
                          ↓
                    Causal Model Estimation
                          ↓
                 Effect Size Quantification
                          ↓
              SLM-Generated Explanations
```

#### Components

**1. SLM-Assisted Causal Graph Generation (3.8B SLM)**

The SLM helps generate and refine causal hypotheses:

- **Input**: Variables (e.g., "financial_aid", "gpa", "retention")
- **Task**: Generate plausible causal relationships
- **Constraints**: Use KG structure as prior knowledge
- **Output**: Causal graph hypotheses

**Example Prompt:**
```
Given variables: [financial_aid, gpa, attendance, retention]

Generate causal relationships considering:
- Temporal ordering (past events cause future events)
- Domain knowledge (aid enables enrollment, grades affect retention)
- Avoid spurious correlations (confounding variables)

Output: List of directed edges with justifications
```

**2. Knowledge Graph as Structural Prior**

Existing KG relationships inform causal graph:

- **Entity relationships** suggest causal pathways
- **Event ordering** provides temporal constraints
- **Statistical correlations** hint at potential causation

**Example**: 
If KG shows `UTILIZES(student, tutoring) → IMPROVES(grades)`, 
then suggest causal hypothesis: tutoring → gpa → retention

**3. Causal Model Estimation**

With proposed DAG, estimate causal effects:

- **Propensity Score Matching**: SLM helps identify control groups
- **Instrumental Variables**: SLM suggests potential instruments
- **Difference-in-Differences**: SLM identifies timing of interventions

**4. SLM Explanation Generation**

Generate human-readable causal narratives:

- Explain identified causal relationships
- Quantify effect sizes in interpretable terms
- Highlight confounders and limitations
- Provide evidence citations from KG

### Experimental Plan

#### Causal Questions

**Question 1**: Does early intervention improve retention?

- **Treatment**: Students identified early as at-risk
- **Outcome**: Retention after first year
- **Confounders**: Family income, major difficulty, campus resources
- **SLM Role**: Generate DAG hypotheses, suggest instruments

**Question 2**: What causal factors predict first-generation student success?

- **Variables**: Support utilization, peer networks, resource availability
- **Outcome**: GPA and retention
- **Confounders**: Pre-college preparation, financial stress
- **SLM Role**: Discover potential causal paths not visible to domain experts

**Question 3**: Does optimal course load allocation improve outcomes?

- **Treatment**: Recommended vs. actual course loads
- **Outcome**: Academic performance and well-being
- **Confounders**: Student capability, external stressors
- **SLM Role**: Suggest plausible mechanisms, estimate effects

#### Data Requirements

**Required Data Types**
1. **Longitudinal**: Student trajectories over 4+ years
2. **Quasi-experimental**: Natural variation (e.g., policy changes)
3. **Administrative**: Grades, enrollment, resource usage
4. **Surveys**: Self-reported stress, engagement, satisfaction

**Sample Size**: 
- Minimum 5,000 students with complete trajectories
- Stratified by demographics and program types
- Temporal span: 2018-2024 (6+ years)

#### Baseline Methods

1. **Expert-Defined DAGs**: Domain experts specify causal graphs
2. **Automated Causal Discovery**: PC-algorithm, FCI (no LLM)
3. **LLM-Assisted**: GPT-4 generates causal hypotheses
4. **Our Method**: SLM with KG priors

#### Evaluation Metrics

**Causal Discovery Accuracy**
- DAG structure precision/recall vs. expert ground truth
- Confounder detection rate
- Spurious edge identification

**Effect Estimation Quality**
- Effect size error vs. observed outcomes
- Statistical significance correctness
- Interpretability scores from domain experts

**System Performance**
- Causal graph generation time
- Hypothesis quality ratings
- Domain expert agreement scores

### Technology Stack

**Causal Inference Libraries**
- **DoWhy** (Microsoft): Causal reasoning framework
- **EconML**: Machine learning for causal effects
- **pgmpy**: Probabilistic graphical models

**SLM Models**
- **Phi-3-mini (3.8B)**: DAG generation, hypothesis formation
- **Fine-tuned variants**: Trained on causal reasoning datasets

**Knowledge Graph Integration**
- Neo4j from Project 2 provides structural priors
- Cypher queries to extract variable relationships

**Statistical Tools**
- **R/packages**: `dagR`, `pcalg` for causal discovery
- **Python**: `dowhy`, `econml` for estimation
- **Visualization**: GraphViz for DAG visualization

**References**
- [CausalML Documentation](https://causalml.readthedocs.io/)
- [DoWhy Framework](https://github.com/microsoft/dowhy)
- [Causal Discovery Methods Survey](https://arxiv.org/abs/2004.14741)

---

## Dissertation Timeline and Milestones

### Semester 1: Foundation and Project 1
**Spring 2026 (or Current Semester)**
- **Weeks 1-4**: Literature review, finalize project scope
- **Weeks 5-8**: SLM fine-tuning infrastructure setup, initial experiments
- **Weeks 9-12**: Project 1 implementation (SLM-first feature engineering)
- **Weeks 13-16**: Experiments on benchmark datasets, preliminary results

### Semester 2: Projects 2 and Project 1 Completion
**Fall 2026**
- **Weeks 1-4**: Project 1 evaluation, paper preparation, and submission
- **Weeks 5-8**: Knowledge graph construction and SLM-RAG development (Project 2)
- **Weeks 9-12**: Project 2 evaluation (KG-RAG for student retention)
- **Weeks 13-16**: Multi-modal agent development (Project 3) and initial integration

### Semester 3: Project 4 and Dissertation Completion
**Spring 2027**
- **Weeks 1-6**: Causal discovery framework development (Project 4)
- **Weeks 7-10**: Comprehensive evaluation across all four projects
- **Weeks 11-14**: Integration of insights across all projects
- **Weeks 15-16**: Dissertation writing, final revisions, and defense preparation

## Expected Deliverables

1. **Four prototype systems** (one per project)
2. **Comprehensive evaluation reports** comparing SLM-first vs. LLM baselines
3. **Three conference papers** (CHI, NeurIPS, or ICML targets)
4. **Open-source software** with institutional deployment documentation
5. **Knowledge graph datasets** (synthetic but realistic) for community use

## Broader Impact

This research contributes to democratizing access to AI capabilities through:
- **Privacy-preserving solutions** for sensitive institutional data
- **Cost-effective deployment** enabling resource-constrained institutions
- **Transparent and explainable** AI systems for educational stakeholders
- **Reusable frameworks** for other domains (healthcare, finance, etc.)

---

## References

### Core Papers
1. Belcak, P., et al. (2025). Small Language Models are the Future of Agentic AI. *arXiv preprint*.
2. Hollmann, N., Müller, S., & Hutter, F. (2023). Large Language Models for Automated Data Science: Introducing CAAFE. *NeurIPS 2023*.
3. Hu, S., et al. (2024). MiniCPM: Unveiling the Potential of Small Language Models. *arXiv:2404.06395*.

### Technology Resources
- [Neo4j Developer Blog](https://neo4j.com/developer-blog/)
- [Microsoft DoWhy](https://github.com/microsoft/dowhy)
- [Hugging Face Small Language Models](https://huggingface.co/collections)
- [MiniCPM GitHub Repository](https://github.com/OpenBMB/MiniCPM)

### Institutional Analytics
- [EDW Framework for Higher Education](https://www.educause.edu/)
- [Campus Analytics Best Practices](https://institutionalresearch.org/)

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Author Contact**: Isaac Kobby Anni - isaackobby.com

