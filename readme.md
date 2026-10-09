# 🏠 Advanced ML Property Valuation Engine & Agent

An AI-powered real estate appraisal engine delivering high-precision property valuations with transparent, consultative reasoning. Built for the [AWS Builder Center Weekend Challenge](https://builder.aws.com/content/3JnXmg1PikRXrSSDFHWEOw5RAur/weekend-challenge-build-an-agent-people-actually-enjoy-using) (`#agents`).

---

## 🚀 Live Demo & Article
* **Live Web Application:** [jainovation.xyz](https://www.jainovation.xyz/)


---

## 🏗️ Architecture & Agent Workflow

The solution pairs machine learning numerical inference with an intelligent Amazon Bedrock consultative agent:

* **Machine Learning Inference Core:** Executes multi-variable regression on 15 core property parameters using Scikit-Learn (`property_valuation_model.pkl`), delivering baseline estimates and confidence intervals (-MAE / +MAE).
* **Amazon Bedrock Agent Layer (`agent_reasoning.py`):** Acts as a real estate advisor, translating raw valuation numbers into plain-English market driver breakdowns, layout sanity checks, and equity-growth recommendations.
* **Streamlit UI (`opp.py`):** Responsive dark-mode dashboard with interactive sliders for immediate parameter exploration and scenario analysis.
* **AWS Cloud Infrastructure:** Designed for serverless deployment across AWS Lambda, Amazon API Gateway, Amazon S3, and Amazon Route 53.

[ Interactive Web UI ]
│ (Adjust layout, age, and amenity parameters)
▼
[ Streamlit App: opp.py ]
│
├──► [ ML Inference Core ] ──► Computes point estimate & bounds
│
└──► [ Amazon Bedrock Agent: agent_reasoning.py ] ──► Explains valuation rationale & detects layout penalties


---

## 📋 Evaluated Parameters (15 Features)

1. **Dimensions & Core Layout:** Living area (sqft), lot size (sqft), bedrooms, bathrooms, floors, fireplace count.
2. **Structural Quality & Modernity:** Property age, construction build quality (1–10), renovation tier (0–10), duplex flag.
3. **Location & Environmental Amenities:** Neighborhood rating, public school quality, municipal water infrastructure reliability, green space/park proximity.

---

## 🛠️ Local Setup & Execution

### 1. Clone the Repository
```bash
git clone [https://github.com/harshj1214-hj/AWS_project.git](https://github.com/harshj1214-hj/AWS_project.git)
cd AWS_project
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Configure AWS Credentials (for Amazon Bedrock)
Ensure your AWS environment has permission to invoke Bedrock models (bedrock:InvokeModel):

Bash
export AWS_DEFAULT_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
4. Run the Streamlit Application
Bash
streamlit run opp.py

📂 Repository Structure
opp.py - Main Streamlit user interface and valuation pipeline.

agent_reasoning.py - Amazon Bedrock agent integration and layout penalty heuristics.

property_valuation_model.pkl - Trained Scikit-Learn regression pipeline.

inference.py - SageMaker serverless inference script.

deploy.py - SageMaker deployment script.

requirements.txt - Project dependencies.
