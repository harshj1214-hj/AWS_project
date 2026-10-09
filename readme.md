# 🏠 Advanced ML Property Valuation Engine & Agent

An AI-powered real estate appraisal engine delivering high-precision property valuations with human-readable, transparent reasoning. Built for the AWS Builder Center Weekend Challenge (#agents).

## 🚀 Live Demo
* **Web Application:** [https://www.jainovation.xyz/](https://www.jainovation.xyz/)

---

## 🏗️ AWS Architecture

* **Amazon Bedrock:** Acts as the consultative agent layer, translating model outputs into plain-English valuation reasoning and layout recommendations.
* **Amazon SageMaker Serverless Inference / Scikit-Learn:** Executes fast, auto-scaling multi-variable regression on 15 core property parameters.
* **AWS Lambda & API Gateway:** Manages routing and inference orchestration between the UI and backend models.
* **Amazon S3:** Stores model artifacts (`model.tar.gz`) and deployment assets.
* **Streamlit UI:** Responsive dashboard featuring real-time sliders across dimensions, age, build quality, and neighborhood ratings.

---

## 📋 Evaluated Parameters
The model scores properties across 15 dimensions:
1. **Dimensions & Layout:** Living area (sqft), lot size, bedrooms, bathrooms, floors, fireplaces.
2. **Structural Quality & Modernity:** Property age, build quality (1–10), renovation tier (0–10), duplex flag.
3. **Location & Environmental Amenities:** Neighborhood rating, public school quality, water infrastructure reliability, proximity to parks.

---

## 🛠️ Local Setup & Deployment

```bash
git clone [https://github.com/harshj1214-hj/AWS_project.git](https://github.com/harshj1214-hj/AWS_project.git)
cd AWS_project
pip install -r requirements.txt
streamlit run app.py
