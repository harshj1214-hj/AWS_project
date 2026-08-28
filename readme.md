# AI Property Valuation Pro

Automated machine learning engine delivering high-precision real estate appraisals using Amazon SageMaker Serverless Inference.

## 🚀 Live Demo
- **Web Application:** [https://www.jainovation.xyz/](https://www.jainovation.xyz/)

## 🏗️ AWS Architecture
- **Amazon S3:** Stores versioned `model.tar.gz` Scikit-Learn regression artifacts.
- **Amazon SageMaker Serverless Inference:** Hosts the auto-scaling prediction endpoint (`property-valuation-endpoint`) with zero idle cost.
- **Boto3 Runtime:** Facilitates low-latency communication between the Streamlit UI and SageMaker endpoints.

## 📊 Features & Model
Trained on 15 core parameters (dimensions, build age, structural quality, school ratings, and local amenities) using a log-transformed regression pipeline to stabilize variance and minimize MAE.
