import sagemaker
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.sklearn.model import SKLearnModel

session = sagemaker.Session()
bucket = session.default_bucket()
role = sagemaker.get_execution_role()

# 1. Upload model archive to S3
s3_uri = session.upload_data(
    path="model.tar.gz", bucket=bucket, key_prefix="house-valuation-model"
)

# 2. Define Scikit-Learn Model
model = SKLearnModel(
    model_data=s3_uri,
    role=role,
    entry_point="inference.py",
    framework_version="1.2-1",
    py_version="py3",
    name="house-valuation-sklearn-model",
)

# 3. Deploy to Serverless Endpoint
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=1024, max_concurrency=5
)
predictor = model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name="property-valuation-endpoint",
)
print(f"Endpoint active: {predictor.endpoint_name}")
