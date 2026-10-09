"""
agent_reasoning.py
Provides Amazon Bedrock consultative agent capabilities for the Property Valuation Engine.
"""

import json
import boto3

def get_bedrock_client(region_name="us-east-1"):
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=region_name
    )

def check_layout_heuristics(specs: dict) -> list:
    """Detects layout imbalances and functional obsolescence."""
    warnings = []
    bedrooms = specs.get("bedrooms", 1)
    bathrooms = specs.get("bathrooms", 1)
    sqft = specs.get("living_area_sqft", 1000)

    if bedrooms >= 5 and bathrooms <= 1.5:
        warnings.append(
            f"Functional Obsolescence Penalty: {bedrooms} bedrooms with only {bathrooms} bathroom(s) "
            "severely limits buyer marketability. Adding a bathroom baseline would recover an estimated 8–12% in value."
        )

    if sqft / max(bedrooms, 1) < 250:
        warnings.append("High Bedroom Density: Living area per bedroom is under 250 sqft, which may reduce perceived luxury tier.")

    return warnings

def generate_agent_explanation(specs: dict, estimated_price: float, client=None) -> str:
    """Calls Amazon Bedrock to explain the valuation rationale in clear, human language."""
    if client is None:
        client = get_bedrock_client()

    heuristics = check_layout_heuristics(specs)
    heuristics_text = "\n".join(heuristics) if heuristics else "No severe layout penalties detected."

    prompt = f"""
You are an expert real estate appraisal advisor. Explain the following property appraisal to a client in a clear, encouraging, and transparent tone.

Property Details:
- Living Area: {specs.get('living_area_sqft')} sqft
- Lot Size: {specs.get('lot_size_sqft')} sqft
- Bedrooms: {specs.get('bedrooms')} | Bathrooms: {specs.get('bathrooms')} | Floors: {specs.get('floors')}
- Age: {specs.get('property_age')} years
- Build Quality (1-10): {specs.get('build_quality')}
- Renovation Tier (0-10): {specs.get('renovation_tier')}
- Neighborhood Score: {specs.get('neighborhood_rating')}/10
- School Quality: {specs.get('school_quality')}/10
- Parks & Greenery: {specs.get('parks_proximity')}/10
- Duplex: {'Yes' if specs.get('is_duplex') else 'No'}

Model Valuation Result: ${estimated_price:,.2f}
Layout Checks:
{heuristics_text}

Provide:
1. A concise, consultative breakdown of what drove this valuation (e.g., impact of renovation tier, schools, or square footage).
2. Actionable advice on how the owner or investor could unlock higher equity.
Keep the tone professional, friendly, and under 150 words.
"""

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 400,
        "temperature": 0.4,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps(payload)
        )
        body_response = json.loads(response["body"].read())
        return body_response["content"][0]["text"]
    except Exception as e:
        # Fallback explanation if Bedrock invocation fails or offline
        fallback_msg = f"Valuation complete: ${estimated_price:,.2f}.\n\n"
        if heuristics:
            fallback_msg += f"Advisor Note: {heuristics[0]}\n\n"
        fallback_msg += (
            f"Key value drivers include a Build Quality score of {specs.get('build_quality')}/10 "
            f"and Renovation Tier {specs.get('renovation_tier')}/10 in a neighborhood rated {specs.get('neighborhood_rating')}/10."
        )
        return fallback_msg
