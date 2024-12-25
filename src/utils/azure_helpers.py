from typing import Dict
import os
from dotenv import load_dotenv

def setup_azure_credentials() -> Dict[str, str]:
    """
    Load Azure credentials from environment variables.
    Returns a dictionary with all necessary Azure credentials.
    """
    load_dotenv()
    
    required_vars = [
        'AD_DEPLOYMENT_ID',
        'AD_ENGINE',
        'AD_OPENAI_API_KEY',
        'AD_OPENAI_API_VERSION',
        'AD_OPENAI_API_BASE'
    ]
    
    credentials = {}
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            raise ValueError(f"Missing required environment variable: {var}")
        credentials[var] = value
        
    return credentials

def validate_azure_credentials(credentials: Dict[str, str]) -> bool:
    """
    Validate that all required Azure credentials are present and non-empty.
    
    Args:
        credentials: Dictionary of Azure credentials
        
    Returns:
        bool: True if all credentials are valid
        
    Raises:
        ValueError: If any credentials are missing or invalid
    """
    required_keys = {
        'AD_DEPLOYMENT_ID': "Deployment ID",
        'AD_ENGINE': "Engine name",
        'AD_OPENAI_API_KEY': "API key",
        'AD_OPENAI_API_VERSION': "API version",
        'AD_OPENAI_API_BASE': "API base URL"
    }
    
    for key, name in required_keys.items():
        if key not in credentials:
            raise ValueError(f"Missing {name} in Azure credentials")
        if not credentials[key]:
            raise ValueError(f"Empty {name} in Azure credentials")
            
    return True