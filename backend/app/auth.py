# backend/app/auth.py

"""
Authentication module for KAIROS API
Handles Bearer token authentication for HackRx competition and general API access.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, Security, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from functools import wraps
import hashlib
import time
import os

logger = logging.getLogger(__name__)

# Competition token from HackRx requirements
HACKRX_TOKEN = os.getenv("HACKRX_TOKEN", "9e7c168e23eaafaf430e54afa3c6e922a0cea39395cc06796e02aa0a7962f31f")

# Additional tokens for different access levels (optional)
API_TOKENS = {
    HACKRX_TOKEN: {
        "name": "HackRx Competition",
        "level": "competition",
        "rate_limit": 100,  # requests per hour
        "expires": None,    # No expiration
        "permissions": ["hackrx", "documents", "health"]
    },
    # Add more tokens as needed for different users/services
}

# Rate limiting storage (in production, use Redis)
rate_limit_storage: Dict[str, Dict[str, Any]] = {}

# Security scheme
security = HTTPBearer(
    scheme_name="Bearer Token",
    description="Provide your API token in the format: Bearer <token>"
)

class AuthenticationError(HTTPException):
    """Custom authentication error."""
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"}
        )

class RateLimitError(HTTPException):
    """Rate limit exceeded error."""
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": "3600"}  # 1 hour
        )

def get_token_info(token: str) -> Optional[Dict[str, Any]]:
    """Get information about a token."""
    return API_TOKENS.get(token)

def check_rate_limit(token: str, endpoint: str = "general") -> bool:
    """
    Check if the token has exceeded its rate limit.
    
    Args:
        token: The API token
        endpoint: The endpoint being accessed
        
    Returns:
        True if within rate limit, False if exceeded
    """
    token_info = get_token_info(token)
    if not token_info:
        return False
    
    rate_limit = token_info.get("rate_limit", 60)  # Default 60 requests/hour
    current_time = time.time()
    hour_ago = current_time - 3600  # 1 hour ago
    
    # Create storage key
    storage_key = f"{hashlib.md5(token.encode()).hexdigest()}:{endpoint}"
    
    # Initialize if not exists
    if storage_key not in rate_limit_storage:
        rate_limit_storage[storage_key] = {
            "requests": [],
            "first_request": current_time
        }
    
    # Clean old requests (older than 1 hour)
    request_times = rate_limit_storage[storage_key]["requests"]
    rate_limit_storage[storage_key]["requests"] = [
        req_time for req_time in request_times if req_time > hour_ago
    ]
    
    # Check if rate limit exceeded
    current_requests = len(rate_limit_storage[storage_key]["requests"])
    if current_requests >= rate_limit:
        logger.warning(f"Rate limit exceeded for token ending in ...{token[-8:]} on {endpoint}")
        return False
    
    # Add current request
    rate_limit_storage[storage_key]["requests"].append(current_time)
    return True

def log_auth_attempt(token: str, success: bool, endpoint: str, ip_address: str = None):
    """Log authentication attempts for monitoring."""
    token_hash = hashlib.md5(token.encode()).hexdigest()[:8]
    status_msg = "SUCCESS" if success else "FAILED"
    
    logger.info(
        f"AUTH {status_msg}: Token ...{token_hash} | "
        f"Endpoint: {endpoint} | IP: {ip_address or 'unknown'}"
    )

async def verify_hackrx_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    """Verify the HackRx competition token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = credentials.credentials
    
    # Check if token exists
    token_info = get_token_info(token)
    if not token_info:
        log_auth_attempt(token, False, "hackrx")
        raise AuthenticationError("Invalid authentication token")
    
    # Check token expiration
    if token_info.get("expires"):
        expire_time = datetime.fromisoformat(token_info["expires"])
        if datetime.utcnow() > expire_time:
            log_auth_attempt(token, False, "hackrx")
            raise AuthenticationError("Token has expired")
    
    # Check rate limiting
    if not check_rate_limit(token, "hackrx"):
        log_auth_attempt(token, False, "hackrx")
        raise RateLimitError(
            f"Rate limit exceeded. Maximum {token_info.get('rate_limit', 60)} requests per hour."
        )
    
    # Check permissions
    permissions = token_info.get("permissions", [])
    if "hackrx" not in permissions:
        log_auth_attempt(token, False, "hackrx")
        raise AuthenticationError("Token does not have hackrx permissions")
    
    log_auth_attempt(token, True, "hackrx")
    return token

async def verify_api_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
    required_permission: str = "general"
) -> str:
    """
    Verify general API token with permission checking.
    
    Args:
        credentials: The Bearer token credentials
        required_permission: Required permission level
        
    Returns:
        The validated token string
    """
    token = credentials.credentials
    
    # Check if token exists
    token_info = get_token_info(token)
    if not token_info:
        log_auth_attempt(token, False, required_permission)
        raise AuthenticationError("Invalid authentication token")
    
    # Check token expiration
    if token_info.get("expires"):
        expire_time = datetime.fromisoformat(token_info["expires"])
        if datetime.utcnow() > expire_time:
            log_auth_attempt(token, False, required_permission)
            raise AuthenticationError("Token has expired")
    
    # Check rate limiting
    if not check_rate_limit(token, required_permission):
        log_auth_attempt(token, False, required_permission)
        raise RateLimitError(
            f"Rate limit exceeded. Maximum {token_info.get('rate_limit', 60)} requests per hour."
        )
    
    # Check permissions
    permissions = token_info.get("permissions", [])
    if required_permission not in permissions and "admin" not in permissions:
        log_auth_attempt(token, False, required_permission)
        raise AuthenticationError(f"Token does not have {required_permission} permissions")
    
    log_auth_attempt(token, True, required_permission)
    return token

def optional_auth(required_permission: str = "general"):
    """
    Decorator for optional authentication.
    If token is provided, it must be valid. If not provided, continues without auth.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to get token from request
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request and "authorization" in request.headers:
                # Token provided, verify it
                auth_header = request.headers["authorization"]
                if auth_header.startswith("Bearer "):
                    token = auth_header[7:]
                    token_info = get_token_info(token)
                    if not token_info:
                        raise AuthenticationError("Invalid token")
                    
                    # Add token info to kwargs
                    kwargs["token_info"] = token_info
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def get_current_user_info(token: str) -> Dict[str, Any]:
    """
    Get information about the current user based on their token.
    
    Args:
        token: The authenticated token
        
    Returns:
        Dictionary with user information
    """
    token_info = get_token_info(token)
    if not token_info:
        return {"name": "Unknown", "level": "none"}
    
    return {
        "name": token_info.get("name", "Unknown"),
        "level": token_info.get("level", "user"),
        "permissions": token_info.get("permissions", []),
        "rate_limit": token_info.get("rate_limit", 60)
    }

def create_api_token(
    name: str,
    level: str = "user",
    permissions: list = None,
    rate_limit: int = 60,
    expires_in_days: int = None
) -> str:
    """
    Create a new API token (for admin use).
    
    Args:
        name: Name/description of the token
        level: Access level (user, admin, competition)
        permissions: List of allowed permissions
        rate_limit: Requests per hour limit
        expires_in_days: Days until expiration (None for no expiration)
        
    Returns:
        The generated token string
    """
    # Generate token
    token_data = f"{name}:{level}:{time.time()}"
    token = hashlib.sha256(token_data.encode()).hexdigest()
    
    # Set expiration
    expires = None
    if expires_in_days:
        expires = (datetime.utcnow() + timedelta(days=expires_in_days)).isoformat()
    
    # Store token info
    API_TOKENS[token] = {
        "name": name,
        "level": level,
        "permissions": permissions or ["general"],
        "rate_limit": rate_limit,
        "expires": expires,
        "created_at": datetime.utcnow().isoformat()
    }
    
    logger.info(f"Created new API token for {name} with level {level}")
    return token

def revoke_token(token: str) -> bool:
    """
    Revoke an API token.
    
    Args:
        token: The token to revoke
        
    Returns:
        True if token was revoked, False if not found
    """
    if token in API_TOKENS:
        token_name = API_TOKENS[token].get("name", "Unknown")
        del API_TOKENS[token]
        logger.info(f"Revoked API token for {token_name}")
        return True
    return False

def get_auth_stats() -> Dict[str, Any]:
    """
    Get authentication statistics.
    
    Returns:
        Dictionary with auth statistics
    """
    return {
        "total_tokens": len(API_TOKENS),
        "active_rate_limits": len(rate_limit_storage),
        "token_types": {
            info.get("level", "unknown"): sum(
                1 for token_info in API_TOKENS.values() 
                if token_info.get("level") == info.get("level", "unknown")
            )
            for info in API_TOKENS.values()
        }
    }

# Health check for auth system
async def auth_health_check() -> Dict[str, Any]:
    """Check the health of the authentication system."""
    return {
        "status": "healthy",
        "tokens_configured": len(API_TOKENS),
        "rate_limit_entries": len(rate_limit_storage),
        "hackrx_token_valid": HACKRX_TOKEN in API_TOKENS
    }