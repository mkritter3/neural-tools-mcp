"""
L9 2025 OAuth2/JWT Authentication Service for MCP Server
Production-grade authentication with token validation and refresh
"""

import os
import time
import jwt
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import redis.asyncio as redis
import secrets
import hashlib

logger = logging.getLogger(__name__)

class TokenValidationError(Exception):
    """Token validation failed"""
    pass

class AuthenticationService:
    """OAuth2/JWT authentication for MCP sessions"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.jwt_secret = os.getenv('JWT_SECRET', self._generate_secret())
        self.jwt_algorithm = 'HS256'
        self.access_token_lifetime = int(os.getenv('ACCESS_TOKEN_LIFETIME', '3600'))  # 1 hour
        self.refresh_token_lifetime = int(os.getenv('REFRESH_TOKEN_LIFETIME', '604800'))  # 7 days
        
        # L9 2025: API key-based auth for development
        self.api_keys = self._load_api_keys()
        
        # OAuth2 client credentials (if configured)
        self.oauth_client_id = os.getenv('OAUTH_CLIENT_ID')
        self.oauth_client_secret = os.getenv('OAUTH_CLIENT_SECRET')
        self.oauth_enabled = bool(self.oauth_client_id and self.oauth_client_secret)
        
    def _generate_secret(self) -> str:
        """Generate secure JWT secret if not provided"""
        secret = secrets.token_urlsafe(64)
        logger.warning("⚠️ Generated JWT secret - configure JWT_SECRET environment variable for production")
        return secret
    
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from environment or configuration"""
        api_keys = {}
        
        # L9 development key
        dev_key = os.getenv('MCP_DEV_API_KEY', 'dev-' + secrets.token_urlsafe(32))
        api_keys[dev_key] = {
            'name': 'L9 Development Key',
            'permissions': ['read', 'write', 'admin'],
            'rate_limit': 1000,  # Higher limit for dev
            'created_at': datetime.now().isoformat()
        }
        
        # Production keys from environment
        prod_keys = os.getenv('MCP_API_KEYS', '').split(',')
        for key in prod_keys:
            if key.strip():
                api_keys[key.strip()] = {
                    'name': 'Production API Key',
                    'permissions': ['read', 'write'],
                    'rate_limit': 60,
                    'created_at': datetime.now().isoformat()
                }
        
        logger.info(f"🔑 Loaded {len(api_keys)} API keys")
        return api_keys
    
    async def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return client info"""
        if not api_key:
            return None
            
        # Check local API keys
        key_info = self.api_keys.get(api_key)
        if key_info:
            return {
                'auth_type': 'api_key',
                'client_id': hashlib.sha256(api_key.encode()).hexdigest()[:16],
                'permissions': key_info['permissions'],
                'rate_limit': key_info['rate_limit'],
                'name': key_info['name']
            }
        
        # Check Redis for dynamic keys (if available)
        if self.redis_client:
            try:
                stored_info = await self.redis_client.get(f"api_key:{api_key}")
                if stored_info:
                    import json
                    return json.loads(stored_info)
            except Exception as e:
                logger.error(f"Redis API key lookup failed: {e}")
        
        return None
    
    def generate_jwt_token(self, client_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate JWT access and refresh tokens"""
        now = datetime.utcnow()
        
        # Access token payload
        access_payload = {
            'client_id': client_info['client_id'],
            'permissions': client_info['permissions'],
            'auth_type': client_info['auth_type'],
            'iat': now,
            'exp': now + timedelta(seconds=self.access_token_lifetime),
            'type': 'access'
        }
        
        # Refresh token payload
        refresh_payload = {
            'client_id': client_info['client_id'],
            'iat': now,
            'exp': now + timedelta(seconds=self.refresh_token_lifetime),
            'type': 'refresh'
        }
        
        access_token = jwt.encode(access_payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        refresh_token = jwt.encode(refresh_payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'Bearer',
            'expires_in': self.access_token_lifetime
        }
    
    def validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check token type
            if payload.get('type') != 'access':
                raise TokenValidationError("Invalid token type")
            
            # Check expiration
            if payload.get('exp', 0) < time.time():
                raise TokenValidationError("Token expired")
            
            return payload
            
        except jwt.InvalidTokenError as e:
            raise TokenValidationError(f"Invalid token: {e}")
    
    def refresh_jwt_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh JWT access token using refresh token"""
        try:
            payload = jwt.decode(refresh_token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Validate refresh token
            if payload.get('type') != 'refresh':
                raise TokenValidationError("Invalid refresh token type")
                
            if payload.get('exp', 0) < time.time():
                raise TokenValidationError("Refresh token expired")
            
            # Generate new access token
            client_info = {
                'client_id': payload['client_id'],
                'permissions': ['read', 'write'],  # Default permissions
                'auth_type': 'jwt_refresh'
            }
            
            return self.generate_jwt_token(client_info)
            
        except jwt.InvalidTokenError as e:
            raise TokenValidationError(f"Invalid refresh token: {e}")
    
    async def authenticate_request(self, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Authenticate incoming request using various methods"""
        
        # Try Authorization header (Bearer token or API key)
        auth_header = headers.get('authorization', '').strip()
        if auth_header:
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
                try:
                    # Try JWT validation
                    payload = self.validate_jwt_token(token)
                    return {
                        'client_id': payload['client_id'],
                        'permissions': payload['permissions'],
                        'auth_type': 'jwt'
                    }
                except TokenValidationError:
                    # Try as API key
                    return await self.validate_api_key(token)
            
            elif auth_header.startswith('ApiKey '):
                api_key = auth_header[7:]
                return await self.validate_api_key(api_key)
        
        # Try X-API-Key header
        api_key = headers.get('x-api-key')
        if api_key:
            return await self.validate_api_key(api_key)
        
        # Try query parameter (development only)
        # This would need to be implemented in the HTTP layer
        
        return None
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke JWT token (add to blacklist)"""
        if not self.redis_client:
            logger.warning("Token revocation requires Redis")
            return False
            
        try:
            # Decode to get expiration
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm], options={"verify_exp": False})
            exp = payload.get('exp', 0)
            
            if exp > time.time():
                # Add to blacklist with TTL
                ttl = int(exp - time.time())
                await self.redis_client.setex(f"token_blacklist:{token}", ttl, "revoked")
                logger.info(f"Token revoked for client {payload.get('client_id', 'unknown')}")
                return True
            
        except Exception as e:
            logger.error(f"Token revocation failed: {e}")
        
        return False
    
    async def is_token_revoked(self, token: str) -> bool:
        """Check if token is revoked"""
        if not self.redis_client:
            return False
            
        try:
            exists = await self.redis_client.exists(f"token_blacklist:{token}")
            return bool(exists)
        except Exception as e:
            logger.error(f"Token revocation check failed: {e}")
            return False
    
    def get_auth_stats(self) -> Dict[str, Any]:
        """Get authentication statistics"""
        return {
            'api_keys_configured': len(self.api_keys),
            'oauth_enabled': self.oauth_enabled,
            'jwt_algorithm': self.jwt_algorithm,
            'access_token_lifetime': self.access_token_lifetime,
            'refresh_token_lifetime': self.refresh_token_lifetime,
            'redis_enabled': self.redis_client is not None
        }


class SecurityMiddleware:
    """Security middleware for MCP requests"""
    
    def __init__(self, auth_service: AuthenticationService):
        self.auth_service = auth_service
        self.required_permissions = {
            'tools/list': ['read'],
            'tools/call': ['write'],
            'resources/list': ['read'],
            'resources/read': ['read'],
            'admin/*': ['admin']
        }
    
    async def check_permissions(self, client_info: Dict[str, Any], tool_name: str) -> bool:
        """Check if client has permission to call tool"""
        client_permissions = client_info.get('permissions', [])
        
        # Admin has all permissions
        if 'admin' in client_permissions:
            return True
        
        # Check specific tool permissions
        for pattern, required_perms in self.required_permissions.items():
            if self._matches_pattern(tool_name, pattern):
                return any(perm in client_permissions for perm in required_perms)
        
        # Default: read permission for unknown tools
        return 'read' in client_permissions
    
    def _matches_pattern(self, tool_name: str, pattern: str) -> bool:
        """Simple pattern matching for tool permissions"""
        if pattern.endswith('*'):
            return tool_name.startswith(pattern[:-1])
        return tool_name == pattern
    
    async def validate_request(self, request_data: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Validate incoming MCP request"""
        
        # Authenticate request
        client_info = await self.auth_service.authenticate_request(headers)
        if not client_info:
            raise TokenValidationError("Authentication required")
        
        # Check tool permissions
        method = request_data.get('method', '')
        if not await self.check_permissions(client_info, method):
            raise TokenValidationError(f"Insufficient permissions for {method}")
        
        return client_info