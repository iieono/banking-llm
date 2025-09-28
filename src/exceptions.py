"""Custom exceptions for BankingLLM system.

This module provides a comprehensive exception hierarchy for proper error handling
throughout the banking intelligence system.
"""

from typing import Optional, Dict, Any


class BankingLLMError(Exception):
    """Base exception for all BankingLLM system errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "BANKING_ERROR"
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for structured logging."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }


class DatabaseError(BankingLLMError):
    """Database-related errors."""

    def __init__(
        self,
        message: str,
        database_name: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code="DB_ERROR", **kwargs)
        if database_name:
            self.details["database"] = database_name
        if query:
            self.details["query"] = query[:200] + "..." if len(query) > 200 else query


class DatabaseConnectionError(DatabaseError):
    """Database connection failures."""

    def __init__(self, message: str, database_name: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            database_name=database_name,
            error_code="DB_CONNECTION_ERROR",
            **kwargs
        )


class QueryExecutionError(DatabaseError):
    """SQL query execution failures."""

    def __init__(
        self,
        message: str,
        query: str,
        database_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            database_name=database_name,
            query=query,
            error_code="QUERY_EXECUTION_ERROR",
            **kwargs
        )


class LLMError(BankingLLMError):
    """LLM service related errors."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        user_query: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code="LLM_ERROR", **kwargs)
        if model_name:
            self.details["model"] = model_name
        if user_query:
            self.details["user_query"] = user_query[:100] + "..." if len(user_query) > 100 else user_query


class LLMServiceUnavailableError(LLMError):
    """LLM service is unavailable or not responding."""

    def __init__(self, message: str, service_url: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="LLM_SERVICE_UNAVAILABLE", **kwargs)
        if service_url:
            self.details["service_url"] = service_url


class SQLGenerationError(LLMError):
    """Failed to generate valid SQL from user query."""

    def __init__(self, message: str, user_query: str, raw_response: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            user_query=user_query,
            error_code="SQL_GENERATION_ERROR",
            **kwargs
        )
        if raw_response:
            self.details["raw_llm_response"] = raw_response[:200] + "..." if len(raw_response) > 200 else raw_response


class ValidationError(BankingLLMError):
    """Input validation errors."""

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        invalid_value: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        if field_name:
            self.details["field"] = field_name
        if invalid_value:
            self.details["invalid_value"] = str(invalid_value)[:100]


class SecurityError(BankingLLMError):
    """Security-related errors (SQL injection, unauthorized operations)."""

    def __init__(
        self,
        message: str,
        security_violation: Optional[str] = None,
        blocked_content: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code="SECURITY_ERROR", **kwargs)
        if security_violation:
            self.details["violation_type"] = security_violation
        if blocked_content:
            self.details["blocked_content"] = blocked_content[:100] + "..." if len(blocked_content) > 100 else blocked_content


class ExportError(BankingLLMError):
    """Excel export related errors."""

    def __init__(
        self,
        message: str,
        export_format: Optional[str] = None,
        file_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code="EXPORT_ERROR", **kwargs)
        if export_format:
            self.details["format"] = export_format
        if file_path:
            self.details["file_path"] = file_path


class ConfigurationError(BankingLLMError):
    """Configuration and settings related errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        if config_key:
            self.details["config_key"] = config_key
        if config_value:
            self.details["config_value"] = str(config_value)