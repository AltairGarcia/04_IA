"""Unit tests for error_handling.py module."""
import pytest
import time
from unittest.mock import patch, MagicMock
import requests
from error_handling import ErrorHandler, ErrorCategory


class TestErrorHandler:
    """Tests for the ErrorHandler class."""

    def test_categorize_error_api_error(self):
        """Test API error categorization."""
        error = requests.exceptions.HTTPError("500 Server Error")
        category, message = ErrorHandler.categorize_error(error)
        assert category == ErrorCategory.SERVER_API_ERROR
        assert "Server API error" in message

    def test_categorize_error_network(self):
        """Test network error categorization."""
        error = requests.exceptions.ConnectionError("Connection failed")
        category, message = ErrorHandler.categorize_error(error)
        assert category == ErrorCategory.NETWORK_ERROR
        assert "Connection failed" in message

    def test_categorize_error_validation(self):
        """Test validation error categorization."""
        error = ValueError("Invalid input format")
        category, message = ErrorHandler.categorize_error(error)
        assert category == ErrorCategory.VALIDATION_ERROR
        assert "Invalid input" in message

    def test_categorize_error_timeout(self):
        """Test timeout error categorization."""
        error = TimeoutError("Operation timed out")
        category, message = ErrorHandler.categorize_error(error)
        assert category == ErrorCategory.TIMEOUT_ERROR
        assert "timeout" in message.lower()

    def test_categorize_error_unknown(self):
        """Test unknown error categorization."""
        error = Exception("Some unexpected error")
        category, message = ErrorHandler.categorize_error(error)
        assert category == ErrorCategory.UNKNOWN_ERROR
        assert "Unknown error" in message

    def test_format_error_response(self):
        """Test formatting of error responses."""
        error = ValueError("Test error")
        context = {"user_id": "123", "action": "test"}
        response = ErrorHandler.format_error_response(error, context)

        assert response["success"] is False
        assert "Test error" in response["error"]
        assert response["error_type"] == ErrorCategory.VALIDATION_ERROR.value
        assert isinstance(response["timestamp"], float)
        assert response["context"] == context

    def test_format_error_response_no_context(self):
        """Test error response formatting without context."""
        error = Exception("Generic error")
        response = ErrorHandler.format_error_response(error)

        assert response["success"] is False
        assert "Generic error" in response["error"]
        assert response["error_type"] == ErrorCategory.UNKNOWN_ERROR.value
        assert isinstance(response["timestamp"], float)
        assert "context" not in response
