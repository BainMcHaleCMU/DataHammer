"""
Custom Settings Module

This module provides a custom implementation of the Settings class
for the agent framework.
"""

from typing import Any, Dict, Optional, Type
import logging


class Settings:
    """
    Global settings for the application.
    
    This class provides a centralized way to manage global settings
    and configuration options.
    """
    
    # Class-level attributes for global settings
    llm = None  # Global LLM instance
    embed_model = None  # Global embedding model
    
    # Additional settings with default values
    chunk_size = 1024
    chunk_overlap = 20
    
    @classmethod
    def configure(cls, **kwargs) -> None:
        """
        Configure global settings.
        
        Args:
            **kwargs: Settings to configure
        """
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                logging.warning(f"Unknown setting: {key}")
    
    @classmethod
    def get_settings(cls) -> Dict[str, Any]:
        """
        Get all current settings.
        
        Returns:
            Dict of all settings
        """
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith("__") and not callable(value)
        }
    
    @classmethod
    def reset(cls) -> None:
        """Reset all settings to their default values."""
        cls.llm = None
        cls.embed_model = None
        cls.chunk_size = 1024
        cls.chunk_overlap = 20