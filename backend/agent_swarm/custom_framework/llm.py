"""
Custom LLM Module

This module provides a custom implementation of the LLM (Language Learning Model) interface
for the agent framework.
"""

from typing import Any, Dict, List, Optional, Union
import os
import logging
import json
import google.generativeai as genai


class LLM:
    """
    Base Language Learning Model interface.
    
    This class defines the interface for language models and provides
    common functionality for all LLM implementations.
    """
    
    def __init__(self, model_name: Optional[str] = None, **kwargs):
        """
        Initialize the LLM.
        
        Args:
            model_name: Optional name of the model to use
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
    
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: The prompt to complete
            **kwargs: Additional completion parameters
            
        Returns:
            The generated completion text
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response to a conversation.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional chat parameters
            
        Returns:
            The generated response text
        """
        raise NotImplementedError("Subclasses must implement this method")


class Gemini(LLM):
    """
    Google Gemini implementation of the LLM interface.
    """
    
    def __init__(self, model: str = "models/gemini-2.0-flash-lite", **kwargs):
        """
        Initialize the Gemini LLM.
        
        Args:
            model: The Gemini model to use
            **kwargs: Additional model parameters
        """
        super().__init__(model_name=model, **kwargs)
        
        # Check for API key
        api_key = kwargs.get("api_key") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Get the model
        self.model = genai.GenerativeModel(model_name=model)
        
        self.logger.info(f"Initialized Gemini LLM with model: {model}")
    
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion using Gemini.
        
        Args:
            prompt: The prompt to complete
            **kwargs: Additional completion parameters
            
        Returns:
            The generated completion text
        """
        try:
            # Merge instance kwargs with method kwargs
            params = {**self.kwargs, **kwargs}
            
            # Generate the response
            response = self.model.generate_content(prompt, **params)
            
            # Extract and return the text
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating completion: {str(e)}")
            return f"Error: {str(e)}"
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a chat response using Gemini.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional chat parameters
            
        Returns:
            The generated response text
        """
        try:
            # Merge instance kwargs with method kwargs
            params = {**self.kwargs, **kwargs}
            
            # Convert messages to Gemini format
            gemini_messages = []
            for msg in messages:
                role = msg.get("role", "user").lower()
                content = msg.get("content", "")
                
                if role == "system":
                    # Gemini doesn't have a system role, so we'll add it as a user message
                    gemini_messages.append({"role": "user", "parts": [content]})
                    gemini_messages.append({"role": "model", "parts": ["I understand and will follow these instructions."]})
                else:
                    gemini_messages.append({"role": "user" if role == "user" else "model", "parts": [content]})
            
            # Generate the response
            chat = self.model.start_chat(history=gemini_messages[:-1] if gemini_messages else [])
            response = chat.send_message(gemini_messages[-1]["parts"][0] if gemini_messages else "", **params)
            
            # Extract and return the text
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating chat response: {str(e)}")
            return f"Error: {str(e)}"