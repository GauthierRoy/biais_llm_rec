from collections import abc
import ollama
import abc
import json
import ollama # Make sure ollama is installed: pip install ollama
from openai import OpenAI # Make sure openai is installed: pip install openai
from typing import List, Dict, Any, Optional, Type, Union
from pydantic import BaseModel
from tqdm import tqdm

# Define the standard message format (common for both)
ChatMessage = Dict[str, str] # e.g., {'role': 'user', 'content': '...'}

class LLMInterface(abc.ABC):
    """Abstract base class for LLM inference clients."""

    @abc.abstractmethod
    def chat(self, model: str, messages: List[ChatMessage]) -> str:
        """
        Sends messages to the specified LLM model and returns the text response.

        Args:
            model: The identifier of the model to use.
            messages: A list of message dictionaries (e.g., [{'role': 'user', 'content': '...'}])

        Returns:
            The response content as a string.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
            Exception: Can raise exceptions from the underlying library on API errors.
        """
        pass
class OllamaClient(LLMInterface):
    """LLMInterface implementation using the ollama library."""
    def __init__(self, ollama_options: Dict[str, Any] = None):
        """
        Initializes the Ollama client.
        Args:
            ollama_options: Optional dictionary of options for ollama.chat (e.g., {'temperature': 0.7}).
        """
        self.ollama_options = ollama_options or {}
        try:
            # Optional: Check if ollama server is reachable on init
            ollama.list()
            print("Ollama backend initialized successfully.")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama during init. Ensure Ollama server is running. Error: {e}")


    def chat(self, model: str, messages: List[ChatMessage]) -> str:
        """Sends messages via ollama.chat and returns the content string."""
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                options=self.ollama_options # Pass options here
            )
            # Standard ollama response format: {'message': {'role': 'assistant', 'content': '...'}}
            return response['message']['content']
        except Exception as e:
            print(f"Error during Ollama chat for model '{model}': {e}")
            # Return empty string or re-raise, depending on desired error handling
            # Returning empty string allows the loop to continue potentially.
            return ""
            # Or uncomment below to stop execution on error
            # raise e

class VLLMClient(LLMInterface):
    """
    LLMInterface implementation using vLLM's OpenAI-compatible API.
    Assumes vLLM is running and serving models via an endpoint like http://localhost:8000/v1
    """
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "dummy", client_options: Dict[str, Any] = None):
        """
        Initializes the vLLM client via its OpenAI-compatible API.
        Args:
            base_url: The URL of the vLLM OpenAI-compatible server endpoint.
            api_key: API key (usually not required by vLLM, can be 'dummy' or 'no-key').
            client_options: Optional dictionary of generation parameters for the OpenAI API call
                            (e.g., {'temperature': 0.7, 'max_tokens': 500}).
        """
        self.client_options = client_options or {}
        try:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
             # Optional: Check connection on init
            self.client.models.list() # Attempts to list models from the endpoint
            print(f"vLLM backend initialized successfully via OpenAI API at: {base_url}")
        except Exception as e:
            print(f"Warning: Could not connect to vLLM OpenAI API at '{base_url}' during init. Error: {e}")
            self.client = None # Mark client as unusable

    def chat(self, model: str, messages: List[ChatMessage]) -> str:
        """Sends messages via OpenAI client to vLLM and returns the content string."""
        if not self.client:
             print("Error: vLLM client not initialized properly.")
             return ""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **self.client_options # Pass generation parameters here
            )
            # Standard OpenAI response format
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during vLLM chat (via OpenAI API) for model '{model}': {e}")
            return ""
    
    # def chat_structured(
    #     self,
    #     model: str,
    #     messages: List[ChatMessage],
    #     output_schema: Type[BaseModel]
    # ) -> Optional[Dict[str, Any]]:
    #     """
    #     Generates a structured JSON response conforming to the provided Pydantic schema.

    #     Relies on vLLM server's 'guided_json' capability. Make sure your vLLM
    #     version supports this and is configured correctly if needed.

    #     Args:
    #         model: The identifier of the model to use (as known by vLLM).
    #         messages: A list of message dictionaries.
    #         output_schema: The Pydantic BaseModel class defining the desired output structure.

    #     Returns:
    #         A dictionary parsed from the JSON response conforming to the schema,
    #         or None if an error occurs or the response is not valid JSON.
    #     """
    #     if not self.client:
    #         print("vLLM client not initialized properly. Cannot perform structured chat.")
    #         return None

    #     # 1. Get the JSON schema from the Pydantic model
    #     json_schema = output_schema.model_json_schema()

    #     # 2. Call the API with the extra_body parameter
    #     # Combine general client options with the specific guided_json body
    #     # Note: Some options like temperature might behave differently with guided generation.
    #     api_params = {
    #         "model": model,
    #         "messages": messages,
    #         **self.client_options, # Include temperature, max_tokens etc. if set
    #         "extra_body": {"guided_json": json_schema} # Key vLLM parameter
    #     }

    #     response = self.client.chat.completions.create(**api_params)

    #     raw_response_content = response.choices[0].message.content

    #     if not raw_response_content:
    #         print("Received empty content in structured response.")
    #         return None

    #     # 3. Parse the JSON string response
    #     parsed_json = json.loads(raw_response_content)
    #     # Optional: Validate against the Pydantic model rigorously
    #     # try:
    #     #    validated_model = output_schema.model_validate(parsed_json)
    #     #    return validated_model.model_dump() # Return dict from validated model
    #     # except ValidationError as val_err:
    #     #    logging.error(f"LLM output failed Pydantic validation for schema {output_schema.__name__}: {val_err}\nContent: {raw_response_content}")
    #     #    return None # Or return the raw parsed_json if partial validation is ok?
    #     return parsed_json # Return the parsed dict directly for now
