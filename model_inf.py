from collections import abc
import abc
import json
# import ollama  # Make sure ollama is installed: pip install ollama
from typing import List, Dict, Any, Optional, Type, Union
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

# Define the standard message format (common for both)
ChatMessage = Dict[str, str]  # e.g., {'role': 'user', 'content': '...'}

# Mapping for model names between ollama and vLLM
import json
with open("config/model_config.json", "r") as f:
    mapping_name_dict = json.load(f)


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
        # Optional: Check if ollama server is reachable on init
        ollama.list()
        print("Ollama backend initialized successfully.")
        

    def chat(self, model: str, messages: List[ChatMessage]) -> str:
        """Sends messages via ollama.chat and returns the content string."""        
        
        if model in mapping_name_dict:
            model = mapping_name_dict[model]["ollama"]
        else:
            raise ValueError(
                f"Model '{model}' not found in conversion map. Available models: {mapping_name_dict.keys()}"
            )
        response = ollama.chat(
            model=model,
            messages=messages,
            options=self.ollama_options,  # Pass options here
        )
        # Standard ollama response format: {'message': {'role': 'assistant', 'content': '...'}}
        return response["message"]["content"]


class VLLMClient(LLMInterface):
    """
    LLMInterface implementation using vLLM's OpenAI-compatible API.
    Assumes vLLM is running and serving models via an endpoint like http://localhost:8000/v1
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8010/v1",
        api_key: str = "dummy",
        options: Dict[str, Any] = None,
    ):
        """
        Initializes the vLLM client via its OpenAI-compatible API.
        Args:
            base_url: The URL of the vLLM OpenAI-compatible server endpoint.
            api_key: API key (usually not required by vLLM, can be 'dummy' or 'no-key').
            client_options: Optional dictionary of generation parameters for the OpenAI API call
                            (e.g., {'temperature': 0.7, 'max_tokens': 500}).
        """
        self.client_options = options or {}
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        # Optional: Check connection on init
        self.client.models.list()  # Attempts to list models from the endpoint
        print(
            f"vLLM backend initialized successfully via OpenAI API at: {base_url}"
        )

    def chat(self, model: str, messages: List[ChatMessage]) -> str:
        """Sends messages via OpenAI client to vLLM and returns the content string."""
        # Convert model name if necessary
        print(mapping_name_dict[model])
        if model in mapping_name_dict:
            model = mapping_name_dict[model]["vllm"]
        else:
            raise ValueError(
                f"Model '{model}' not found in conversion map. Available models: {mapping_name_dict.keys()}"
            )
        # Call the OpenAI API with the vLLM server
        if "qwen" in model.lower():
            opts = dict(self.client_options)
            extra = dict(opts.get("extra_body") or {})
            chat_kwargs = dict(extra.get("chat_template_kwargs") or {})
            chat_kwargs["enable_thinking"] = False
            extra["chat_template_kwargs"] = chat_kwargs
            opts["extra_body"] = extra
        else:
            opts = dict(self.client_options)
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **opts,  # Pass generation parameters here
        )
        # Standard OpenAI response format
        return response.choices[0].message.content

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
