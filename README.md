# llama2_Langchain_Sagemaker_Adapter
Short code snippets to adapt llama2 for use in langchain with Sagemaker

It seems like as of 07/18/2023, Langchain’s built-in SagemakerEndpoint class does not natively support Llama 2 model, mainly because

1) The Llama 2 model requires an extra custom attribute be passed into its input payload, which is a field signaling that the user has read and accepted End User Agreement.

2) The Llama 2 model expects a different format of input compared to Falcon series model. It expects input in this format, rather than a simple prompt as a string be used in “inputs” field.

        payload = {
            "inputs": [
                [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ]
            ],
            "parameters": self.parameters
        }

I wrote some codes that make changes to Langchain’s SagemakerEndpoint class on-the-fly, and wrote a LLamaContentHandler class that handles the special case for Llama model:

```python
# LangChain Declaration for SageMaker Endpoint
# Register transform_input and transform_output of ContentHandler as shown below by referring to Falcon's input and output .

from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from typing import Optional, List, Any
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.utils import enforce_stop_tokens

def new_call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
    _model_kwargs = self.model_kwargs or {}
    _model_kwargs = {**_model_kwargs, **kwargs}
    _endpoint_kwargs = self.endpoint_kwargs or {}

    body = self.content_handler.transform_input(prompt, _model_kwargs)
    content_type = self.content_handler.content_type
    accepts = self.content_handler.accepts

    # send request
    try:
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            Body=body,
            ContentType=content_type,
            Accept=accepts,
            CustomAttributes='accept_eula=true',  # Added this line
            **_endpoint_kwargs,
        )
    except Exception as e:
        raise ValueError(f"Error raised by inference endpoint: {e}")

    text = self.content_handler.transform_output(response["Body"])
    if stop is not None:
        text = enforce_stop_tokens(text, stop)

    return text

# Monkey patch the class
SagemakerEndpoint._call = new_call

class LLamaContentHandler(LLMContentHandler):
    def __init__(self, parameters):
        self.parameters = parameters
    
    @property
    def content_type(self):
        return "application/json"
    
    @property
    def accepts(self):
        return "application/json"
    
    def transform_input(self, prompt: str, model_kwargs: dict):
        system_content = "You are a helpful assistant."
        
        payload = {
            "inputs": [
                [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ]
            ],
            "parameters": self.parameters
        }
        
        return json.dumps(payload)
    
    def transform_output(self, response_body):
        # Load the response
        output = json.load(response_body)

        # Extract the 'content' value where the 'role' is 'assistant'
        user_response = next((item['generation']['content'] for item in output if item['generation']['role'] == 'assistant'), '')
        return user_response
```

I hope this helps!

Laiming Huang
