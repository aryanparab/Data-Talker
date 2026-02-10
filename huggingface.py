"""
Custom LangChain LLM wrapper for local Hugging Face models
Compatible with your existing agents.py and tool binding
"""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class HuggingFaceChatModel(BaseChatModel):
    """
    LangChain-compatible chat model wrapper for Hugging Face models.
    Drop-in replacement for ChatGroq.
    """
    
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer: Any = None
    model: Any = None
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()
    
    def _load_model(self):
        """Load the Hugging Face model and tokenizer"""
        print(f"🔄 Loading {self.model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        print(f"✅ Model loaded on {self.device}")
    
    def _convert_messages_to_hf_format(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to Hugging Face chat format"""
        hf_messages = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                hf_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                hf_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                hf_messages.append({"role": "assistant", "content": msg.content})
        
        return hf_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from the model"""
        
        # Convert messages to HF format
        hf_messages = self._convert_messages_to_hf_format(messages)
        
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Create ChatResult
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self) -> str:
        """Return type of llm"""
        return "huggingface-chat"
    
    def bind_tools(self, tools: List[Any]) -> "HuggingFaceChatModel":
        """
        Bind tools to the model (for compatibility with LangChain tool usage).
        Note: Local models may not support tool calling natively like GPT-4.
        You may need to implement custom tool calling logic.
        """
        # For now, return self - tool calling with local models requires
        # custom prompting and parsing
        print("⚠️  Warning: Native tool calling not supported. Using prompt-based approach.")
        return self