import warnings
warnings.filterwarnings("ignore")

from typing import Any, Callable

from langchain_community.llms import VLLMOpenAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.core.prompts import PROMPTS
from app.core.setting import settings


LLM_Creator = Callable[[str, float, int, str, int], Any]


class LLM_Factory:
    LLM_Creators = {}

    @staticmethod
    def reg_creator(name: str, create_func: LLM_Creator):
        LLM_Factory.LLM_Creators[name] = create_func

    @staticmethod
    def get_creator(name: str) -> LLM_Creator:
        return LLM_Factory.LLM_Creators.get(name)


class LLM:

    @staticmethod
    def load_prompts():
        return PROMPTS

    Prompts = load_prompts()

    @staticmethod
    def get_llms():
        llm_flash = LLM_Factory.get_creator(settings.LLM_SERVING)(
            settings.LLM_FLASH_MODEL,
            settings.LLM_TEMP,
            settings.LLM_MAX_CTX,
            settings.SERVING_BASE_URL,
            settings.SERVING_MAX_RETRIES,
        )
        llm_thinking = LLM_Factory.get_creator(settings.LLM_SERVING)(
            settings.LLM_THINKING_MODEL,
            settings.LLM_TEMP,
            settings.LLM_MAX_CTX,
            settings.SERVING_BASE_URL,
            settings.SERVING_MAX_RETRIES,
            True,
        )
        return {
            'Flash': llm_flash,
            'Thinking': llm_thinking,
        }

    @staticmethod
    def init_openai_llm(model_name: str, temperature: float,
                        context_window: int, base_url: str, max_retries: int, cot: bool = False):
        kw = {
            'model': model_name,
            'base_url': base_url,
            'api_key': settings.SERVING_API_KEY,
            'max_retries': max_retries,
        }
        if not cot:
            kw['temperature'] = temperature
        return ChatOpenAI(**kw)

    @staticmethod
    def init_ollama_llm(model_name: str, temperature: float,
                        context_window: int, base_url: str, max_retries: int, cot: bool = False):
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            num_ctx=context_window,
            base_url=base_url,
        )

    @staticmethod
    def init_vllm_llm(model_name: str, temperature: float,
                      context_window: int, base_url: str, max_retries: int, cot: bool = False):
        return VLLMOpenAI(
            model_name=model_name,
            max_tokens=context_window,
            temperature=temperature,
            openai_api_key="EMPTY",
            openai_api_base=base_url,
        )

LLM_Factory.reg_creator('openai', LLM.init_openai_llm)
LLM_Factory.reg_creator('ollama', LLM.init_ollama_llm)
LLM_Factory.reg_creator('vllm', LLM.init_vllm_llm)
