"""
Wrapper to interact with OpenAI API. Current offering : GPT4o
"""
import os
import openai
import logging
import sys
from dotenv import load_dotenv
from openai import OpenAI
from openai import AzureOpenAI

class OpenAIAPI:
    """
    Wrapper Class to make API calls to GPT4o for text generation.
    """
    def __init__(self, model="gpt4o", loglevel="debug"):
        """
        Constructor sets the openai.api_key using either parameter passed or environment variable "OPENAI_API_KEY"
        Input :::: API_KEY : string (optional)
        Output :::: None
        """
        if loglevel == "debug":
            LOG_LEVEL=logging.DEBUG
        elif loglevel == "info":
            LOG_LEVEL=logging.INFO
        elif loglevel == "error":
            LOG_LEVEL=logging.ERROR
        else:
            LOG_LEVEL=logging.DEBUG

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(LOG_LEVEL)
        
        ch = logging.StreamHandler()
        ch.setLevel(LOG_LEVEL)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        self.logger.addHandler(ch)
        
        self.api_base = os.environ.get("AZURE_OPENAI_ENDPOINT", "")                
        self.logger.debug("API endpoint: %s", self.api_base)
        
        
        self.deployment_name = os.environ.get("OPENAI_MODEL") 
        self.logger.debug("deployment_name: %s", self.deployment_name)
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if self.api_key == "":
            t = self.deployment_name.split('-')
            self.deployment_name = '-'.join(t[:-3]+[''.join(t[-3:])])
            self.api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
            self.logger.info("Getting OpenAIAPI Key from environment variable : AZURE_OPENAI_API_KEY")
            self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "")
            self.logger.debug("api_version: %s", self.api_version)
            self.client = AzureOpenAI(
                api_key=self.api_key,  
                azure_endpoint = self.api_base,
                api_version=self.api_version
            )        
        else:
            self.logger.info("Getting OpenAIAPI Key from environment variable : OPENAI_API_KEY")
            self.client = OpenAI(
                api_key=self.api_key,  
            )        
        self.logger.debug("api_key: %s", self.api_key)        

    def make_call(self, messages, temperature, max_token=4096):
        """
        Function to make a call to davinci model for text generation with query prompt.
        Note : temperature, frequency_penalty and presence_penalty are model parameters that are not kept configurable currently.
        Input :::: prompt_str : string
        Output :::: resp_text : string
        """
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    max_tokens=max_token, 
                    temperature=temperature
                )
                if response.choices[0].finish_reason == 'content_filter':
                    out = ""
                else:
                    out = response.choices[0].message.content
                break
            except openai.BadRequestError as e:
                self.logger.error(f"OpenAI API returned an BadRequestError: {e}")
                out = ""
                break
            except openai.RateLimitError as e:
                self.logger.error("Didn't receive any response from OpenAI.")
                self.logger.error(e)
                self.logger.error("Try again.")
            except Exception as e:
                self.logger.error(e)
                sys.exit(1)
        
        return out
