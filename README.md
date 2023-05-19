# Assistant

This project is to tie together various LLM-related pieces, to try and build an AI assistant that can research and provide feedback on things I'm working on.
It is very heavily inspired by autogpt, but with a different purpose and structure and overall features

## How to Use

Checkout the code, and in the base directory run `docker compose build` (or your equivalent), then `docker compose up`.
It _should_ just work out of the box. However, the default agent setup is now pointing at Azure OpenAI so you'll need
keys for that, as well as for whatever tools are setup in the tools list.

If you have API keys for various things, you can look at the .env-template files - rename them to .env and add
keys or configurations as needed. You can find them in each of the sub-directories from the main dir. The critical one
is the one in `agent/`

To run the client, cd into the client/ directory and run `pip install -r requirements.txt` followed by `python ./client.py`

## Goals

An app that does the following:

  * Listens to mic and does speech-to-text
  * Speaks for responses
  * Ties together multiple LLM in a similar way to Jarvis
  * Can Google and do self-feedback in the way auto-gpt does
  * Pluggable models and a clear abstraction for them in the code, for future extensibility
  * A longer-term memory and the ability to use that memory with local models (ie. sidestep the current API limits)
  * Visual inputs from a camera and image recognition

Beyond the above, this repo is for experimentation and to help me understand things like fine-tuning and in-context learning

## Container List

`agent` is the nexus, runs the websocket and connections to the other models. This one is where
most of the development will happen, and will be the entry point for cli-based use.

`webui` provides a basic web interface, and sends messages to the gpt4all container

`dalai` is an alternate model for me to ensure that I've appropriate abstracted the models

`milvus` is a vector database I believe I'll eventually need as a memory - it's a replacement for pinecone in the autogpt setup

## Process

Most of the code you'll find in here is either from other projects, or from ChatGPT or CoPilot.
Other projects that have been an inspiration include:

  * AutoGPT
  * Dalai
  * Milvus
  * https://github.com/Venthe/chatgpt4all-webui
  * https://github.com/randaller/llama-chat
  * https://open-assistant.io/


## Plans

* Build the app out to be a basic web interface for chat, and make that interface easy to use as an API
* Need to decide on the base model and how it gets accessed - look at app/ for the challenge
* 