# Assistant

This project is to tie together various LLM-related pieces, to try and build an AI assistant that can research and provide feedback on things I'm working on.

## How to Use

Checkout the source code, cd to the "agent" directory, run `python -m pip install -U --upgrade-strategy eager -r requirements.txt` to install dependencies.
then run `python ./client_gradio.py` to start the agent, and look for the locahost URI printed in the output - you should
be able to web browse to that.

If you have API keys for various things, you can look at the .env-template files - rename them to .env and add
keys or configurations as needed. You can find them in each of the sub-directories from the main dir. The critical one
is the one in the root of the project. Rerun the agent if you change any of these.

If you want to do certain things, you'll need the containers running - we're down to a text-to-speech and a chromadb for the google docs
plugin. Simply run `docker-compose up -d` and it should run the containers for you - or, if you want to point at a SaaS solution,
check the .env file.

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

## Process

Most of the code you'll find in here is either from other projects, or from ChatGPT or CoPilot.
Other projects that have been an inspiration include:

  * AutoGPT
  * Dalai
  * Milvus
  * https://github.com/Venthe/chatgpt4all-webui
  * https://github.com/randaller/llama-chat
  * https://open-assistant.io/

** Prompting for a task I want the agent to be able to do:

Given a document to analyse, you should do the following steps:

* Load the document into your document store
* Search the document for primary subject matter
* Search your memory for the same subject matter
* Search the web for websites pertaining to this subject matter
* Save websites that talk about this subject matter
* Based on what you've learnt from the websites and your memory, provide an analysis of the initial document.

You have the following tools to help you do this:

* DocStore to load the document, or the Google doc loader for google docs
* web scraper for websites
* memory to store scraped websites and the document itself