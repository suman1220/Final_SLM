# main.py

from main_only_model import run_initialmodel
from main_for_guardrail import run_guardrails
from ragModel import run_rag

def go_to_initialmodel():
    print("Redirecting to Inital Model...")
    run_initialmodel()

def go_to_guardrails():
    print("Redirecting to Guardrails...")
    run_guardrails()

def go_to_rag():
    print("Redirecting to Rag Model...")
    run_rag()
