import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


app = FastAPI()


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Request model
class CodeRequest(BaseModel):
    code: str


# AI response model
class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# Execute Python code
def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    stdout = StringIO()
    stderr = StringIO()

    sys.stdout = stdout
    sys.stderr = stderr

    try:
        exec(code)

        # Successful execution: return exact stdout
        output = stdout.getvalue()

        return {
            "success": True,
            "output": output
        }

    except Exception:
        # Error: return exact traceback
        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# Analyze error using AI
def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:

    client = OpenAI(
        api_key=os.environ["AIPIPE_TOKEN"],
        base_url="https://aipipe.org/openai/v1"
    )

    prompt = f"""
Analyze the Python code and traceback below.

Identify the exact line number or line numbers in the user's Python
code where the error occurred.

Return ONLY valid JSON in exactly this format:

{{
  "error_lines": [3]
}}

Rules:
- Return only line numbers from the user's CODE.
- Do not include line numbers from the traceback itself unless they
  correspond to the user's code.
- If there is one error, return one line number.
- If there are multiple relevant error lines, return all relevant
  line numbers.
- Do not return explanations.

CODE:
{code}

TRACEBACK:
{traceback_text}
"""

    response = client.chat.completions.create(
        model="openai/gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_object"
        }
    )

    result = ErrorAnalysis.model_validate_json(
        response.choices[0].message.content
    )

    return result.error_lines


# Main endpoint
@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):

    execution = execute_python_code(request.code)

    # Successful code -> do not call AI
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # Error -> call AI
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    return {
        "error": error_lines,
        "result": execution["output"]
    }