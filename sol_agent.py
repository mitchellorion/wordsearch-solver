import os
import sys
import json
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

tools = [
    {
        "type": "function",
        "name": "read_file",
        "description": "Read the contents of a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "The path to the file"}
            },
            "required": ["filepath"]
        }
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write or overwrite a file with new content.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "The path to the file"},
                "content": {"type": "string", "description": "The entire content to write"}
            },
            "required": ["filepath", "content"]
        }
    },
    {
        "type": "function",
        "name": "run_command",
        "description": "Run a terminal command (e.g. tests or python scripts) and get output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"}
            },
            "required": ["command"]
        }
    },
    {
        "type": "function",
        "name": "list_directory",
        "description": "List files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path to the directory"}
            },
            "required": ["path"]
        }
    }
]

def execute_tool(name: str, args: dict) -> str:
    try:
        if name == "read_file":
            with open(args["filepath"], "r", encoding="utf-8") as f:
                return f.read()
        elif name == "write_file":
            with open(args["filepath"], "w", encoding="utf-8") as f:
                f.write(args["content"])
            return f"Successfully wrote to {args['filepath']}"
        elif name == "run_command":
            result = subprocess.run(args["command"], shell=True, capture_output=True, text=True)
            output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nExit Code: {result.returncode}"
            return output
        elif name == "list_directory":
            files = os.listdir(args["path"])
            return "\n".join(files)
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error executing {name}: {str(e)}"

def chat_loop():
    print("=========================================")
    print("   Sol Agent - Autonomous Coding Assistant")
    print("   Powered by GPT-5.6-Sol (Responses API)")
    print("=========================================")
    print("Type 'exit' or 'quit' to exit.\n")
    
    instructions = "You are a local autonomous coding assistant. You can read files, modify files, and run commands. The user is a developer."
    previous_response_id = None
    
    # Bypass local SSL verification issues
    import httpx
    global client
    client = OpenAI(http_client=httpx.Client(verify=False))
    
    while True:
        try:
            user_input = input("\033[92mYou: \033[0m")
            if user_input.strip().lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue
                
            current_input = [{
                "role": "user",
                "content": user_input
            }]
            
            while True:
                kwargs = {
                    "model": "gpt-5.6-sol",
                    "instructions": instructions,
                    "input": current_input,
                    "tools": tools,
                }
                if previous_response_id:
                    kwargs["previous_response_id"] = previous_response_id
                    
                response = client.responses.create(**kwargs)
                previous_response_id = response.id
                
                current_input = []
                has_tool_calls = False
                
                # Process the output stream
                for item in response.output:
                    item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
                    
                    if item_type == "message":
                        content_list = getattr(item, "content", []) or (item.get("content", []) if isinstance(item, dict) else [])
                        for c in content_list:
                            c_type = getattr(c, "type", None) or (c.get("type") if isinstance(c, dict) else None)
                            if c_type == "output_text":
                                text = getattr(c, "text", "") or (c.get("text", "") if isinstance(c, dict) else "")
                                print(f"\033[96mSol: \033[0m{text}")
                                
                    elif item_type == "function_call":
                        has_tool_calls = True
                        name = getattr(item, "name", "") or (item.get("name", "") if isinstance(item, dict) else "")
                        args_str = getattr(item, "arguments", "{}") or (item.get("arguments", "{}") if isinstance(item, dict) else "{}")
                        call_id = getattr(item, "call_id", "") or (item.get("call_id", "") if isinstance(item, dict) else "")
                        
                        print(f"\033[93m[Sol is using tool: {name}]\033[0m")
                        try:
                            args = json.loads(args_str)
                        except json.JSONDecodeError:
                            args = {}
                        result = execute_tool(name, args)
                        
                        current_input.append({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": str(result)
                        })
                
                if not has_tool_calls:
                    break
                    
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")
            break

if __name__ == "__main__":
    chat_loop()
