import os
from dotenv import load_dotenv
from langsmith import Client

def test_connection():
    load_dotenv()
    
    api_key = os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT")
    endpoint = os.getenv("LANGCHAIN_ENDPOINT")
    
    print(f"Testing connection to: {endpoint}")
    print(f"Project: {project}")
    print(f"API Key (start): {api_key[:10]}...")
    
    client = Client(api_key=api_key, api_url=endpoint)
    
    try:
        # Try to list projects to verify API key
        projects = list(client.list_projects(name=project))
        if projects:
            print(f"Successfully found project: {projects[0].name}")
        else:
            print(f"Project '{project}' not found, but API key is valid.")
        
        # Try to create a dummy run
        run = client.create_run(
            name="Connection Test",
            run_type="chain",
            inputs={"test": "input"},
            project_name=project
        )
        print(f"Successfully created test run: {run.id}")
        
    except Exception as e:
        print(f"\nCONNECTION FAILED!")
        print(f"Error: {e}")
        if "403" in str(e):
            print("\nAdvice: The 403 error means your API key is invalid or lacks permissions.")
            print("1. Check if the key is copied correctly (no extra spaces).")
            print("2. Check if the key has been revoked in LangSmith settings.")
            print("3. Try creating a brand NEW API Key in LangSmith.")

if __name__ == "__main__":
    test_connection()
