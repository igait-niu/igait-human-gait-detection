import os
import subprocess

def main():
    print("Running tests!\n")
    
    files = []
    directory_path = os.path.abspath("./data")  

    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        
        if os.path.isfile(full_path):            
            
            command = ["python3", "yolo_slowfast.py", "--input", full_path, "--output", f"data/{full_path}"]

            print(f"\nProcessing File: {entry}")
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                print(result.stdout)
                print(f"Return Code: {result.returncode}")
                print("Success\n")
    
            except subprocess.CalledProcessError as e:
                print(f"Command failed with exit code {e.returncode}")
                print(f"Error output:\n{e.stderr}")
            except FileNotFoundError:
                print(f"Error: Command '{command[0]}' not found.")
            except Exception as e:
                print(f"Unexpected error: {e}")

            files.append(full_path)
    
    return files

if __name__ == "__main__":
    main()