PROMPT_FILE_PATH = 'custom_prompts.txt'
def readPrompt():
    file_path = PROMPT_FILE_PATH
    try:
        with open(file_path, 'r') as file:
            lines_array = file.readlines()
        
        print(lines_array)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")

readPrompt()