from pathlib import Path
from time import time_ns

def read_file_text(file_path):                                                                                                                                                            
    """Reads text from a file and returns the content as string"""                                                                                                                        
    try:                                                                                                                                                                                  
        with open(file_path, 'r', encoding='utf-8') as file:                                                                                                                              
            text = file.read()                                                                                                                                                            
        return text                                                                                                                                                                       
    except FileNotFoundError:                                                                                                                                                             
        print(f"Error: File '{file_path}' not found!")                                                                                                                                    
        return None                                                                                                                                                                       
    except Exception as e:                                                                                                                                                                
        print(f"Error reading file: {e}")                                                                                                                                                 
        return None
    
if __name__ == '__main__':
    fp = str(input("Give a file path: "))
    print(read_file_text(fp))
    
def save_to_file(context='', file_path=None):
    if file_path is None:
        file_path = f'./chat_{time_ns()}.txt'

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        file.write(context)
        
    return str(path)
