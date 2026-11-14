from pathlib import Path
 

def read_file_text(file_path):                                                                                                                                                            
    """Read a file, searching the current directory tree for a bare filename."""
    path = Path(file_path)

    if not path.is_file() and not path.is_absolute() and path.parent == Path("."):
        path = next(
            (
                candidate
                for candidate in Path.cwd().rglob("*")
                if candidate.is_file() and candidate.name == path.name
            ),
            None,
        )

    if path is None or not path.is_file():
        raise FileNotFoundError(f"File '{file_path}' not found")
    print(f"File found in : `{path.absolute()}`")
    return path.read_text(encoding="utf-8")
    
if __name__ == '__main__':
    fp = str(input("Give a file path: "))
    print(read_file_text(fp))
    
 
