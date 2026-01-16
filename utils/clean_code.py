import os
import sys
import tokenize
from io import BytesIO

def remove_comments_and_docstrings(source):
    io_obj = BytesIO(source.encode('utf-8'))
    out = ""
    removed_content = []
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    
    try:
        tokgen = tokenize.tokenize(io_obj.readline)
        for tok in tokgen:
            token_type = tok.type
            token_string = tok.string
            start_line, start_col = tok.start
            end_line, end_col = tok.end
            
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
                
            if token_type == tokenize.COMMENT:
                removed_content.append(f"[Comment] Line {start_line}: {token_string.strip()}")
                pass
            
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
                    else:
                        removed_content.append(f"[Docstring] Line {start_line}: {token_string.strip()[:50]}...")
                        pass
                else:
                    removed_content.append(f"[Docstring] Line {start_line}: {token_string.strip()[:50]}...")
                    pass
            else:
                out += token_string
                
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
            
    except tokenize.TokenError:
        return source, ["Error parsing tokens"]
        
    return out, removed_content

def process_file(file_path, target_root_dir):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned_content, removed_items = remove_comments_and_docstrings(content)
        
        lines = [line for line in cleaned_content.splitlines() if line.strip()]
        final_content = '\n'.join(lines) + '\n'

        rel_path = os.path.relpath(file_path, start=target_root_dir)
        output_path = os.path.join(target_root_dir, "bak", rel_path)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
            
        print(f"\nProcessed: {file_path} -> {output_path}")
        if removed_items:
            print("  --- Removed Content ---")
            for item in removed_items:
                print(f"  {item}")
        else:
            print("  (No comments or docstrings found)")
            
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_code.py <directory_path>")
        sys.exit(1)
        
    target_dir = os.path.abspath(sys.argv[1])
    print(f"Processing directory: {target_dir}")
    print(f"Backup location: {os.path.join(target_dir, 'bak')}")
    
    for root, dirs, files in os.walk(target_dir):
        if 'bak' in root.split(os.sep):
            continue

        for file in files:
            if file.endswith(".py") and file != os.path.basename(__file__):
                file_path = os.path.join(root, file)
                process_file(file_path, target_dir)

if __name__ == "__main__":
    main()

