import os
import sys
import tokenize
import io
import argparse
import shutil

def remove_comments(source_code):
    io_obj = io.BytesIO(source_code.encode("utf-8"))
    try:
        tokens = tokenize.tokenize(io_obj.readline)
        filtered_tokens = [t for t in tokens if t.type != tokenize.COMMENT]
        return tokenize.untokenize(filtered_tokens)
    except tokenize.TokenError:
        return source_code

def main():
    parser = argparse.ArgumentParser(description="Remove comments from Python files")
    parser.add_argument("target_dir", type=str, help="Path to the target directory")
    args = parser.parse_args()

    target_dir = args.target_dir

    if not os.path.exists(target_dir):
        print(f"Error: Directory {target_dir} not found.")
        sys.exit(1)

    print(f"Processing directory: {target_dir}")

    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                
                if os.path.abspath(file_path) == os.path.abspath(__file__):
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source = f.read()

                    cleaned_source = remove_comments(source)

                    if cleaned_source != source:
                        backup_path = file_path + ".bak"
                        shutil.copy2(file_path, backup_path)
                        
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(cleaned_source)
                        
                        print(f"Cleaned: {file_path} (Backup saved as .bak)")
                    else:
                        print(f"Skipped (No changes): {file_path}")

                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    main()

