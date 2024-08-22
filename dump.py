from transformers import AutoTokenizer
import os
import concurrent.futures
from threading import Lock
import sys
import ast

# Define the list of file extensions to include
INCLUDE_EXTENSIONS = [
    ".py",
    ".ipynb",
    ".dockerfile",
    "Dockerfile",
    ".txt",
    ".cfg",
    ".yml",
    ".yaml",
    ".json",
    ".ini",
    ".protobuf",
    ".capnp",
    ".conf",
]

# Define the list of paths to ignore
IGNORE_DIRS = [".", "__pycache__"]
IGNORE_PATHS = [
    "/packages/syft/tests/mongomock",
    "/packages/syft/tests/utils/",
    "/packages/syft/tests/",
]


def remove_docstrings_and_comments(source):
    """
    Remove docstrings from a given Python source code string.
    """

    def is_docstring(node):
        return isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ) and ast.get_docstring(node)

    def remove_docstrings_recursively(node):
        for child in ast.iter_child_nodes(node):
            if is_docstring(child):
                child.body = [
                    n
                    for n in child.body
                    if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Str)
                ]
            remove_docstrings_recursively(child)

    tree = ast.parse(source)
    remove_docstrings_recursively(tree)
    return ast.unparse(tree)


def process_file(
    file_path, lock, output_file, remove_comments_and_imports, remove_docstrings
):
    with open(file_path, "r") as infile:
        content = infile.read()

    if remove_comments_and_imports:
        content_lines = content.splitlines()
        content_lines = [
            line
            for line in content_lines
            if not line.strip().startswith("#")
            and not line.strip().startswith("import")
        ]
        content_lines = [
            line
            for line in content_lines
            if not (line.strip().startswith("from") and "import" in line)
        ]
        content = "\n".join(content_lines)

    if remove_docstrings:
        content = remove_docstrings_and_comments(content)

    with lock:
        with open(output_file, "a") as outfile:
            outfile.write(f"# File: {file_path}\n\n")
            outfile.write(content)
            outfile.write("\n\n")  # Add spacing between files


def scan_and_concatenate(
    source_dir, output_file, script_name, remove_comments_and_imports, remove_docstrings
):
    # Collect all files with the specified extensions
    matching_files = []
    for root, dirs, files in os.walk(source_dir):
        # Ignore hidden folders and __pycache__ folders
        dirs[:] = [d for d in dirs if not any(ignored in d for ignored in IGNORE_DIRS)]
        dirs[:] = [
            d
            for d in dirs
            if not any(
                ignored in os.path.abspath(root + "/" + d) for ignored in IGNORE_PATHS
            )
        ]

        for file in files:
            if (
                any(file.endswith(ext) for ext in INCLUDE_EXTENSIONS)
                and file != os.path.basename(output_file)
                and file != script_name
            ):
                matching_files.append(os.path.join(root, file))

    # Sort the files to make the order deterministic
    matching_files.sort()

    # Dump the sorted list of files to 'indexed_files'
    with open("indexed_files", "w") as index_file:
        for file_path in matching_files:
            index_file.write(f"{file_path}\n")

    # Create a lock for thread-safe writing
    lock = Lock()

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_file,
                file,
                lock,
                output_file,
                remove_comments_and_imports,
                remove_docstrings,
            )
            for file in matching_files
        ]
        concurrent.futures.wait(futures)

    # Print the number of files processed
    print(f"Number of files processed: {len(matching_files)}")


def count_tokens_in_file(file_path, model_name="gpt2"):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Read the contents of the file
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Return the number of tokens
    return len(tokens)


if __name__ == "__main__":
    # Specify the directory of your git repository and this script's name
    git_folder = "./packages/syft"
    script_name = os.path.basename(__file__)

    # Optional: Take output file path/name from command line arguments
    output_python_file = sys.argv[1] if len(sys.argv) > 1 else "merged_output.py"

    # Optional: Flags to remove comments, imports, and docstrings
    remove_comments_and_imports = True
    remove_docstrings = True

    # Clear the output file before appending
    open(output_python_file, "w").close()

    scan_and_concatenate(
        git_folder,
        output_python_file,
        script_name,
        remove_comments_and_imports,
        remove_docstrings,
    )
    print(f"All specified files have been concatenated into {output_python_file}")
    print(f"File list has been saved to 'indexed_files'")

    # Count the number of tokens
    token_count = count_tokens_in_file(output_python_file)

    print(f"Number of tokens in the file: {token_count}")
