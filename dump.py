from transformers import AutoTokenizer
import os
import concurrent.futures
from threading import Lock
import sys
import ast
import csv

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
    file_path, lock, content_list, remove_comments_and_imports, remove_docstrings
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

    char_count = len(content)

    if char_count > 0:  # Only process files with more than 0 characters
        with lock:
            content_list.append((file_path, content, char_count))


def scan_and_concatenate(
    source_dir,
    output_prefix,
    script_name,
    remove_comments_and_imports,
    remove_docstrings,
    n_chunks=2,
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
                and file != os.path.basename(output_prefix)
                and file != script_name
            ):
                matching_files.append(os.path.join(root, file))

    # Sort the files to make the order deterministic
    matching_files.sort()

    # Create a lock for thread-safe writing
    lock = Lock()

    # Store contents and char counts in a list
    content_list = []

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_file,
                file,
                lock,
                content_list,
                remove_comments_and_imports,
                remove_docstrings,
            )
            for file in matching_files
        ]
        concurrent.futures.wait(futures)

    # Calculate total characters and determine chunk sizes
    total_chars = sum(char_count for _, _, char_count in content_list)
    chunk_size = total_chars // n_chunks
    current_chunk_chars = 0
    current_chunk_index = 1
    current_chunk_file = f"{output_prefix}_chunk_{current_chunk_index}.txt"

    # Clear and create the initial output file
    open(current_chunk_file, "w").close()

    # Open the CSV file and write headers
    with open("indexed_files.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["File", "Char Count"])

        # Write content to chunks
        for file_path, content, char_count in content_list:
            if (
                current_chunk_chars + char_count > chunk_size
                and current_chunk_index < n_chunks
            ):
                current_chunk_index += 1
                current_chunk_file = f"{output_prefix}_chunk_{current_chunk_index}.txt"
                open(current_chunk_file, "w").close()
                current_chunk_chars = 0

            with open(current_chunk_file, "a") as outfile:
                outfile.write(f"# File: {file_path}\n\n")
                outfile.write(content)
                outfile.write("\n\n")  # Add spacing between files

            current_chunk_chars += char_count
            csv_writer.writerow([file_path, char_count])

    # Print the number of files processed
    print(f"Number of files processed: {len(matching_files)}")
    print(f"Output written to {n_chunks} chunks.")


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
    git_folder = "./packages/syft/src/syft/service"
    script_name = os.path.basename(__file__)

    # Optional: Take output file path/name from command line arguments
    output_prefix = sys.argv[1] if len(sys.argv) > 1 else "merged_output"

    # Optional: Flags to remove comments, imports, and docstrings
    remove_comments_and_imports = True
    remove_docstrings = True

    # Optional: Number of chunks
    n_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    scan_and_concatenate(
        git_folder,
        output_prefix,
        script_name,
        remove_comments_and_imports,
        remove_docstrings,
        n_chunks=n_chunks,
    )
    print(f"All specified files have been concatenated into {n_chunks} chunks.")
    print(f"File list and char counts have been saved to 'indexed_files.csv'")

    # Count the number of tokens
    token_count = sum(
        count_tokens_in_file(f"{output_prefix}_chunk_{i+1}.txt")
        for i in range(n_chunks)
    )

    print(f"Number of tokens in the files: {token_count}")
