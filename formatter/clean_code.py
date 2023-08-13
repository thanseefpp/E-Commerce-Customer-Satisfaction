import os
import subprocess

import autopep8
import isort
import pylint.lint


def find_python_files(location):
    python_files = []
    for root, _, files in os.walk(location):
        python_files.extend(
            os.path.join(root, file) for file in files if file.endswith('.py')
        )
    return python_files


def clean_indentation(file_path):
    with open(file_path, 'r') as file:
        original_code = file.read()
    formatted_code = autopep8.fix_code(original_code)
    with open(file_path, 'w') as file:
        file.write(formatted_code)


def remove_unused_variables(file_path):
    # Run pylint with the 'unused-variable' message control to detect unused variables
    pylint.lint.Run([file_path, "--disable=all",
                    "--enable=unused-variable"], do_exit=False)


def remove_unused_imports(file_path):
    isort.file(file_path)


def remove_whitespace(file_path):
    subprocess.run(
        ['flake8', '--ignore=E501', file_path])


def code_cleaner(file_path):
    try:
        clean_indentation(file_path)
        remove_unused_variables(file_path)
        remove_unused_imports(file_path)
        remove_whitespace(file_path)
        print("Code cleaning is successful!")
    except Exception as e:
        print(f"Error during code cleaning: {e}")


if __name__ == "__main__":
    current_working_directory = os.getcwd()
    if python_files := find_python_files(current_working_directory):
        print("Python files found in subfolders:")
        for file in python_files:
            code_cleaner(file)
    else:
        print("No Python files with .py extension found in subfolders.")
