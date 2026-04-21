## 2024-05-24 - Unsafe System Calls Replaced
**Vulnerability:** The use of `os.system()` to execute shell commands like `pip install` and `rm -r` can lead to command injection and unintended side effects, especially if inputs are dynamic.
**Learning:** The codebase previously relied on `os.system()` which evaluates commands in a subshell, making it vulnerable and less portable.
**Prevention:** Always use safe alternatives like `subprocess.check_call()` with list arguments to bypass the shell, and native Python libraries like `shutil.rmtree()` for file system operations.
