## 2025-02-27 - [Command Injection via `os.system`]
**Vulnerability:** Found insecure usage of `os.system` in `data/pretraining-partial-table-maker.py` for both package installation (`os.system("pip install dustmaps")`) and directory deletion (`os.system("rm -r ...")`). This approach is vulnerable to command injection and relies on the shell environment.
**Learning:** Shell commands should be avoided when native Python equivalents are available. `os.system` runs a subshell and can execute unintended commands if its arguments are not fully sanitized.
**Prevention:** Use `subprocess.check_call` with a list of arguments for executing external programs, ensuring arguments aren't executed by a shell. Use `shutil.rmtree` for directory removal, avoiding external OS commands altogether.
