import psutil

# Check memory
mem = psutil.virtual_memory()
print(mem.percent)
