import time

def print_typewriter(text, delay=0.05, request_new_line=False):
  # Looping directly over a string iterates through one character at a time
  for char in text:
    print(char, end="", flush=True)
    time.sleep(delay)

  if request_new_line:
    print()

