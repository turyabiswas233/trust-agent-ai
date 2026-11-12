import datetime

def get_current_time() -> str:
    # Returns the current date and time.
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
