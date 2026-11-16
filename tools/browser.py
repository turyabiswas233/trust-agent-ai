import platform
import webbrowser


def open_url(url: str) -> bool:
    # Check what operating system is running
    system_info = platform.system()
    print(f"System: {system_info}")

    # Try to get and open the default browser
    try:
        browser = webbrowser.get()
        # Use getattr safely in case the browser object lacks a 'name' attribute
        browser_name = getattr(browser, "name", type(browser).__name__)
        print(f"Default browser found: {browser_name}")

        # Open the URL
        result = webbrowser.open(url, new=0)
        return bool(result)

    except Exception as e:
        print(f"Error opening default browser: {e}")

        # Check common browsers to see which ones are registered/available
        common_browsers = [
            "chrome",
            "google-chrome",
            "firefox",
            "edge",
            "msedge",
        ]
        available_browsers = []

        for name in common_browsers:
            try:
                webbrowser.get(name)
                available_browsers.append(name)
            except webbrowser.Error:
                pass

        print(f"Available browsers on this system: {available_browsers}")
        print("You can manually open a specific browser using:")
        print("    import webbrowser")
        print("    webbrowser.get('chrome').open('https://www.google.com')")
        return False


# Example usage:
if __name__ == "__main__":
    open_url("https://www.chatgpt.com")
