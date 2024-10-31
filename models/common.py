import os


class colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDCOLOR = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def wrap_text(text: str, color: str):
        return color + text + colors.ENDCOLOR


def get_root_directory():
    current_directory = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_directory, "../"))
