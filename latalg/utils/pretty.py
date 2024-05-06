from __future__ import annotations

import colorama

def init():
    colorama.init()

def section_print(message):
    print(colorama.Fore.GREEN + '==> ' + message + colorama.Style.RESET_ALL)

def subsection_print(message):
    print(colorama.Fore.BLUE + '====> ' + message + colorama.Style.RESET_ALL)

def warning_print(message):
    print(colorama.Fore.YELLOW + message + colorama.Style.RESET_ALL)
