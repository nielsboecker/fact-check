# https://stackoverflow.com/questions/15429689/python-format-slice-long-string-and-add-dots
def truncate(string, width=150):
    if len(string) > width:
        string = string[:width-3] + '...'
    return string
