import json

from termcolor import colored

from util.strings import truncate


def extract_lines(lines_json_string: str):
    raw_lines = lines_json_string.split('\n')
    parsed_lines = []

    for raw_line in raw_lines:
        line_parts = raw_line.split('\t')
        line_index = int(line_parts[0])
        line_text = line_parts[1]
        # Use set to implicitly discard duplicate anchor tokens
        line_anchors = set(line_parts[2:])
        parsed_lines.append(WikiLine(line_index, line_text, line_anchors))

    return parsed_lines


class WikiDocument:
    def __init__(self, doc_json_string: str):
        data = json.loads(doc_json_string)
        self.id = data['id']
        self.text = data['text']
        self.lines = extract_lines(data['lines'])

    def __str__(self):
        return 'ID:\t{}\nTEXT:\t{}\n'.format(colored(self.id, attrs=['bold']),
                                             colored(truncate(self.text), attrs=['underline']))


class WikiLine:
    def __init__(self, index, text, anchors):
        self.index = index
        self.text = text
        self.anchors = anchors
