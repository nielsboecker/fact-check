import json

from termcolor import colored

from util.strings import truncate


def extract_lines(lines_json_string: str) -> list:
    raw_lines = lines_json_string.split('\n')
    parsed_lines = []

    for raw_line in raw_lines:
        line_parts = raw_line.split('\t')
        line_id = int(line_parts[0])
        line_text = line_parts[1]
        # use set to implicitly discard duplicate anchor tokens
        line_anchors = set(line_parts[2:])
        parsed_lines.append(WikiLine(line_id, line_text, line_anchors))

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
    def __init__(self, id: int, text: str, anchors: list):
        self.id = id
        self.text = text
        self.anchors = anchors
