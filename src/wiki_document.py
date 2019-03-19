import json

from termcolor import colored

DEBUG = False


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
    def __init__(self, doc_json_string):
        # Use raw string to account for \t and \n characters
        raw_doc_json_string = r"{}".format(doc_json_string)
        data = json.loads(raw_doc_json_string, encoding='asciixx')
        # id, foo = extract_data(doc_json_string)
        self.id = data['id']
        self.text = data['text']
        self.lines = extract_lines(data['lines'])

    def __str__(self):
        return 'ID:\t\t{}\nTEXT:\t{}\n'.format(colored(self.id, attrs=['bold']),
                                               colored(self.text, attrs=['underline']))


class WikiLine:
    def __init__(self, index, text, anchors):
        self.index = index
        self.text = text
        self.anchors = anchors


if __name__ == '__main__':
    if (DEBUG):
        test_doc = WikiDocument(
            r'{"id": "Hello", "text": "Hello is a salutation or greeting in the English language . It is first attested in writing from 1826 . ", "lines": "0\tHello is a salutation or greeting in the English language .\tsalutation\tsalutation\tgreeting\tgreeting habits\tEnglish language\tEnglish language\n1\tIt is first attested in writing from 1826 .\n2\t"}')
        print(test_doc)
