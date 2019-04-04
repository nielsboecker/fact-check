import random

from dataaccess.access_claims import get_claim_row
from dataaccess.access_wiki_page import get_random_wiki_line
from model.wiki_document import WikiDocument, WikiLine


def get_irrelevant_line(wiki_page: WikiDocument, relevant_line_ids: list) -> WikiLine:
    candidate_ids = list(range(len(wiki_page.lines)))
    candidate_ids = [line_id for line_id in candidate_ids if line_id not in relevant_line_ids]

    if not candidate_ids:
        # if all sentences in this wiki page are relevant, return random line from other page
        return get_random_wiki_line()

    line = wiki_page.lines[random.choice(candidate_ids)]
    if not line.text:
        # some empty lines in dataset
        return get_random_wiki_line()

    return line


def is_relevant(doc_id: str, line_id: int, evidence_map: dict) -> bool:
    if doc_id in evidence_map:
        return line_id in evidence_map[doc_id]
    else:
        return False


def get_evidence_page_line_map(claim_id: int, dataset: str) -> dict:
    mapping = {}
    evidence = get_claim_row(claim_id, dataset)['evidence'][0]
    for _, _, page_id, line_id in evidence:
        mapping.setdefault(page_id, []).append(line_id)
    return mapping
