def get_part_strings(part, total):

    return (part, part / total * 100, total)

def merge_sentences(sentences):

    return " ".join(sentences)

def word_count(sentences):

    if sentences is None or len(sentences) == 0:
        return 0

    return len(merge_sentences(sentences).split())