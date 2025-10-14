from collections import defaultdict
import re


def parse_simple_formula(formula):
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    counts = defaultdict(float)
    for el, num in re.findall(pattern, formula):
        counts[el] += float(num) if num else 1.0
    return dict(counts)


def get_el_amt_dict(formula):
    separators = r'[@/–]'
    parts = re.split(separators, formula)

    total_counts = defaultdict(float)

    for part in parts:
        part = part.strip()
        m = re.fullmatch(r'\(?([A-Za-z0-9]+)\)?([\d\.]*)', part)
        if m:
            f, factor = m.groups()
            factor = float(factor) if factor else 1.0
            counts = parse_simple_formula(f)
            for el, n in counts.items():
                total_counts[el] += n * factor
        else:
            counts = parse_simple_formula(part)
            for el, n in counts.items():
                total_counts[el] += n

    return dict(total_counts)
