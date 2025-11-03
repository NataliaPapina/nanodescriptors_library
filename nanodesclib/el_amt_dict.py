import re
from collections import defaultdict

def get_el_amt_dict(formula: str) -> dict:
    """
    Возвращает словарь {element: количество атомов} для формулы.
    Поддерживает:
      - дробные индексы (O2.92)
      - группы с коэффициентами ((Fe2O3)0.7)
      - сложные формулы с разделителями – / @
    """
    def multiply_dict(d, factor):
        return {k: v * factor for k, v in d.items()}

    def parse_simple(f):
        """Разбор простой формулы без групп"""
        pattern = r'([A-Z][a-z]?)(\d*\.?\,?\d*)'
        counts = defaultdict(float)
        for el, num in re.findall(pattern, f):
            counts[el] += float(num) if num else 1.0
        return dict(counts)

    def parse_group(f):
        """Разбор формулы с возможными скобками и коэффициентами"""
        counts = defaultdict(float)
        # (X2Y3)0.7
        pattern = r'\(([^()]+)\)(\d*\.?\d*)'
        while True:
            m = re.search(pattern, f)
            if not m:
                break
            subformula, factor = m.groups()
            factor = float(factor) if factor else 1.0
            sub_counts = multiply_dict(parse_group(subformula), factor)
            for el, n in sub_counts.items():
                counts[el] += n
            f = f[:m.start()] + f[m.end():]
        simple_counts = parse_simple(f)
        for el, n in simple_counts.items():
            counts[el] += n
        return dict(counts)

    separators = r'[-/@]'
    parts = re.split(separators, formula)
    total_counts = defaultdict(float)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        sub_counts = parse_group(part)
        for el, n in sub_counts.items():
            total_counts[el] += n
    return dict(total_counts)
