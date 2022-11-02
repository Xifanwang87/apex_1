import re
def camelcase_to_snakecase(name):
    name = name.replace(' ', '_')
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    res = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    res = re.sub('[_]+', '_', res)
    return res