from .base import get_model as amna__base_model
from .subport_one import get_model as amna__subportfolio_one
from .subport_two import get_model as amna__subportfolio_two

def get_models():
    models = {
        'base': amna__base_model,
        'subportfolio_one': amna__subportfolio_one,
        'subportfolio_two': amna__subportfolio_two,
    }
    return models
