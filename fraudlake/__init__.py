"""FraudLake — code partagé entre les notebooks, le scoring batch et l'API de scoring.

Un seul endroit définit les features, les règles d'explication et les seuils de surveillance :
c'est la garantie qu'il n'y a pas d'écart entre l'entraînement et la production (*training/serving skew*).
"""

__version__ = "1.0.0"
