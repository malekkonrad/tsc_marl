"""Uruchamianie — wejścia procesu i orkiestracja biegów.

Warstwa najwyższa: wolno jej importować wszystko poniżej, nikt nie importuje jej.
`cli.py` jest cienki z założenia (rozdziela ruch), a logika siedzi w `train.py`,
`train_core.py`, `evaluate.py` i `suite.py` — dzięki temu testy i podprocesy
`suite` wołają te same funkcje, a nie CLI.
"""
