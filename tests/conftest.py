"""Configura o path do pacote para os testes.

Note:
    Esta inserção mantém a suíte independente de instalação do pacote.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
