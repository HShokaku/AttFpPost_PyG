from rdkit import RDLogger


def disable_rdkit_logging():
    RDLogger.DisableLog("rdApp.*")
