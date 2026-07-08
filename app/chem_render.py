"""RDKit helpers: turn SMILES into PNG images for molecules and reactions.

Deterministic — the picture is the *correct* structure for the given SMILES,
never a hallucinated drawing. Any failure returns None so the caller can fall
back to plain text. Used by `figures.py`.

Structures are drawn monochrome (black atoms, bonds and labels) via a
black-and-white atom palette, rather than RDKit's default CPK colours (blue N,
red O, green Cl …), to match the rest of the document.
"""
from __future__ import annotations

from io import BytesIO

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    RDLogger.DisableLog("rdApp.*")  # silence parse-error spam; we handle failures
    _RDKIT_OK = True
except Exception:  # noqa: BLE001 - rdkit missing / failed to import
    _RDKIT_OK = False


def available() -> bool:
    return _RDKIT_OK


def _bw(drawer) -> None:
    """Force an all-black palette so no atom is coloured."""
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()


def _mol_png_bw(mol, size: tuple[int, int]) -> bytes | None:
    d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    _bw(d)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return d.GetDrawingText()


def mol_png(smiles: str) -> bytes | None:
    if not _RDKIT_OK or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return _mol_png_bw(mol, (340, 250))
    except Exception:  # noqa: BLE001 - Cairo backend unavailable: colour beats nothing
        img = Draw.MolToImage(mol, size=(340, 250))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def rxn_png(smiles: str) -> bytes | None:
    if not _RDKIT_OK or not smiles:
        return None
    try:
        rxn = AllChem.ReactionFromSmarts(smiles, useSmiles=True)
    except Exception:  # noqa: BLE001
        return None
    if rxn is None or rxn.GetNumReactantTemplates() == 0:
        return None
    n = (rxn.GetNumReactantTemplates() + rxn.GetNumProductTemplates()
         + rxn.GetNumAgentTemplates())
    width = min(1600, max(500, 250 * n))
    try:
        d = rdMolDraw2D.MolDraw2DCairo(width, 300)
        _bw(d)
        d.DrawReaction(rxn)
        d.FinishDrawing()
        return d.GetDrawingText()
    except Exception:  # noqa: BLE001 - fall back to the default (coloured) renderer
        img = Draw.ReactionToImage(rxn)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
