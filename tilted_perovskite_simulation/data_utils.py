import numpy as np
import os
import glob
from pathlib import Path
from diffpy.Structure import loadStructure
from diffpy.srreal.pdfcalculator import PDFCalculator
from pathlib import Path
from sympy import *
from typing import *
from scipy.interpolate import interp1d


def write_single_pdf(args):
    cif_file_dir, pdf_file_dir, pc_cfg = args
    pc = PDFCalculator(**pc_cfg)
    s = loadStructure(cif_file_dir)
    try:
        r, g = pc(s)
        file = open(pdf_file_dir, "a")
        rg = np.array([r, g])
        rg = rg.T
        np.savetxt(file, rg, fmt=["%f", "%f"])
        file.close()
    except Exception as ex:
        print(str(ex))
        print("Exception: PDFCalculator not applied on {}.".format(cif_file_dir))
    base_name = os.path.basename(pdf_file_dir)
    file_name_without_suffix = os.path.splitext(base_name)[0]
    return [file_name_without_suffix] + list(pc_cfg.values())


def write_multiple_pdf(cif_folder_dir: str, pdf_folder_dir: str, pc_cfg: dict):
    if not os.path.exists(pdf_folder_dir):
        os.makedirs(pdf_folder_dir)

    for cif_file_dir in glob.glob("{}/*.cif".format(cif_folder_dir)):
        cif_file_name = os.path.basename(cif_file_dir)
        pdf_file_dir = Path(pdf_folder_dir, cif_file_name).with_suffix(".gr")
        args = (cif_file_dir, pdf_file_dir, pc_cfg)
        write_single_pdf(args)
