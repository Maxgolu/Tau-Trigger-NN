# Figure scripts

For single-object results figures and tables, use
[the packaged evidence and reproduction commands](../../../results/single_object/README.md).
The `results/` scripts write to `results/generated/single_object/paper/` by
default, leaving the approved submission figures and PDF unchanged.

The data figures were made from the real Signal/Background files (training split, seed 42).

    python prep.py "path/to/Tau Trigger Project"   # writes obj.pkl, T.npy, E.npy here
    python fig_tensor.py                            # fig_calo_input.pdf
    python fig_feat.py                              # fig_features.pdf
    python fig_app.py                               # fig_frac_conditioned.pdf

Requires numpy, pandas, matplotlib. The scripts use the TeX Gyre Termes font if it is installed
(edit the font lines if not). Copy the resulting PDFs into ../ (the figures folder).
