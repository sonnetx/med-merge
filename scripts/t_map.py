from med_merge.data.ham10000 import _dx_to_index, CLASS_NAMES, DX_ALIASES
seen = {"melanocytic_Nevi":6405,"melanoma":1076,"benign_keratosis-like_lesions":1048,
        "basal_cell_carcinoma":487,"actinic_keratoses":315,"vascular_lesions":136,
        "dermatofibroma":110}
idx = {dx:_dx_to_index(dx) for dx in seen}
print("mapping:", {k:(v,CLASS_NAMES[v]) for k,v in idx.items()})
print("distinct indices:", sorted(set(idx.values())), "expected 0..6")
assert sorted(set(idx.values()))==list(range(7)), "not a bijection onto 7 classes"
assert _dx_to_index("mel")==CLASS_NAMES.index("mel"), "abbreviation form broke"
try:
    _dx_to_index("not_a_diagnosis"); print("FAIL: unknown label did not raise")
except ValueError: print("unknown label raises as intended")
print("melanoma ->", _dx_to_index("melanoma"), "which is", CLASS_NAMES[_dx_to_index("melanoma")])
