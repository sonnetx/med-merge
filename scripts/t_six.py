from med_merge.data.registry import build_dataset
CX="/path/to/data/processed_chexpert"
SPEC=[("isic_mel","/path/to/data/isic/challenges/2017/merged_isic_2017_data/images",{}),
      ("ham10000_mel","/path/to/data/huggingface",{}),
      ("chexpert_pe",f"{CX}/combined_train_valid_chexpert_v1.0",{"csv_path":f"{CX}/explore_chexpert/train_valid_combined.csv"}),
      ("chexpert_cm",f"{CX}/combined_train_valid_chexpert_v1.0",{"csv_path":f"{CX}/explore_chexpert/train_valid_combined.csv"}),
      ("patchcamelyon","/path/to/data/huggingface",{}),
      ("pathmnist_bin","/path/to/data/medmnist",{})]
for name,dd,kw in SPEC:
    try:
        d=build_dataset(name,dd,split="validation",**kw)
        n=len(d); pos=sum(int(d[i][1].item()) for i in range(0,n,max(1,n//150)))
        tot=len(range(0,n,max(1,n//150)))
        print(f"OK   {name:<15} n={n:<7} sampled pos {pos}/{tot}")
    except Exception as e:
        print(f"FAIL {name:<15} {type(e).__name__}: {str(e)[:80]}")
