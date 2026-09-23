import os, sys, time
OFF = os.environ.get("MODE","offline")=="offline"
if OFF:
    os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
from med_merge.models.factory import create_model, BACKBONE_REGISTRY
from med_merge.config.schema import ModelConfig, DatasetConfig
IDS=["openai/clip-vit-base-patch16","google/vit-base-patch16-224",
     "facebook/dinov3-vits16-pretrain-lvd1689m","microsoft/rad-dino",
     "facebook/dinov2-base","facebook/vit-mae-base",
     "microsoft/beit-base-patch16-224-pt22k-ft22k"]
for bid in IDS:
    h = BACKBONE_REGISTRY[bid][1]
    try:
        t0=time.time()
        create_model(DatasetConfig(name="isic_mel",num_classes=1,task_type="binary"),
                     model_config=ModelConfig(backbone=bid,hidden_size=h,num_layers=12))
        print(f"OK   {bid:<48} {time.time()-t0:5.1f}s")
    except Exception as e:
        print(f"FAIL {bid:<48} {type(e).__name__}")
