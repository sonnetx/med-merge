import os, time
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from med_merge.models.factory import create_model
from med_merge.config.schema import ModelConfig, DatasetConfig
for bid in ["openai/clip-vit-base-patch16",
            "microsoft/beit-base-patch16-224-pt22k-ft22k"]:
    t0 = time.time()
    try:
        mc = ModelConfig(backbone=bid, hidden_size=768, num_layers=12)
        dc = DatasetConfig(name="isic_mel", num_classes=1, task_type="binary")
        create_model(dc, model_config=mc)
        print(f"OK   {bid}  loaded offline in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"FAIL {bid}  {type(e).__name__}: {str(e)[:130]}")
