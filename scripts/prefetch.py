"""Pre-cache backbones on the login node so compute nodes can run fully offline."""
import time
from transformers import AutoModel, AutoConfig, AutoImageProcessor
IDS = ["google/vit-base-patch16-224",
       "facebook/dinov3-vits16-pretrain-lvd1689m",
       "microsoft/rad-dino"]
for bid in IDS:
    t0 = time.time()
    try:
        AutoConfig.from_pretrained(bid)
        AutoModel.from_pretrained(bid)
        try: AutoImageProcessor.from_pretrained(bid)
        except Exception: pass
        print(f"CACHED {bid}  {time.time()-t0:.0f}s")
    except Exception as e:
        print(f"FAILED {bid}  {type(e).__name__}: {str(e)[:110]}")
