import mteb
# set GPU 1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

benchmark = mteb.get_benchmark("ViDoRe(v3)")
model = mteb.get_model("vidore/colqwen2.5-v0.2")

results = mteb.evaluate(model=model, tasks=benchmark)
