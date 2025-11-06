# %%
from datasets import load_dataset, Features, Value
from datasets.features import LargeList
# set the second GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
dataset_name = "vidore/vidore_v3_industrial"

queries_features = Features({
    "query_id": Value("int64"),
    "query": Value("string"),
    "language": Value("string"),
    "query_types": LargeList(Value("string")),
    "query_format": Value("string"),
    "content_type": LargeList(Value("string")),
    "raw_answers": LargeList(Value("string")),
    "query_generator": Value("string"),
    "query_generation_pipeline": Value("string"),
    "source_type": Value("string"),
    "query_type_for_generation": Value("string"),
    "answer": Value("string"),
})

qrels_features = Features({
    "query_id": Value("int64"),
    "corpus_id": Value("int64"),
    "score": Value("int64"),
    "content_type": LargeList(Value("string")),
    "bounding_boxes": LargeList({
        "annotator": Value("int64"),
        "x1": Value("int64"),
        "x2": Value("int64"),
        "y1": Value("int64"),
        "y2": Value("int64"),
    }),
})

dataset = {
    "queries": load_dataset(dataset_name, data_dir="queries", split="test", features=queries_features),
    "qrels": load_dataset(dataset_name, data_dir="qrels", split="test", features=qrels_features),
    "corpus": load_dataset(dataset_name, data_dir="corpus", split="test"),
}

query_sample = dataset["queries"][10]
print('Query:', query_sample['query'])
print("Answer:", query_sample['answer'])

# %%
related_qrels = dataset["qrels"].filter(lambda x: x['query_id'] == query_sample['query_id'])


# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_bbox(image, bboxes):
    _, ax = plt.subplots(figsize=(18, 12))
    ax.imshow(image), ax.axis('off')
    for bbox in bboxes:
        rect = patches.Rectangle((bbox['x1'], bbox['y1']), bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1'], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

for qrel in related_qrels:
    plot_bbox(dataset["corpus"][qrel['corpus_id']]['image'], qrel['bounding_boxes'])

# %%
