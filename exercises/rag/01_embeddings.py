import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from langchain_aws.embeddings import BedrockEmbeddings
from utils import get_attendance_id

plt.style.use('./skiller-whale.mplstyle')

# Exercise 1 - Embeddings
#
# This exercise visualizes the embeddings of a few text strings.
# Since the original embeddings are 256 dimensions, it uses a machine learning technique
#   called PCA to reduce the dimensions to 2D so we can plot them.
#
# Part 1
#   * Read through the code and run it.
#   * It will save a figure of the embeddings in 2D space in `embeddings.png`.
#   * Open the figure (you can do so from VSCode directly).
#   * The original embeddings are 256 dimensions
#       - you can uncomment the print statements to see them.
#
# Part 2
#   * Add different food items to the `text` list.
#   * What foods are close together in the embedding space?
#
#   * [Optional] If you have time, try exploring queries/texts of your own.
embeddings = BedrockEmbeddings(
    model_id='amazon.titan-embed-text-v2:0',
    endpoint_url='https://bedrock-runtime.aws-proxy.skillerwhale.com/',
    region_name='eu-west-1',
    aws_access_key_id=get_attendance_id(),
    aws_secret_access_key='<unused>',
    model_kwargs={
        'dimensions': 256
    }
)

# You can add more text items to this list.
text = [
    'banana',
    'apple',
    'I love bananas and apples',
]

print('Computing embeddings for text.')

# Compute embeddings for each text.
embeds = embeddings.embed_documents(text)

# Reduce from 256 dimensions to 2D using PCA.
pca = PCA(n_components=2)
pca_embeds = pca.fit_transform(np.array(embeds))

# You can uncomment these to see the original and transformed embeddings.
# print()
# print(f"Original embedding for {text[0]}: {[float(f'{e:.3f}') for e in embeds[0]]}")
# print()
# print(f'Transformed embedding for {text[0]}: {[float(f"{e:.3f}") for e in pca_embeds[0]]}')

# Compute similarities between each pair of texts.
for i, t1 in enumerate(text):
    for j, t2 in enumerate(text[i+1:], start=i+1):
        sim = np.dot(embeds[i], embeds[j]) / (np.linalg.norm(embeds[i]) * np.linalg.norm(embeds[j]))
        print(f'Similarity between "{t1}" and "{t2}": {sim:.3f}')

# Plot and label, save to file.
fig, ax = plt.subplots()
ax.scatter(pca_embeds[:,0], pca_embeds[:,1])
for emb, txt in zip(pca_embeds, text):
    ax.annotate(txt, (emb[0] + .01, emb[1] + .01))

plt.xlabel('Embedding Dimension 1')
plt.ylabel('Embedding Dimension 2')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.tight_layout()

plt.savefig('embeddings.png')
print('Figure saved to embeddings.png.')
