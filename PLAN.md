# Wiki Connect
Finding new connections between Wikipedia pages.

## Overview and Goals
This project aims to uncover hidden connections between Wikipedia pages that are not directly linked. While Wikipedia uses contextual linking to connect related topics within descriptions, some connections remain undiscovered because the related topics are not explicitly mentioned. The goal is to use deep learning and graph-based approaches to identify these hidden relationships and enhance the network of links between Wikipedia pages.

## Project Type
We have chosen to mix the **bring your own data** and  **bring my own method** project types. We will develop a dataset based on Wikipedia articles and use a combination of deep learning model to discover hidden connections between them. Although there are other approaches, trying to predict links between Wikipedia articles [3], they use a different methodology and do not leverage the latest advancements in graph neural networks.

## Project Description


### Problem Statement

Wikipedia uses contextual linking to connect related articles by making specific words within a page clickable. These links provide easy navigation between related topics. However, many connections between articles remain hidden because the related words aren't explicitly mentioned and therefore linked. This project aims to discover these hidden connections using deep learning models that identify related articles based on implicit content similarities.

### Dataset

We will create a program that generates a graph of Wikipedia articles, where nodes represent articles and edges represent contextual links between them. The graph will be seeded from a specific Wikipedia article, which will serve as the starting point. The initial page will be scanned for links, and the process will continue recursively.

### Approach

To capture the semantics of each article, we will use state-of-the-art text embedding models, such as BERT [2], to create meaningful vector representations of the content. We will then use graph neural networks (GNNs) [1] based on message passing to enhance these node representations by incorporating information from neighboring nodes. This will allow us to better understand typical linking patterns between articles. Finally, we will use a simple classifier model to predict the likelihood of connections between pairs of articles that are currently unconnected.


## Work Breakdown Structure

| Task                                       | Estimated Time  |
|--------------------------------------------|-----------------|
| **Dataset Generation**                     |                 |
| Dataset collection (Wikipedia web scraping)| 2 days          |
| Data preprocessing (parsing articles)      | 3 days          |
| Graph construction (initial links)         | 2 days          |
|                                            |                 |
| **Model Development**                      |                 |
| GNN design and implementation              | 7 days          |
| Model training and fine-tuning             | 5 days          |
| Model evaluation and testing               | 4 days          |
|                                            |                 |
| **Results Analysis**                       |                 |
| Application to visualize results           | 5 days          |
| Writing final report                       | 3 days          |
| Preparing final presentation               | 2 days          |

## Conclusion
The project will explore new methods for discovering hidden relationships between Wikipedia pages by leveraging NLP techniques and graph-based models. This approach could enhance Wikipedia's structure and provide a richer experience for its users by surfacing new connections between articles.

## References
1. **Zhang, Muhan, and Yixin Chen. "Link prediction based on graph neural networks." Advances in neural information processing systems 31 (2018).**
2. **Devlin, Jacob. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).**
3. **Milne, David, and Ian H. Witten. "Learning to link with wikipedia." Proceedings of the 17th ACM conference on Information and knowledge management. 2008.**


