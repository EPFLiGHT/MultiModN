# MultiModN
This repository is the official implementation of the NeurIPS 2023 paper titled ["MultiModN- Multimodal, Multi-Task, Interpretable Modular Networks"](https://arxiv.org/abs/2309.14118) written by Vinitra Swamy*, Malika Satayeva*, Jibril Frej, Thierry Bossy, Thijs Vogels, Martin Jaggi, Tanja Käser*, and Mary-Anne Hartley*.

## Model Architecture
<p align="center">
	<img src="https://docs.google.com/drawings/d/e/2PACX-1vQLCjRaSAQaDSqNhQhy7xMgbmxl_RzwToWchuc6t41hbTDMFdQO-yYIcp35qY3QUakUHOE4XgFu1L0q/pub?w=1531&h=510" />
</p>

We present MultiModN, a multimodal, modular network that fuses latent representations in a sequence of any number, combination, or type of modality while providing granular real-time predictive feedback on any number or combination of predictive tasks. MultiModN's composable pipeline is interpretable-by-design, as well as innately multi-task and robust to the fundamental issue of biased missingness. We perform four experiments on several benchmark MM datasets across 10 real-world tasks (predicting medical diagnoses, academic performance, and weather), and show that MultiModN's sequential MM fusion does not compromise performance compared with a baseline of parallel fusion. By simulating the challenging bias of missing not-at-random (MNAR), this work shows that, contrary to MultiModN, parallel fusion baselines erroneously learn MNAR and suffer catastrophic failure when faced with different patterns of MNAR at inference. To the best of our knowledge, this is the first inherently MNAR-resistant approach to MM modeling. 

**MultiModN provides granular insights, robustness, and flexibility without compromising performance.**

## Quick start
Quick start running MultiModN on Titanic example pipeline with a multilayer perceptron encoder:
```
./datasets/titanic/get_data.sh
python3 pipelines/titanic/titanic_mlp_pipeline.py
```
Open `pipelines/titanic/plots/titanic_mlp.png` to look at the training curves.

## MultiModN metrics
During training and evaluation, the metrics of the model are stored in a history at each epoch in a matrix of dimensions $(E+1) * D$, where E is the number of encoders and D the number of decoders. Each row represents the metrics for a target at each state of the model.

## Code structure
### MultiModN
`/multimodn` package contains the MultiModN model and its modules:
* Encoders
* Decoders
* State

### Datasets
`/dataset` package contains the MultiModDataset abstract class, compatible with MultiModN.

Specific **datasets** are added in the `/dataset` directory and must fulfill the following requirements:
* Contain a dataset class that inherit MultiModDataset or has a method to convert into a MultiModDataset
* Contain a `.sh` script responsible of getting the data and store it in `/data` folder

`__getitem__` function of MultiModDataset subclasses must yield elements of the following shape:

```
tuple
(
	data: [torch.Tensor],
	targets: numpy.ndarray,
	(optional) encoding_sequence: numpy.ndarray
)
```

namely a tuple containing an array of tensors representing the features for each subsequent encoder, a numpy array representing the different targets and optionally a numpy array giving the order in which to apply the encoders to the subsequent data tensors. Note: `data` and `encoding_sequence` must have the same length.

#### **Missing values**
The user should be able to choose to keep missing values (nan values). Missing values can be present in the tensors yielded by the dataset and are managed by MultiModN.

### Pipelines
```/pipeline``` package contains the training pipelines using MultiModN for Multi-Modal Learning. It follows the following steps:

* Create `MultiModDataset` and the `dataloader` associated
* Create the list of encoders according to the features shape of the MultiModDataset
* Create the list of decoders according to the targets of the MultiModDataset
* Init, train and test the MultiModN model
* Store the trained model, training history and save learning curves

## Contributing 

This code is provided for educational purposes and aims to facilitate reproduction of our results, and further research 
in this direction. We have done our best to document, refactor, and test the code before publication.

If you find any bugs or would like to contribute new models, training protocols, etc, please let us know. Feel free to file issues and pull requests on the repo and we will address them as we can.

## Citations
If you find this code useful in your work, please cite our paper:

```
Swamy, V., Satayeva M., Frej J., Bossy T., Vogels T., Jaggi M., Käser T., Hartley, M. (2023). 
MultiModN- Multimodal, Multi-Task, Interpretable Modular Networks. 
In: Proceedings of the 37th Advances in Neural Information Processing Systems (NeurIPS 2023).
```
