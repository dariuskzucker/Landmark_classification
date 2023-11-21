# European Landmark Classification

## Goal: Binary classifcation between the Hofsburg Imperial Palace and the Pantheon.

Hofsburg:

![image](https://github.com/dariuskzucker/Landmark_classification/assets/33701468/eab51cef-8a83-44c9-9161-227d20373300)

Pantheon:

![image](https://github.com/dariuskzucker/Landmark_classification/assets/33701468/f1e4fb97-5024-46d1-ac27-dc4655de1254)


## Implementation:
- Self-trained CNN model using Resnet-18 architecture.
- Reaches over 97% test AUROC on the landmark binary classification task.


#### Data augmentation: Random rotation, grayscales, flips, colorjitter, and more. Played around with a combination of these.

##### Before Data Augmentation. Notice how model is overfitting to particular features in the sky.

<img width="346" alt="Screenshot 2023-11-21 at 1 47 06 PM" src="https://github.com/dariuskzucker/Landmark_classification/assets/33701468/fd875005-0913-4b7c-90de-e10f5540ebc6">
<img width="340" alt="Screenshot 2023-11-21 at 1 47 36 PM" src="https://github.com/dariuskzucker/Landmark_classification/assets/33701468/bcb62a27-115c-452f-bff4-e2664459d0a9">

##### After Data Augmentation. Reduced overfitting to the sky.

<img width="340" alt="Screenshot 2023-11-21 at 1 48 39 PM" src="https://github.com/dariuskzucker/Landmark_classification/assets/33701468/cb3bcaa2-e753-4eaa-a3e9-cbb9a51d2b9c">
<img width="342" alt="Screenshot 2023-11-21 at 1 48 22 PM" src="https://github.com/dariuskzucker/Landmark_classification/assets/33701468/21b14a53-cad6-4dd4-96fa-d65c28751634">


#### Transfer learning: Trained a source model to classify 8 other landmarks (The Colosseum, Petronas Towers, Rialto Bridge, Museu Nacional d'Art de Catalunya, St Stephen's Cathedral in Vienna, Berlin Cathedral, Hagia Sophia, Gaudi Casa Batllo in Barcelona).
Froze convolutional layers and retrained on binary target task.



#### Hyperparameter tuning: Used a large weight decay to prevent to prevent overfitting, StepLR scheduler to employ a decreasing learning rate, experimented freezing different source model layers. Found cross entropy loss with Adam optimizer to work best for this task.

#### Model architectures: Implemented a custom CNN with 3 convolutional layers. Ended up shuffling between Resnet architectures, with Resnet-18 providing the best balance between bias and variance.

#### Neural Net Heatmap Visualizations: Used [Grad-Cam]([url](https://github.com/jacobgil/pytorch-grad-cam)) to visualize what CNN was learning. Found evidence of overfitting and changed accordingly.



