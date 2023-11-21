# Landmark_classification

Self-trained CNN model using Resnet-18 architecture.
Reaches over 97% test AUROC on the landmark binary classification task.


Data augmentation: Random rotation, grayscales, flips, colorjitter, and more. Played around with a combination of these.
Transfer learning: Trained a source model to classify 8 other landmarks. Froze convolutional layers and retrained on binary target task.
Hyperparameter tuning: Used a large weight decay to prevent to prevent overfitting, StepLR scheduler to employ a decreasing learning rate, experimented freezing different source model layers. Found cross entropy loss with Adam optimizer to work best for this task.
Model architectures: Implemented a custom CNN with 3 convolutional layers. Ended up shuffling between Resnet architectures, with Resnet-18 providing the best balance between bias and variance.
Neural Net Heatmap Visualizations: Used [Grad-Cam]([url](https://github.com/jacobgil/pytorch-grad-cam))'s 


Colosseum, Petronas Towers, Rialto Bridge, Museu Nacional d'Art de Catalunya, St Stephen's Cathedral in Vienna, Berlin Cathedral, Hagia Sophia, Gaudi Casa Batllo in Barcelona
