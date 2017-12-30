## Approaches taken by each paper 

#### LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection

##### Points:
Auto Encoder trained on Normal proper signal
Detects Anomaly when input and output considerably different from the network.


#### Unsupervised Learning of Video Representations using LSTMs

##### Approaches:
Encoder - Decoder to trained to predict self as well as future signal
Combination of both self and future prediction networks, a Composite model produces best results.


#### HIERARCHICAL MULTISCALE RECURRENT NEURAL NETWORKS 

##### Approaches:
Hierarchical structure where model learns boundary by itself.
Novel: Flush, Copy and update operations

Useful when we don’t know how boundary exists in the given data
##### Advantages:
Long term dependencies due to lesser updates in Upper layer, Faster also
##### Points:
Less hidden units in bottom layer because it needs to capture small variations.


#### Learning to Generate Long-term Future via Hierarchical Prediction 

No Hierarchical RNN in this architecture
They use a Pose predictor and predict future poses
They utilize a simple LSTM 
