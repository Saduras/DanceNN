# DanceNN

The idea is to use an existing neural network to predict poses from dancing videos and use this as training data.
Another neural net is then supposed to learn from the audio and estimated poses to predict poses/dance animation for unseen songs.

## Generating pose data

For this step I uses [eldar's implementation for tensorflow](https://github.com/eldar/pose-tensorflow).
 
I run the pose estimation for every frame and save them to an csv.

[Example of preditected poses over original video here!](https://youtu.be/rVGG7zHLbQs)
