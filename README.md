# Classify Unseen Objects

[Project website](https://www.cg.tuwien.ac.at/courses/projekte/%E2%82%AC1000-Classify-Unseen-Objects)

We want to classify objects in scenes that contain many similar objects (e.g., in a factory hall, warehouse, or office). However, especially for non-standard objects, training these specific classes requires a lot of effort to create enough ground truth. Instead, we aim at classifying yet unseen objects automatically and label them just with weak supervision, i.e., a human-in-the-loop being queried for unknown classes for training on-the-fly.


The project implements a pipeline for object detection and discovery from pointcloud scenes and includes external projects for unsupervised object detection (UnScene3D) and feature extraction (Point-MAE). 

Detailed instructions for project setup in [SETUP.md](https://github.com/kohjakob/classify-unseen-objects/blob/master/SETUP.md).