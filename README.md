# Curriculum Knowledge Switching for Pancreas Segmentation
Pancreas segmentation is very difficult since pancreas occupies only a very small fraction less than 0.5\% of a CT volume and suffers from high anatomical variability. Most existing methods use a two-stage framework: the coarse and the fine. We argue that both stages have the same purpose of improving pancreatic-pixel classification accuracies. Inspired by this observation, we transfer fine-model weights to the coarse. If we directly copy the pre-trained fine model, the performance is low due to the domain gap of the different input images, so we further propose a momentum update strategy for transferring models. Our momentum update stands on the other observation that input images are in three different domains: the small image cropped by the ground-truth bounding box ($D_1$), the small image cropped by the coarse predicted bounding box ($D_2$), and the large raw image ($D_3$). The momentum update training approach of the coarse model is cast into three steps: train the coarse model by $D_1$ firstly, by $D_2$ secondly, and by $D_3$ thirdly. In the three steps, we copy the model weights step-by-step with the momentum update approach in order to improve the coarse accuracy. The coarse benefits from domain adaptively since the first-step model is trained with strong supervision of the ground-truth bounding box and has a good pancreatic pixel-wise accuracy. The second and the third steps gradually adapt to domain $D_3$ that is the true domain of the coarse model. A higher detection accuracy produces a better region proposal and it renders the fine obtain a better segmentation accuracy. We conduct several experiments on the NIH dataset with different neural network backbones and the results show we obtain the state-of-the-art performance in terms of DSC metric.

# Experiment
```sh
bash f2.sh
```

# Citation
We appreciate it if you cite the following paper:
```
@InProceedings{TangMICCAI2022,
  author =    {Yumou Tang and Kun Zhan and Zhibo Tian and Mingxuan Zhang and Saisai Wang and Xueming Wen},
  title =     {Momentum update for pancreas segmentation},
  booktitle = {ICASSP},
  year =      {2023}
}

```

# Contact
https://kunzhan.github.io/

If you have any questions, feel free to contact me. (Email: `ice.echo#gmail.com`)
