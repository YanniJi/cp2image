# cp2image

This repository contains the code of the CP2Image [CP2Image: Generating high-quality single-cell images using CellProfiler representations](https://proceedings.mlr.press/v227/ji24a.html)

The code is based on [tueimage/cytoVAE](https://github.com/tueimage/cytoVAE), with modification for model achitecture.

---

**Abstract:**

*Single-cell high-throughput microscopy images contain key biological information underlying normal and pathological cellular processes. Image-based analysis and profiling are powerful and promising for extracting this information but are made difficult due to substantial complexity and heterogeneity in cellular phenotype. Hand-crafted methods and machine learning models are popular ways to extract cell image information. Representations extracted via machine learning models, which often exhibit good reconstruction performance, lack biological interpretability. Hand-crafted representations, on the contrary, have clear biological meanings and thus are interpretable. Whether these hand-crafted representations can also generate realistic images is not clear. In this paper, we propose a CellProfiler to image (CP2Image) model that can directly generate realistic cell images from CellProfiler representations. We also demonstrate most biological information encoded in the CellProfiler representations is well-preserved in the generating process. This is the first time hand-crafted representations be shown to have generative ability and provide researchers with an intuitive way for their further analysis.

---

## Citation
```
@InProceedings{pmlr-v227-ji24a,
  title = 	 {CP2Image: Generating high-quality single-cell images using CellProfiler representations},
  author =       {Ji, Yanni and Cutiongco, Marie and Jensen, Bj\orn Sand and Yuan, Ke},
  booktitle = 	 {Medical Imaging with Deep Learning},
  pages = 	 {274--285},
  year = 	 {2024},
  volume = 	 {227},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {10--12 Jul},
  publisher =    {PMLR}
   }
```

## Repository overview
- **training_cytoVAE.ipynb**: Notebook to run the training process.
- **models/**: The architecture, loss objective, optimizer of the model
- **dataManagers/**: Batch generation of input images and CellProfiler representations
- **exp_cytoVAE_demo/**: Configuration file for a given experiment

## Environment
- **python** 3.8.2
- **tensorflow-gpu** 1.15.0


