# MaterialNet-Multi-scale-Texture-Hierarchy-and-Multi-view-Surface-Reflectance-forMaterial-Type-Recog
- [MaterialNet: Multi-scale Texture Hierarchy and Multi-view Surface Reflectance forMaterial Type Recognition](https://bmvc2022.mpi-inf.mpg.de/361/)
- **Dongjin Lee**, Hyun-Cheol Kim, Jeongil Seo, Seungkyu Lee

## Motivation
<img width="60%" alt="image" src="https://user-images.githubusercontent.com/96943196/203362325-47da1906-62eb-4dc1-b836-8763c28699c7.png">

Certain material type is well categorized by its surface characteristics such as reflectance, stiffness, friction, roughness, and texture. While haptic properties are difficult to be estimated from visual data, texture could be easily observed from color image. Surface reflectance is another distinguishing property of a material that can be estimated from multiple viewpoint observations. We claim that texture features robust to environmental changes, their hierarchy along multiple scales (Figure 1. (a)), and surface reflectance (Figure 1. (b)) obtained from multi-view images can characterize material types comprehensively.

## Proposed Methods

<img width="85%" alt="image" src="https://user-images.githubusercontent.com/96943196/203363466-9996f236-e39b-4869-9f8e-77828ec238e1.png">

- **MSTH-Net** encodes view-independent comprehensive multi-scale textures and their hierarchy.
	- **MSTH-Net Part1** : Take both entire and salient features from each layer.
	- **MSTH-Net part2** : Enhance the salient features and build texture hierarchy.
- For the multi-view environment, **multi-view MSTH-Net(Figure 4. (b))** is constructed by collecting as many texture extractors (part 1) as the number of views MaterialNet accepts.


<br/>
<img width="85%" alt="image" src="https://user-images.githubusercontent.com/96943196/203363538-21f1f35f-0fca-4fbb-9275-635fc803ceb6.png">

- **MVSR-Net** encodes view-specific features revealing surface reflectance of a material type
- **Dual MaterialNet (Figure 5. (c))** : combination of MaterialNet and diff-MaterialNet.
	- **MaterialNet (Figure 5. (a))** : combination of MSTH-Net and MVSR-Net 
	- **Diff-MaterialNet (Figure 5. (b))** : network that has the same structure as MaterialNet, but receives a difference images of every two consecutive color images aligned by affine transformation before subtraction


## Poster
<img width="100%" src="https://user-images.githubusercontent.com/96943196/203357728-96254970-413e-4515-8767-5f552a1af12f.png"/>

## Citation
	@inproceedings{LEE_2022_BMVC,
	author    = {DONGJIN LEE and Seungkyu Lee},
	title     = {MaterialNet: Multi-scale Texture Hierarchy and Multi-view Surface Reflectance for Material Type Recognition},
	booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
	publisher = {{BMVA} Press},
	year      = {2022},
	url       = {https://bmvc2022.mpi-inf.mpg.de/0361.pdf}
	}
