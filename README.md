# CenterNet on WIDER 2019 pedestrian detection
This proj are forked from [here](https://github.com/xingyizhou/CenterNet)

# Features  
- a pedestrian detection challenger for [WIDER 2019 track 2](https://competitions.codalab.org/competitions/22852#results)
- new heat map loss for center point localization 
using [Weighted Hausdorff Distance](https://arxiv.org/pdf/1806.07564.pdf)
- new bounding box regression loss, 
using [Bounded IOU Loss](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tychsen-Smith_Improving_Object_Localization_CVPR_2018_paper.pdf)

# train

```bash
bash pdet_train.sh
```
# test
```bash
bash pdet_test.sh
```

# val score

`model`: DLA34

|  experiment  |  AP |
|--------------|-----|
|resol_512_512 |0.41 |
|resol_384_768 |0.45+|
|resol_448_896 |0.533|

# Glossary 4 Development
- `batch`/`output`/``: a dictionary, keys: 
    + `input`, processed images in a batch
    + `hm` for heatmap
    + `reg` offset term: 
    + `reg_mask` regression boolean mask
    + `wh` width and height
    + `ind` indices of the target's coordinates, shape: batch×max_obj
    + `target` batch×max_obj×dim
    
output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
      
new task: `haus`, Hausdorff + Bounded IOU

