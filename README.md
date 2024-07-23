
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<br />
<p align="center">
    <img src="src/banner_small.png">
</p>



## About The Project
This repository provides the network implementation and training routines for the paper "From EMBER to FIRE: predicting high resolution baryon fields from dark matter simulations with Deep Learning".
The code is written using the Tensorflow2 API, is easy to use and supports parallel training on multiple GPUs.
Simulations are part of the [FIRE project](https://fire.northwestern.edu).


### Networks
Pretrained networks and prediction maps can be found at [Google Drive](https://drive.google.com/drive/folders/10_7Y3xjwHeZFdX6Fm5luhl-lMkVKH63k?usp=sharing).


### Prerequisites and Usage
Note that the you may need to modify the code for your specific project application.

1. Clone the repo
   ```sh
   git clone https://github.com/maurbe/ember.git
   ```
3. Install dependencies
   ```sh
   pip install numpy, matplotlib, tensorflow-gpu
   ```
4. Set the datapaths in `module.py`
5. Train the network
   ```sh
   python train.py
   ```

If you make use of the project in any way, please cite
```sh
@article{10.1093/mnras/stab3088,
    author = {Bernardini, M and Feldmann, R and Anglés-Alcázar, D and Boylan-Kolchin, M and Bullock, J and Mayer, L and Stadel, J},
    title = "{From EMBER to FIRE: predicting high resolution baryon fields from dark matter simulations with Deep Learning}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    year = {2022},
    month = {1},
    issn = {0035-8711},
    doi = {10.1093/mnras/stab3088},
    url = {https://doi.org/10.1093/mnras/stab3088}
}
```

## License

Distributed under the MIT License.
