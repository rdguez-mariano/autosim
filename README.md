# Automatic detection of repeated objects

This repository implements a method for automatic detection of repeated objects in images. The companion paper (and more about this method) can be found [here](https://rdguez-mariano.github.io/pages/autosim).

## Prerequisites

In order to quickly start using this code we advise you to install [anaconda](https://www.anaconda.com/distribution/) and follow this guide.

##### Creating a conda environment

```bash
conda create --name autosim python=3.6

source activate autosim

pip install --upgrade pip
pip install -r requirements.txt
```

##### Compiling the C++ library

```bash
mkdir -p build && cd build && cmake .. && make
```

## Reproducing results from the [companion paper](https://rdguez-mariano.github.io/pages/autosim)

Recreate figure 3 from the companion paper:

```bash
python fig_prelabeling.py
```

Detect best three repeated objects: 

```bash
python autosim.py -q coca.png -a im3_sub.png -l "build/libautosim.so" -m 0.8 -n 1000 -i 0 -r 4 -w 800
```

Args explanation:

- `-q coca.png`. Query image in which to detect repeated objects.
- `-a im3_sub.png.png`. A-contrario image for the [a-contrario matching criterion](https://rdguez-mariano.github.io/pages/hyperdescriptors) compatible with repeated structures. 
- `-l build/libautosim.so`. Path to the compiled libautosim library.
- `-m 0.8`. Matching threshold (between 0 and 1) for the a-contrario matching criterion.
- `-n 1000`. Maximum number of matches to be taken into account.
- `-i 0`. RANSAC information type: 0 - Typical RANSAC; 1 - [RANSAC 2pts](https://rdguez-mariano.github.io/pages/locate); 2 - [RANSAC affine](https://rdguez-mariano.github.io/pages/locate).
- `-r 4`. Rho as in [rho-hyperdescriptors](https://rdguez-mariano.github.io/pages/hyperdescriptors). It sets the threshold for considering several spatially close keypoints as a single keypoint.
- `-w 800`. It scales down the query image if its width is greater than 800.

## Authors

* **Mariano Rodríguez** - [web page](https://rdguez-mariano.github.io/)
* **Jean-Michel Morel**
* **Julie Delon** - [web page](https://delon.wp.imt.fr/)


## License

The code is distributed under the permissive MIT License - see the [LICENSE](LICENSE) file for more details.

## Uninstall

```bash
conda deactivate
conda-env remove -n autosim
rm -R /path/to/autosim
```

## Github repository

<https://github.com/rdguez-mariano/autosim>
