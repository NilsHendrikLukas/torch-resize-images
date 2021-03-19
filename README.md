# Torch Resize Images

An easy-to-use, command line tool for resizing and center-cropping images using PyTorch.


**Tested for:** 

``
python=3.7
``

**Install:**

`
git clone https://github.com/NilsHendrikLukas/torch-resize-images  
`

`
cd torch-resize-images  
`

`
pip -r install requirements.txt  
`


**Usage:**

`
python resize_images.py --root <path_to_images> --output_dir <path_to_output> 
`


**Optional Parameters:**

[--width] Output image width. Default: 224

[--height] Output image height. Default: 224

[--no-center-crop] Do not center crop the image to the desired width and height. 

[-n, --n_procs] Number of parallel processes to run. Default: 1  

[--overwrite] Overwrite the output file if it exists. Default: True

[--gpu] Specify the GPU index. If none is specified, the GPU with the most available memory is automatically chosen.
