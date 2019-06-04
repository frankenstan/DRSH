# DRSH
Deep Recurrent Scaling Hashing

Code for paper Supervised Hashing with Recurrent Scaling. Apperas in 3rd Asia Pacific Web and Web-Age Information Management Joint Conference on Web and Big Data (APWeb-WAIM), 2019.

# Requirements
* theano
* keras
* python 2.7

# Run the codes
To run the training/test codes you should download the raw images of NUS-WIDE dataset [here](https://drive.google.com/file/d/0B7IzDz-4yH_HMFdiSE44R1lselE/view?usp=sharing) first, and put the images in one folder. Then using the trained VGG-16 caffemodel to extract features. The attributes are extracted through the finetuned caffemodel training on attributes.

The finetuned caffemodel for extracting attributes can be downloaded [here](https://drive.google.com/open?id=1oV489jhiiTspdPlv4D2zTTYHyxtZctC0 "With a Title"). The file that contains labels of attributes of COCO dataset for extracting NUS-WIDE images' attributes can be downloaded [here](https://drive.google.com/open?id=1aZezvxZmCIXogpTqyJbqMQrERVvZFnXD "With a Title"). The classified h5 file for finding positive and negative triplets can be downloaded [here](https://drive.google.com/open?id=1D78n_zBaDe494RgmsqkR7HIbPok8R_s8 "With a Title"). You also should download the 300-dimension GloVe vectors.

After getting the files mentioned above, run the __NUSex_att.py__, __e_x.py__ and __transfer.py__ to get __NUSGlove.h5__. Run the __NUSex_fea.py__ to get __NUSF.h5__.

# Models
The models for generating hash codes in different code lengths can be downloaded [here](https://drive.google.com/open?id=1M0KUA0KWqim_0fvu5ZNX1yIUHB1u76Jp)

#Reference
If you find this codebase useful in your research, please consider citing the following paper:

    @InProceedings{Fu2019DRSH,
      author = {Fu, Xiyao and Bin, Yi and Wang, Zheng and Wei, Qin and Chen, Si},
      title = {Supervised Hashing with Recurrent Scaling},
      booktitle = {The Asia Pacific Web (APWeb) and Web-Age Information Management (WAIM) Joint Conference on Web and Big Data(APWeb-WAIM)},
      month = {August},
      year = {2019}
    }
