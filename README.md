# Chinese character "recognition"

## Part 1

For the data preparation, I first compared the `info.json` and `train.jsonl` files. Realising that all the needed information was available in `train.jsonl`, I proceeded to only use that. The function `get_meta()` basically just retrieves all the needed information from `train.jsonl`, and discards everything else. 

In `get_bboxes()`, the bounding box annotations for each image are retrieved. I used the `'adjusted_bbox'` annotations as that was [recommended on the dataset website](https://ctwdataset.github.io/tutorial/1-basics.html#Appendix:-Adjusted-bounding-box-conversion). To calculate the opposing corners of the bounding boxes I used [parts of a script on the dataset github](https://github.com/yuantailing/ctw-baseline/blob/master/classification/create_pkl.py#L20). `get_boxes()` returns a dictionary where each key is a file name, and the corresponding value is a nested list of corner coordinates of the bounding boxes. The bounding boxes containing non-chinese characters are ignored. 

To get the labels of each image, I tried many different approaches in order to make it efficient. The `get_labels()` function uses code provided in [this stackoverflow answer](https://stackoverflow.com/a/62235347/14112047), and is fairly quick compared to the other approaches I tried (~3h vs ~4 days). `get_labels()` checks, for each image, which points are in a/multiple bounding box/-es, resulting in a huge True/False array. This array is then reshaped, converted from boolean to binary, summed, and, finally, clipped at 1, to get the labels for each image. The function returns a similar structure to `get_boxes`, a dictionary, where each key is a file name, and the corresponding value is the label array for that file. 

After that, I loaded the images using a function, `get_imgs()` (also known as `get_data()`), from a previous assignment (can't find the assignment on Asad's or Axel's Github, but [here it is on mine](https://github.com/sagahansson/lt2316-h20-b/blob/main/ab.ipynb)). I modified `get_imgs()` slightly to return a dictionary, where each key is a file name, and the corresponding value is an array representation of that image. I then used `train_test_split` from `sklearn` to split the data into train, validation and test sets. 

## Part 2

To load the data for the networks, I used `DataLoader` for each set. 

I was interested in implementing CNN architechtures that I wasn't familiar with, so I chose Lenet first. In the model, I wanted to use `nn.Sequential`, since I hadn't used that before, and thought it'd be good practice. In order to implement the model, I used a few different tutorials, combining them into what is now the model. Due to memory limitations, I had to significantly reduce the feature/kernel sizes of each layer. After implementing the `Lenet()` model, I wrote the `train()` and `test()` functions, again with the assitance of a few tutorials. 

I then chose another CNN architechture that I hadn't explored previously, AlexNet. For practice, I wanted to keep using `nn.Sequential`, which turned out to be a good idea as AlexNet has quite a few convolutional/pooling blocks. Again, I consulted a number of tutorials, combined and modified them to suit the `train()` and `test()` functions. As with the `Lenet()` model, I had to reduce the size of the features/kernels. For the `Alexnet()` model, the feature/kernel sizes were originally so large that I decided to divide them all by 16, which worked well. 

## Part 3

For evaluation, I decided to use BCE and accuracy

- mention adam? 


- alexnet:
-- smalldata:
--- avgpool: Accuracy: 0.27647, Loss: 0.01311
--- no:      Accuracy: 0.33159, Loss: 0.10957
--halfdata:
--- avgpool:
--- no:
-- alldata: 
--- avgpool: Accuracy: 0.33159, Loss: 0.10957
--- no:      Accuracy: 0.33159, Loss: 0.10957


- examine some imgs: they are all the same colour, indicated that the model only predicts 0s. understandable cos of unbalanced data, next time balance it