## CVAT
...

## LabelIMG
- Easy to install
- Bulk upload
- Correct Annotations
- User friendly

## VGG Image Annotator
- web-based
- correct annotations
- no Yolo format

## Supervisely
- Web based
- Advanced Annotation
- Variety of formats

# Labeling for YOLO
For every image, we will have a text file with the same name as image file has. For instance, if we have image with name image001, then we need to have text file with name image001.txt (or json).

Inside the text file, in one line for one object, we will have the following variables:
- class number
- centre in x
- centre in y
- object width and
- object height 

Then we have to normalize theses values, dividing them for the respective values:

- centre in x / image width
- centre in y / image height
- object width / image width
- object height / image height

Then, we got something like this:

Class   x       y       w       h
4       0.5356  0.5637  0.0150  0.0274
2       0.3187  0.5722  0.0154  0.0260

```
:~$ ffmpeg -i forest-road.mp4 -vf fps=4 image-%d.jpeg
```
> - 4 frames por segundo
> - o nome é image-n.jpeg, onde n é a iteração
> - modo verbose ativado
> - -i indica o vídeo de entrada