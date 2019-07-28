# face-detector-app

## Steps

Install Conda
https://www.anaconda.com/distribution/

Create a new environment and activate
```
conda create -n tf-keras python=3.7
conda activate tf-keras
```
Clone the repo
```
git clone https://github.com/mohankumarSriram/face-detector-app.git
```

- Download label images [here](https://drive.google.com/open?id=1JfMAWG0R80dTKfkJxVyUCKAbnmD270Vv)
- Download pre-trained models [here](https://drive.google.com/open?id=1aa8tAh-cXqwjCuILzzsmdIMJc9PNyXaz)

```
unzip ~/Downloads/label_images.zip -d ./
unzip ~/Downloads/models.zip -d ./
```

Install requiremnts
```
pip install -r requirements.txt
```

Launch App
```
python serve.py
```

Note: Please refer to the training process in this [repo](https://github.com/mohankumarSriram/face-verification) 
