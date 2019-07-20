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

- Download data [here](https://drive.google.com/open?id=1ydV6Mrx-WfShCjaf79dPWvpp85XqLziN)
- Download pre-trained models [here](https://drive.google.com/open?id=185ox0xeZNkisCUVoMQTIbVt4XyzpQIfc)

```
unzip ~/Downloads/data.zip -d ./
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
