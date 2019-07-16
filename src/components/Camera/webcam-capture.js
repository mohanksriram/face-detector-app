import React, { Component } from 'react';
import Webcam from "components/Camera/react-webcam"
import logo from "assets/img/reactlogo.png";
import axios from 'axios';
import { textAlign, flexbox } from '@material-ui/system';
import Button from "components/CustomButtons/Button.jsx";
import Card from "components/Card/Card.jsx";


const centerButton = {
  justifyContent: 'center',
  alignSelf: 'center',
};

const Photo = (props) => (
  <div className="output" style={{display: 'inline-block'}}>
    <img id="photo" alt="Your photo" src={'/logo'} style={{width: '640px', height: '480px'}}/>
  </div>
);

const MatchingPhoto = (props) => (
  <div className="output" style={{display: 'inline-block'}}>
    <img id="photo2" alt="Your photo" src={'/logo'} style={{width: '640px', height: '480px'}}/>
  </div>
);


export default class WebcamCapture extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
        file: null,
        predictions: [],
        imageSelected: false,
        isLoading: false,
        rawFile: null,
      }
    }

    setRef = webcam => {
      this.webcam = webcam;
    };
  
    capture = () => {
      const imageSrc = this.webcam.getScreenshot();
      console.log('obtained image src: ', imageSrc);
      const photo = document.getElementById('photo');
      photo.setAttribute('src', imageSrc);
      this.setState({
        file: imageSrc,
      });
    };


    // justpredict = () => {
    //   const imageSrc = this.webcam.getScreenshot();
    //   console.log('setting raw file up for prediction');
    //  const photo = document.getElementById('photo');
    //  console.log('photo source is: ', imageSrc);
    //   this.setState({
    //     rawFile: imageSrc,
    //     file: imageSrc,
    //     isLoading: true,
    //   });
    // };


    // handlePredictClick = () => {

    // }

    handlePredictClick = async (event) => {
      this.setState({isLoading: true});
      let resPromise = null;
      console.log('current url for prediction: ', this.state.file)

      // axios.post('/api/classify', {
      //   imageBase64: this.state.file
      // })

      // axios({
      //   method: 'get',
      //   url: '/api/classify',
      //   data: {
      //     imageBase64: this.state.file,
      //   }
      // });

      resPromise = axios.get('/api/classify', {
        params: {
            url: this.state.file
        }
    });


    try {
      const res = await resPromise;
      const payload = res.data;
      console.log('received payload: ', payload)
      const photo = document.getElementById('photo2');
      photo.setAttribute('src', `/serveImage/${payload.prediction_name}`);
      // this.setState({prediction_name: payload.prediction_name, isLoading: false});
      console.log(payload)
    } catch (e) {
        alert(e)
    }
      // try {
      //     const res = await resPromise;
      //     const payload = res.data;

      //     this.setState({predictions: payload.predictions, isLoading: false});
      //     console.log(payload)
      // } catch (e) {
      //     alert(e)
      // }
  };
  
    render() {
      const videoConstraints = {
        width: 1280,
        height: 720,
        facingMode: "user"
      };
  
      return (
        <div style={{display: 'flex'}}>
          <Card profile>
          <Webcam
            audio={false}
            height={480}
            ref={this.setRef}
            screenshotFormat="image/jpeg"
            width={640}
            videoConstraints={videoConstraints}
          />
          <Button color="primary" round onClick={ this.capture } style={centerButton}>Capture Snap</Button>
          </Card>

          <Card profile>
          <Photo />
          <div>
          {/* <Button color="primary" round>
                Follow
              </Button> */}
          <Button color="primary" round id="saveButton" onClick={ this.handlePredictClick } style={centerButton}>Get Predictions</Button>
          
          </div>
          </Card>

          <Card profile>
              <MatchingPhoto />
          </Card>
          
        </div>
      );
    }
  }