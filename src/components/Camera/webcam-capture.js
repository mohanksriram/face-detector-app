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
        confidence: 0,
        predictions: ['disha', 'suriya', 'mohan'],
        probs: [0.2, 0.5, 0.8],
        imageSelected: false,
        isLoading: false,
        rawFile: null,
        computeTime: 0,
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
      }, this.handlePredictClick);
    };

    renderPrediction() {
      const predictions = this.state.predictions
      const probs = this.state.probs
      const computeTime = this.state.computeTime

      if (predictions.length > 0) {

          const predictionItems = predictions.map(function (ele, i) {
            return <li>{ele} ({probs[i]}) </li> 
          });

          return (
              <ul>
                  {predictionItems}
                  <span> prediction time is: </span> {computeTime} <span> seconds</span>
                  <ul>
                  <span>Confidence: </span> {this.state.confidence}
                  </ul>
              </ul>
          )

      } else {
          return (
            <ul>
              <span>Confidence: </span> {this.state.confidence}
            </ul>
          )
      }
  }


    handlePredictClick = async (event) => {
      console.log('calling handlePredict')
      this.setState({isLoading: true});
      let resPromise = null;
      console.log('current url for prediction: ', this.state.file)

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
      photo.setAttribute('src', `/serveImage/${payload.svc_class}`);
      this.setState({predictions: payload.predictions, 
        probs: payload.probs,
        computeTime: payload.compute_time,
        confidence: payload.confidence,
        isLoading: false});
      console.log(payload)
    } catch (e) {
        alert(e)
    }
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
          {this.renderPrediction()}
          </Card>

          <Card profile>
          <Photo />
          <div>
          <MatchingPhoto />
          </div>
          </Card>          
        </div>
      );
    }
  }