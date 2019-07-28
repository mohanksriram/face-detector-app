import React, { Component } from 'react';
import Webcam from "components/Camera/react-webcam"
import logo from "assets/img/reactlogo.png";
import axios from 'axios';
import { textAlign, flexbox, fontSize } from '@material-ui/system';
import Button from "components/CustomButtons/Button.jsx";
import Card from "components/Card/Card.jsx";


const centerButton = {
  justifyContent: 'center',
  alignSelf: 'center',
};

const Photo = (props) => (
  <div className="output" style={{display: 'inline-block'}}>
    <canvas id="photo" style={{width: "640px", height: "480px"}} ></canvas>
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
        imageSelected: false,
        isLoading: false,
        rawFile: null,
        computeTime: 0,
        faceRect: [30, 30, 50, 50],
        className: null
      }
    }

    setRef = webcam => {
      this.webcam = webcam;
    };

    drawRect = () => {
      const photo = document.getElementById('photo');
      var ctx = photo.getContext("2d");

      const faceRect = this.state.faceRect;

      // Green rectangle
      ctx.beginPath();
      ctx.lineWidth = "4";
      ctx.strokeStyle = "green";
      let left = Math.floor(faceRect[0]*(photo.width))
      let top = Math.floor(faceRect[1]*(photo.height))
      let right = Math.floor(faceRect[2]*(photo.width))
      let bottom = Math.floor(faceRect[3]*photo.height)

      console.log(`draw rect: ${top}, ${right}, ${bottom}, ${left}`)

      ctx.rect(left, top, right-left, bottom-top);
      ctx.stroke();
    }
  
    capture = () => {

      const imageSrc = this.webcam.getScreenshot();
      const faceRect = this.state.faceRect;
      console.log('obtained image src: ', imageSrc);
      const photo = document.getElementById('photo');
      let ctx = photo.getContext("2d");
      let img_buffer = new Image;
      this.drawRect();
      img_buffer.onload = function() {
        let imgWidth = img_buffer.width;
        let imgHeight = img_buffer.height;
        photo.width = imgWidth;
        photo.height = imgHeight;
        ctx.drawImage(img_buffer, 0, 0, imgWidth, imgHeight);
    }
    img_buffer.src = imageSrc;
      this.setState({
        file: imageSrc,
      }, this.handlePredictClick);
    };

    rtspCapture = () => {
      this.handlePredictClick();
    }

    renderPrediction() {
      const prediction = this.state.className

      const bold_style = {fontSize: "-webkit-xxx-large", fontWeight: "800", paddingTop: "130px"}
      
      if (prediction == null) {
        return (
          <span style={bold_style}> Please Come Closer</span>
        )
      }
      else {
        return (
        <span style={bold_style}> {prediction} </span>
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

      const name = payload.class_name;
      const matchingPhoto = document.getElementById('photo2');

      if (name != null) {
        matchingPhoto.setAttribute('src', `/serveImage/${name}`);
      } else {
        matchingPhoto.setAttribute('src', `/serveImage/stranger`);
      }

      this.setState({predictions: payload.predictions, 
        probs: payload.probs,
        className: payload.class_name,
        computeTime: payload.compute_time,
        faceRect: payload.face_rect,
        isLoading: false}, this.capture);
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
  
      const bg_color = this.state.className != null ? 'green' : 'yellow'

      return (
        <div style={{display: 'flex', backgroundColor: bg_color}}>
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