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
        confidence: 0,
        predictions: ['nobody'],
        probs: [0.2, 0.5, 0.8],
        imageSelected: false,
        isLoading: false,
        rawFile: null,
        computeTime: 0,
        faceRect: [30, 30, 50, 50],
        svc_class: 'nobody'
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
      let top = faceRect[0]
      let right = faceRect[1]
      let bottom = faceRect[2]
      let left = faceRect[3]

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
      img_buffer.onload = function() {
        let imgWidth = img_buffer.width;
        let imgHeight = img_buffer.height;
        photo.width = imgWidth;
        photo.height = imgHeight;
        ctx.drawImage(img_buffer, 0, 0, imgWidth, imgHeight);
        
        // Draw Face
        // ctx.beginPath();
        // ctx.strokeStyle = "green";
        // ctx.rect(faceRect[0], faceRect[1], faceRect[2], faceRect[3]);
        // ctx.stroke()
    }
    img_buffer.src = imageSrc;
      // ctx.drawImage(imageSrc, 0, 0)
      // photo.setAttribute('src', imageSrc);
      this.setState({
        file: imageSrc,
      }, this.handlePredictClick);
    };

    renderPrediction() {
      const predictions = this.state.predictions
      const probs = this.state.probs
      const computeTime = this.state.computeTime

      const bold_style = {fontSize: "-webkit-xxx-large", fontWeight: "800", paddingTop: "130px"}
      if (predictions.length == 1) {
        return (
          <span style={bold_style}> Please Come Closer</span>
        )
      }

      if (predictions.length > 0) {

        return (
        <span style={bold_style}> {this.state.svc_class} </span>
        )


          // // const predictionItems = predictions.map(function (ele, i) {
          // //   return <li>{ele} ({probs[i]}) </li> 
          // // });

          // return (
          //     <ul>
          //         {predictionItems}
          //         <span> prediction time is: </span> {computeTime} <span> seconds</span>
          //         <ul>
          //         <span>Confidence: </span> {this.state.confidence}
          //         </ul>
          //     </ul>
          // )

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
        svc_class: payload.svc_class,
        computeTime: payload.compute_time,
        confidence: payload.confidence,
        faceRect: payload.face_rect,
        isLoading: false}, this.capture);
      console.log(payload)
      // this.capture();
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
  
      const bg_color = this.state.predictions.length > 1 ? 'green' : 'yellow'

      // console.log(`bg_color: ${bg_color}`)
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