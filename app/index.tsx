// wfefewfwe
import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { RNCamera } from 'react-native-camera';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';

export default function App() {
  const [isModelReady, setIsModelReady] = useState(false);
  const [predictedText, setPredictedText] = useState('');
  let model: tf.GraphModel;

  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready(); // Prepare TensorFlow
        const modelJson = require('./model/model.json');
        const modelWeights = require('./model/weights.bin');
        model = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
        setIsModelReady(true);
      } catch (error) {
        console.error('Error loading model:', error);
      }
    };
    loadModel();
  }, []);

  // Define preprocessImage function here
  const preprocessImage = (frame: any): tf.Tensor | null => {
    try {
      return tf.browser
        .fromPixels(frame)
        .resizeBilinear([224, 224])
        .expandDims(0)
        .toFloat()
        .div(255.0);
    } catch (error) {
      console.error('Error during image preprocessing:', error);
      return null;
    }
  };

  // Define handleCameraFrame function here
  const handleCameraFrame = async (frame: any) => {
    if (!isModelReady) return;

    const imageTensor = preprocessImage(frame);

    if (!imageTensor) {
      console.error('Invalid image tensor');
      return;
    }

    try {
      const prediction = model.predict(imageTensor);

      if (prediction instanceof tf.Tensor) {
        const text = interpretPrediction(prediction);
        setPredictedText(text);
      } else {
        console.error('Prediction output is not a Tensor:', prediction);
      }
    } catch (error) {
      console.error('Error during prediction:', error);
    }
  };

  // Replace the existing interpretPrediction function with this one
  const interpretPrediction = (tensor: tf.Tensor): string => {
    let predictionArray: Float32Array;
  
    if (tensor.dtype === 'float32') {
      predictionArray = tensor.dataSync() as Float32Array;
    } else if (tensor.dtype === 'int32') {
      predictionArray = new Float32Array(tensor.dataSync() as Int32Array);
    } else if (tensor.dtype === 'bool') {
      // Convert boolean tensors to Float32Array
      const boolArray = tensor.dataSync() as Uint8Array;
      predictionArray = new Float32Array(boolArray.map(value => (value ? 1 : 0)));
    } else {
      throw new Error(`Unsupported tensor dtype: ${tensor.dtype}`);
    }
  
    const maxIndex = predictionArray.indexOf(Math.max(...predictionArray));
    const gestures = ['Kumusta', 'Mahal Kita', 'Salamat', 'Okay']; // Replace with your gestures
    return gestures[maxIndex] || 'Unknown';
  };

  return (
    <View style={styles.container}>
      {isModelReady ? (
        <>
          <RNCamera
            style={styles.camera}
            type={RNCamera.Constants.Type.front}
            captureAudio={false}
            onPictureTaken={() => {}}
          />
          <Text style={styles.text}>{predictedText}</Text>
        </>
      ) : (
        <Text style={styles.text}>Loading model...</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  camera: {
    width: '100%',
    height: '70%',
  },
  text: {
    fontSize: 24,
    margin: 10,
    textAlign: 'center',
  },
});
