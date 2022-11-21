const { Menu, dialog, app } = require('@electron/remote');
const { ipcRenderer } = require('electron');

const fs = require('fs');
const path = require('path');
const wav = require('node-wav');
const Meyda = require('meyda');
const tf = require('@tensorflow/tfjs-node');
const D3Node = require('d3-node');


const debug = true;


/**************** Variables *******************/
// List of Meyda features
let featuresList = [
    'rms',
    'zcr',
    'spectralRolloff',
    'spectralCentroid',
    'spectralSpread',
    'spectralSkewness',
    'spectralKurtosis',
    'spectralFlatness',
    'mfcc',
    'chroma',
    'loudness',
    'energy',
    'perceptualSharpness',
    'spectralSlope'
];
// List of features from Emo-soundscape
const finalFeatureSet = [
    'rms_mean',
    'rms_std',
    'zerocross_mean',
    'zerocross_std',
    'rolloff_mean',
    'rolloff_std',
    'centroid_mean',
    'centroid_std',
    'spread_mean',
    'spread_std',
    'skewness_mean',
    'skewness_std',
    'kurtosis_mean',
    'kurtosis_std',
    'flatness_mean',
    'flatness_std',
    'mfcc_mean_1',
    'mfcc_mean_2',
    'mfcc_mean_3',
    'mfcc_mean_4',
    'mfcc_mean_5',
    'mfcc_mean_6',
    'mfcc_mean_7',
    'mfcc_mean_8',
    'mfcc_mean_9',
    'mfcc_mean_10',
    'mfcc_mean_11',
    'mfcc_mean_12',
    'mfcc_mean_13',
    'mfcc_std_1',
    'mfcc_std_2',
    'mfcc_std_3',
    'mfcc_std_4',
    'mfcc_std_5',
    'mfcc_std_6',
    'mfcc_std_7',
    'mfcc_std_8',
    'mfcc_std_9',
    'mfcc_std_10',
    'mfcc_std_11',
    'mfcc_std_12',
    'mfcc_std_13',
    'chromagram_mean_1',
    'chromagram_mean_2',
    'chromagram_mean_3',
    'chromagram_mean_4',
    'chromagram_mean_5',
    'chromagram_mean_6',
    'chromagram_mean_7',
    'chromagram_mean_8',
    'chromagram_mean_9',
    'chromagram_mean_10',
    'chromagram_mean_11',
    'chromagram_mean_12',
    'chromagram_std_1',
    'chromagram_std_2',
    'chromagram_std_3',
    'chromagram_std_4',
    'chromagram_std_5',
    'chromagram_std_6',
    'chromagram_std_7',
    'chromagram_std_8',
    'chromagram_std_9',
    'chromagram_std_10',
    'chromagram_std_11',
    'chromagram_std_12',
    'loudness_mean',
    'loudness_std',
    'energy_mean',
    'energy_std',
    'perceptual_sharp_mean',
    'perceptual_sharp_std',
    'spectral_slope_mean',
    'spectral_slope_std'
];

// file io
let csvFile; // csv file
let csvFeatureData = []; // csv read feature data

let fileDir; // directory containing all the audio files
let fileList; // container for audio filename *
let n_clips;

// feature extraction
let normalizedAllFilesFeature; // container for normalized feature of all files *

// model Prediction
const INPUT_DIM = 74; // model input dimension
let prediction = []; // container for arousal and valence prediction of all files *

// audio metadata
const sampleRate = 44100;
const bufferSize = 1024; // 23ms
const hopSize = bufferSize / 2;
const windowingFunction = 'hanning';
const n_mfcc = 13;



/**********************************************/
// Select HTML elements
/**********************************************/
const audioFile = document.getElementById('audioName');

const audioElement = document.getElementById('audio');

const statusText = document.getElementById("status");

const fileBtn = document.getElementById("fileBtn");
fileBtn.onclick = getFileDir;

const extractBtn = document.getElementById("extractBtn");
extractBtn.onclick = getFeatures;

const predictBtn = document.getElementById('predictBtn');
predictBtn.onclick = predict;

const saveOutputBtn = document.getElementById("saveOutputBtn");
saveOutputBtn.onclick = saveToCSV;

// const saveSelectedFileBtn = document.getElementById("saveSelectedFileBtn");
// saveSelectedFileBtn.onclick = saveSelectedFile;
// plot();



/**********************************************/
// Load Models
/**********************************************/
const aroModel = await tf.loadLayersModel('../models/arousal/model.json');
const valModel = await tf.loadLayersModel('../models/valence/model.json');



/**********************************************/
// d3
/**********************************************/
const d3n = new D3Node();
const d3 = d3n.d3;

// create array of objects containing filename and prediction
function createDataObject(filenames, dataPoints) {
    return filenames.map((fn, i) => {
        return {
            name: fn,
            x: dataPoints[i][1],
            y: dataPoints[i][0]
        }
    })
}


// plot audio file and prediction on to 
async function plot() {
    const size = [600, 600];
    const height = size[1];
    const width = size[0];
    const padding = {vertical: 28.5, horizontal: 28.5};
    const radius = 4;

    const data = await createDataObject(fileList, prediction);

    // const data = [
    //     {name: 'audio1.wav', x: -1, y: -1},
    //     {name: 'audio2.wav', x: -0.75, y: -0.8},
    //     {name: 'audio3.wav', x: -0.5, y: -0.5},
    //     {name: 'audio4.wav', x: 0, y: 0.1},
    //     {name: 'audio5.wav', x: 0.5, y: 0.75},
    //     {name: 'audio6.wav', x: 1, y: 1}
    // ];

    // const xScale = d3.scaleLinear().domain(d3.extent(data, d => d.x)).nice()
    // .range([padding.horizontal, width - padding.horizontal]);

    // const yScale = d3.scaleLinear().domain(d3.extent(data, d => d.y)).nice()
    // .range([height - padding.vertical, padding.vertical]);
    
    const xScale = d3.scaleLinear().domain([-1, 1]).nice()
    .range([padding.horizontal, width - padding.horizontal]);

    const yScale = d3.scaleLinear().domain([-1, 1]).nice()
    .range([height - padding.vertical, padding.vertical]);

    const xAxis = g => g
    .attr("transform", `translate(0,${height - padding.vertical})`)
    .call(d3.axisBottom(xScale).ticks(null, "+"))
    .call(g => g.select(".domain").remove())
    .call(g => g.selectAll(".tick line")
        .filter(d => d === 0)
        .clone()
        .attr("y2", -height - padding.vertical - padding.vertical)
        .attr("stroke", "#ccc"))
    .call(g => g.append("text")
        .attr("fill", "#000")
        .attr("x", padding.vertical)
        .attr("y", 5)
        .attr("dx", "55em")
        .attr("dy", "-1em")
        .attr("text-anchor", "end")
        .attr("font-weight", "bold")
        .text("Valence"));

    const yAxis = g => g
    .attr("transform", `translate(${padding.horizontal},0)`)
    .call(d3.axisLeft(yScale).ticks(null, "+"))
    .call(g => g.select(".domain").remove())
    .call(g => g.selectAll(".tick line")
        .filter(d => d === 0)
        .clone()
        .attr("x2", width - padding.horizontal - padding.horizontal)
        .attr("stroke", "#ccc"))
    .call(g => g.append("text")
        .attr("fill", "#000")
        .attr("x", 5)
        .attr("y", padding.horizontal)
        .attr("dy", "0.32em")
        .attr("text-anchor", "start")
        .attr("font-weight", "bold")
        .text("Arousal"));

    const color = d3.scaleSequential(d3.interpolateRdBu).domain([1, -1]);

    const svg = d3.select('svg')
    .attr('viewBox', [0, 0, width, height]);

    svg.append('g')
    .call(xAxis);
    
    svg.append('g')
    .call(yAxis)

    svg.append("g")
      .attr("stroke", "#000")
      .attr("stroke-opacity", 0.2)
    .selectAll("circle")
    .data(data)
    .join("circle")
      .attr("cx", d => xScale(d.x))
      .attr("cy", d => yScale(d.y))
      .attr("fill", d => color(d.x))
      .attr("r", radius)
      .on('mouseover', mouseOver)
      .on('mouseout', mouseOut)
      .on('click', mouseClick);
    
    function mouseOver(d) {
        d3.select(this)
        .attr('fill', 'grey')
        .attr('r', radius * 2);

        svg.append("text")
        .attr('id', `${d.name}`)
        .attr('class', 'hover')
        .attr('x', () => xScale(d.x) - 30)
        .attr('y', () => yScale(d.y) - 15)
        .text(() => d.name);
    }
    
    function mouseOut(d) {
        d3.select(this)
        .attr('fill', d => color(d.x))
        .attr('r', radius);

        d3.selectAll('.hover').remove();
    }

    function mouseClick(d) {
        const name = `${fileDir}/${d.name}`;
        audioFile.innerHTML = d.name;
        audioElement.src = name;
    }
}



/**********************************************/
// Fuctions
/**********************************************/

/****************** File io *******************/
// get all files inside directory
async function getFileDir() {
    fileDir = null;
    fileList = null;
    n_clips = null;

    try {
        const selectedDir = await dialog.showOpenDialog({
            title: 'Open Directory',
            defaultPath: app.getPath('desktop'),
            buttonLabel: 'Open',
            properties: ['openDirectory'],
        });
        fileDir = selectedDir.filePaths;
        const canceled = selectedDir.canceled;
        
        fileList = fs.readdirSync(fileDir[0]);

        // check and remove files that doesn't have .wav ext
        fileList = fileList.filter(s => s.substring(s.lastIndexOf('.')+1, s.length) == 'wav')
        n_clips = fileList.length;

        if (debug) {
            console.log(fileDir);
            console.log(canceled);
            console.log(fileList);
        }
        statusText.innerText = 'Directory Selected! Please Extract Feature!';
        extractBtn.disabled = false;
    } catch (err) {
        console.log(err);
    }
}


// save feature and prediction to csv files
async function saveToCSV() {
    let featureFilePath;
    let featureCanceled;
    let predictFilePath;
    let predictCanceled;
    const featureFileName = 'Output_Normalized_Features';
    const predictFileName = csvFile ? csvFile.split('/').slice(-1).toString().slice(0,-4) + '_prediction': 'Output_Prediction'; // get csv filename without path and extension

    try {
        // select folder to export feature if csv file not selected
        if (normalizedAllFilesFeature && !csvFile) {
            const selectFeatureFolder = await dialog.showSaveDialog({
                title: 'Save Feature File',
                defaultPath: path.join(app.getPath('desktop'), featureFileName) || path.join(__dirname, '../../../export/', featureFileName),
                buttonLabel: 'Save',
                filters: [
                    {
                        name: 'CSV Files',
                        extensions: ['csv']
                    }
                ],
                properties: [
                    'createDirectory',
                    'showOverwriteConfirmation'
                ]
            });
            featureFilePath = selectFeatureFolder.filePath;
            featureCanceled = selectFeatureFolder.canceled;
        }

        // select folder to export prediction
        if (prediction.length !== 0) {
            const selectPredictFolder = await dialog.showSaveDialog({
                title: 'Save Prediction File',
                defaultPath: path.join(app.getPath('desktop'), predictFileName) || path.join(__dirname, '../../../export/', predictFileName),
                buttonLabel: 'Save',
                filters: [
                    {
                        name: 'CSV Files',
                        extensions: ['csv']
                    }
                ],
                properties: [
                    'createDirectory',
                    'showOverwriteConfirmation'
                ]
            });
            predictFilePath = selectPredictFolder.filePath;
            predictCanceled = selectPredictFolder.canceled;
        }

        if (debug) {
            console.log(`feature filepath: ${featureFilePath}`);
            console.log(`prediction filepath: ${predictFilePath}`);
            console.log(`feature cancelled: ${featureCanceled}`);
            console.log(`prediction cancelled: ${predictCanceled}`);
        }
        
        // write feature data to csv
        if (normalizedAllFilesFeature && !featureCanceled && !csvFile) {
            const featureData = createCSVContent(normalizedAllFilesFeature, 'feature');
            fs.writeFile(featureFilePath.toString(), featureData, err => {
                if (err) throw err;
            })
        }
        
        // write prediction data to csv
        if (prediction.length !== 0 && !predictCanceled) {
            const predictData = createCSVContent(prediction, 'predict');
            debug ? console.log(predictData) : '';
            fs.writeFile(predictFilePath.toString(), predictData, err => {
                if (err) throw err;
            })
        }

        statusText.innerText = 'CSV File Saved!';
        extractBtn.disabled = true;
        predictBtn.disabled = true;
        saveOutputBtn.disabled = true;
    } catch (err) {
        console.log(err);
    }
} 


// create csv content
function createCSVContent(array, option) {
    debug ? console.log(array[0]) : '';
    let content;
    let len = array.length;

    switch (option) {
        case 'feature':
            content = ',' + finalFeatureSet.toString() + '\n';
            break;
        case 'predict':
            content = ',' + 'arousal,valence\n';
            break;
        default:
            throw "createCSVContent() parameter 1: 'feature' | 'predict' "
    }
    if (csvFile) {
        for (let i=0; i<len; ++i) {
            content = content + ',' + array[i].toString() + '\n';
        }
    } else {
        for (let i=0; i<len; ++i) {
            content = content + fileList[i] + ',' + array[i].toString() + '\n';
        }
    }
    return content;
}



/****************** Features ******************/
// get features
async function getFeatures() {
    let audioData; // audio file data
    let signal; // audio signal from audio data
    normalizedAllFilesFeature = null;
    
    // config Meyda
    Meyda.sampleRate = sampleRate;
    Meyda.windowingFunction = windowingFunction;
    Meyda.numberOfMFCCCoefficients = n_mfcc;
    
    try {
        let allfilesFeatureStats = [];
        
        statusText.innerText = 'Extracting Features ...';
        fileList.forEach(e => {
            // read wav file
            const path = fileDir + '/' + e;
            const buffer = fs.readFileSync(path);
            audioData = wav.decode(buffer);
            signal = audioData.channelData[0];

            // check if signal length is the power of 2
            let paddedSig;
            if (!isPowerOf2(signal.length)) {
                const len = signal.length;
                const targetPower = Math.ceil(Math.log2(len));
                // const newLen = Math.pow(2, targetPower);
                const truncLen = Math.pow(2, (targetPower - 1));

                // if ((newLen - len) < (len - truncLen)) {
                //     const padLen = newLen - len;
                //     const zeros = new Float32Array(padLen);

                //     paddedSig = new Float32Array(newLen);
                //     paddedSig.set(signal);
                //     paddedSig.set(zeros, len);
                // } else {
                //     paddedSig = signal.subarray(0, truncLen);
                // }
                
                paddedSig = signal.subarray(0, truncLen);
            } else {
                paddedSig = signal;
            }
            // extract through signal
            let featureContainer = [];

            const sigLen = paddedSig.length;
            for (let i=0; i<sigLen; i+=bufferSize) {
                const currentSig = paddedSig.subarray(i, i+bufferSize)
                let extractedFeatures = Meyda.extract(featuresList, currentSig);
                featureContainer.push(extractedFeatures);

                // debug ? console.log(extractedFeatures) : '';
            }
            
            if (debug) {
                console.log(e);
                console.log(featureContainer.length);
            }
            // get mean & std of all features
            const featureStats = featureContainer.length != 0 ? getStats(featureContainer) : console.log('No features extracted yet');
            allfilesFeatureStats.push(featureStats);
        })
        normalizedAllFilesFeature = normalizeFeature(allfilesFeatureStats); // normalize to the max of each feature
        
        if (debug) {
            console.log('all feature stats', allfilesFeatureStats);
            console.log('normalized all feature stats', normalizedAllFilesFeature);
        }
        
    } catch (err) {
        console.log(err);
    }
    statusText.innerText = 'Feature Extracted! Please Predict Emotion!';
    predictBtn.disabled = false;
}


// get Mean & Std of features
function getStats(featureContainer) {
    /* 
    featureContainer: [
        {rms, zcr, spectralRolloff, ...}
        {rms, zcr, spectralRolloff, ...}
        {rms, zcr, spectralRolloff, ...}
        .
        .
        .
    ] // time series of extracted feature objects

    *****************************
    return: Float32Array(n_features * 2)
        [ rms_mean, rms_std, zcr_mean, zcr_std, spectralRolloff_mean, spectralRolloff_std, ... ]
    */

    if (!Array.isArray(featureContainer)) {
        throw 'Cannot get stats, getStats() parameter 0 not an array';
    }
    const len = featureContainer.length;
    const n = finalFeatureSet.length;
    // debug ? console.log(len, n) : '';

    // Put features into their corresponding array
    let stats = [];
    let featureSet = {
        rms: [],
        zcr: [],
        rolloff: [],
        centroid: [],
        spread: [],
        skewness: [],
        kurtosis: [],
        flatness: [],
        mfcc1: [],
        mfcc2: [],
        mfcc3: [],
        mfcc4: [],
        mfcc5: [],
        mfcc6: [],
        mfcc7: [],
        mfcc8: [],
        mfcc9: [],
        mfcc10: [],
        mfcc11: [],
        mfcc12: [],
        mfcc13: [],
        chroma1: [],
        chroma2: [],
        chroma3: [],
        chroma4: [],
        chroma5: [],
        chroma6: [],
        chroma7: [],
        chroma8: [],
        chroma9: [],
        chroma10: [],
        chroma11: [],
        chroma12: [],
        loudness: [],
        energy: [],
        sharpness: [],
        spectSlope: [],
    } // 37

    featureContainer.forEach(e => {
        featureSet.rms.push(e.rms);
        featureSet.zcr.push(e.zcr);
        featureSet.rolloff.push(e.spectralRolloff);
        featureSet.centroid.push(e.spectralCentroid);
        featureSet.spread.push(e.spectralSpread);
        featureSet.skewness.push(e.spectralSkewness);
        featureSet.kurtosis.push(e.spectralKurtosis);
        featureSet.flatness.push(e.spectralFlatness);
        featureSet.mfcc1.push(e.mfcc[0]);
        featureSet.mfcc2.push(e.mfcc[1]);
        featureSet.mfcc3.push(e.mfcc[2]);
        featureSet.mfcc4.push(e.mfcc[3]);
        featureSet.mfcc5.push(e.mfcc[4]);
        featureSet.mfcc6.push(e.mfcc[5]);
        featureSet.mfcc7.push(e.mfcc[6]);
        featureSet.mfcc8.push(e.mfcc[7]);
        featureSet.mfcc9.push(e.mfcc[8]);
        featureSet.mfcc10.push(e.mfcc[9]);
        featureSet.mfcc11.push(e.mfcc[10]);
        featureSet.mfcc12.push(e.mfcc[11]);
        featureSet.mfcc13.push(e.mfcc[12]);
        featureSet.chroma1.push(e.chroma[0]);
        featureSet.chroma2.push(e.chroma[1]);
        featureSet.chroma3.push(e.chroma[2]);
        featureSet.chroma4.push(e.chroma[3]);
        featureSet.chroma5.push(e.chroma[4]);
        featureSet.chroma6.push(e.chroma[5]);
        featureSet.chroma7.push(e.chroma[6]);
        featureSet.chroma8.push(e.chroma[7]);
        featureSet.chroma9.push(e.chroma[8]);
        featureSet.chroma10.push(e.chroma[9]);
        featureSet.chroma11.push(e.chroma[10]);
        featureSet.chroma12.push(e.chroma[11]);
        featureSet.loudness.push(e.loudness.total / 24);
        featureSet.energy.push(e.energy);
        featureSet.sharpness.push(e.perceptualSharpness);
        featureSet.spectSlope.push(e.spectralSlope);
    })
    // debug ? console.log(featureSet) : '';
    
    // Get mean and std of each feature
    for (let i=0; i<n/2; ++i) {
        stats.push(mean(featureSet[Object.keys(featureSet)[i]]));
        stats.push(std(featureSet[Object.keys(featureSet)[i]]));
    }
    // debug ? console.log(stats) : '';
    
    // Return array of feature statistics as Float32Array
    return new Float32Array(stats)
}


// normalize feature set
function normalizeFeature(allFeatureStats) {
    /* 
    allFeatureStats: [
        Float32Array(74),
        Float32Array(74),
        Float32Array(74),
        .
        .
        .
    ] // Array(n_clips): feature statistics of all clips

    *****************************
    return: Array(n_clips)
        [ Float32Array(74), Float32Array(74), Float32Array(74),  ... ]
    */

    let max = new Array(INPUT_DIM).fill(0);
    let min = new Array(INPUT_DIM).fill(0);

    statusText.innerText = 'Normalizing Feature ...';
    // get max of indices 0 - 73 of all array
    const len = allFeatureStats.length;
    for (let i=0; i<len; ++i) {
        for (let j=0; j<INPUT_DIM; ++j) {
            if (allFeatureStats[i][j] > max[j]) max[j] = allFeatureStats[i][j];
            if (allFeatureStats[i][j] < min[j]) min[j] = allFeatureStats[i][j];
        }
    }
    // normalize data base on the max of each index
    const normalized = allFeatureStats.map(array => {
        return array.map((n, i) => {
            const norm = max[i] - min[i];
            return (n + min[i]) / norm;
        });
    });

    if (debug) {
        console.log(max);
        console.log(min);
    }

    return normalized
}



/************** Model Prediction **************/
// predict
async function predict() {
    let arousal; // arousal score
    let valence; // valence score
    prediction = [];

    statusText.innerText = 'Predicting ...';
    try {
        const allData = normalizedAllFilesFeature ? normalizedAllFilesFeature : csvFeatureData;
        if (allData) {
            statusText.innerText = 'Prediction Emotion ...';
            for (const array of allData) {
                const data = Array.from(array);
                const input = tf.tensor(data).reshape([1,74]);
                arousal = await aroModel.predict(input).data();
                valence = await valModel.predict(input).data();

                if (debug) {
                    console.log(data);
                    console.log(arousal, valence);
                }

                prediction.push([arousal[0], valence[0]]);
            }
            statusText.innerText = 'Prediction Complete! Save Feature and Prediction to CSV?';
            saveOutputBtn.disabled = false;
            debug ? console.log(prediction) : '';

            plot();
        } else {
            console.log('No feature data, select csv or extract feature from audio directory');
        }
    } catch (err) {
        console.log(err);
    }
}



/************** Emotion Taxonomy **************/
// get emotion description base of Russell's circumplex model of affect
function getEmotion(prediction) {
    ;
}



/************* Math Calculations **************/
// check if number is the power of 2
function isPowerOf2(v) {
    return v && !(v & (v - 1));
}


// mean of an array
function mean(a) {
    if (!Array.isArray(a)) throw 'mean() parameter 0 not an array';
    let n = a.length;
    if (n === 0) return 0;
    return (a.reduce((prev, curr) => prev + curr) / n);
}


// standard deviation of an array
function std(a) {
    if (!Array.isArray(a)) throw 'std() parameter 0 not an array';
    const n = a.length;
    if (n === 0) return 0;
    const m = a.reduce((prev, curr) => prev + curr) / n; // calculate mean
    return Math.sqrt(a.map(x => Math.pow(x - m, 2)).reduce((prev, curr) => prev + curr) / n);
}
