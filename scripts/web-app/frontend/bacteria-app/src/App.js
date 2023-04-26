import './App.css';
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'react-dropzone-uploader/dist/styles.css';
import Dropzone from 'react-dropzone-uploader';
import { getDroppedOrSelectedFiles } from 'html5-file-selector';
import FileUploadComponent from './components/fileUpload.component';
// import { CSVLink } from 'react-csv';
// import { CSVReader } from 'react-papaparse';
import Papa from 'papaparse';


const imageMimeType = /image\/(png|jpg|jpeg|tif)/i;


function App() {
  const [bacteria, setBacteria] = useState([]);
  const [threshold, setThreshold] = useState(0.3);
  const [json, setJson] = useState(null);
  const [csvData, setCsvData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleThresholdUpdate = () => {
    const requestOptions = {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ threshold: threshold })
    };

    fetch('/threshold/', requestOptions)
      .then(response => response.json())
      .then(data => console.log(data));
  };

  const fileChangedHandler = (event) => {
    const files = event.target.files;
    const filteredFiles = [...files].filter((file) => file.type.match(imageMimeType));
    setBacteria([...bacteria, ...filteredFiles]);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    setLoading(true);
    const formData = new FormData();
    bacteria.forEach((file) => {
      formData.append("answer_images", file, file.name);
    });

    const requestOptions = {
      method: 'POST',
      body: formData
    };

    fetch("http://127.0.0.1:8000/detect/", requestOptions)
      .then(res => res.json())
      .then(function(res) {
        console.log(res);
        setJson(res);

        // Parse the CSV data and update the state variable
        Papa.parse(res.features, {
          header: true,
          complete: (results) => {
            setCsvData(results.data);
          },
        });

        setLoading(false);
      });
  };

  const downloadJSON = () => {
    const element = document.createElement("a");
    const file = new Blob([JSON.stringify(json)], { type: "application/json" });
    element.href = URL.createObjectURL(file);
    element.download = "output.json";
    document.body.appendChild(element);
    element.click();
  };

  const clearImages = () => {
    setBacteria([]);
    setJson(null);
    setCsvData(null);
    const form = document.querySelector('form');
    form.reset();
  };

  return (
    <div className="App">
      <div className="App-header">
        <h1>Bacteria App</h1>
      </div>
      <div className="App-choose-file">
        {bacteria.map((file, index) => (
          <img key={index} src={URL.createObjectURL(file)} title={file.name} alt={file.name} />
        ))}
  
        <form>
          <fieldset>
            <input onChange={fileChangedHandler} name="ChooseFile" type="file" accept=".tif, .png" multiple></input>
          </fieldset>
        </form>

        <input type="number" value={threshold} onChange={(e) => setThreshold(e.target.value)} />
        <button onClick={handleThresholdUpdate}>Update Threshold</button>

      </div>
  
      <div className="App-content">
        {bacteria.length > 0 && (
          <button className='App-button-clear-images' onClick={clearImages}>Clear</button>
        )}
        <div className='App-loading'>

          <button className='App-button-add-image' onClick={handleSubmit}
          title='Click to Launch the model'>
            Launch
          </button>

  
          {loading && (
            <img src="https://em-content.zobj.net/source/noto-emoji-animations/344/rocket_1f680.gif" 
            alt="launching..."/>
          )}
        </div>
  
        {json && (
          <button className='App-button-load-json' onClick={downloadJSON}>
            Download output.json
          </button>
        )}
      </div>
  
      <div className='App-img-result'>
        {json && json.image && (
          <div>
            {json.image.map((image, index) => (
              <img key={index} src={`data:image/png;base64,${image}`} alt={`Result image ${index}`} />
            ))}
          </div>
        )}
      </div>
  
      <div className='App-img-feature'>
        {json && json.box_id && (
          <div>
            {json.box_id.map((image, index) => (
              <img key={index} src={`data:image/png;base64,${image}`} alt={`Feature index image ${index}`} />
            ))}
          </div>
        )}
      </div>

      <div className='App-feature-result'>
        {csvData && (
          <table>
            <thead>
              <tr>
                {Object.keys(csvData[0]).map((key) => (
                  <th key={key}>{key}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {csvData.map((row, index) => (
                <tr key={index}>
                  {Object.values(row).map((value, index) => (
                    <td key={index}>{value}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

    </div>
  );  
}

export default App;
