import React, { useState, useEffect } from "react";
import axios from "axios";

function App() {
  const [image, setImage] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [outputImage, setOutputImage] = useState(null); // Gambar hasil deteksi YOLO
  const [edgeImage, setEdgeImage] = useState(null); // Gambar hasil deteksi tepi
  const [histogram, setHistogram] = useState(null); // Data histogram
  const [isLoading, setIsLoading] = useState(false); // Indikator loading

  // Fungsi untuk meng-handle input gambar
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith("image/")) { // Validasi tipe file
      setImage(file);
    } else {
      alert("Please select a valid image file");
    }
  };

  // Fungsi untuk mengirim gambar ke server dan mendapatkan prediksi
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!image) return alert("Please select an image");

    const formData = new FormData();
    formData.append("file", image);

    try {
      setIsLoading(true); // Set loading menjadi true sebelum mengirim permintaan
      const response = await axios.post("http://127.0.0.1:8000/predict/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      // Simpan hasil prediksi di state
      setPredictions(response.data.predictions);
      setOutputImage(`data:image/jpeg;base64,${response.data.image}`);
      setEdgeImage(`data:image/jpeg;base64,${response.data.edges}`);
      setHistogram(response.data.histogram);
    } catch (error) {
      console.error("Error during prediction:", error);
      alert("Error during prediction");
    } finally {
      setIsLoading(false); // Set loading menjadi false setelah permintaan selesai
    }
  };

  // Fungsi untuk menggambar histogram pada canvas
  useEffect(() => {
    if (histogram) {
      const canvas = document.getElementById("histogramCanvas");
      const ctx = canvas.getContext("2d");

      // Bersihkan canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Tentukan ukuran grafik
      const canvasWidth = canvas.width;
      const canvasHeight = canvas.height;
      const binWidth = canvasWidth / histogram.length;

      // Cari nilai maksimum pada histogram untuk normalisasi tinggi bar
      const maxVal = Math.max(...histogram);

      // Gambar histogram
      ctx.fillStyle = "blue";
      histogram.forEach((value, index) => {
        const barHeight = (value / maxVal) * canvasHeight;
        ctx.fillRect(index * binWidth, canvasHeight - barHeight, binWidth, barHeight);
      });
    }
  }, [histogram]);

  return (
    <div className="App">
      <h1>Image Upload and Prediction</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleImageChange} />
        <button type="submit" disabled={isLoading}>
          {isLoading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {isLoading && <p>Loading...</p>} {/* Indikator loading */}

      {outputImage && (
        <div>
          <h2>Detected Image:</h2>
          <img src={outputImage} alt="Detected" style={{ maxWidth: "100%" }} />
        </div>
      )}

      {edgeImage && (
        <div>
          <h2>Edge Detection (Canny):</h2>
          <img src={edgeImage} alt="Edge Detection" style={{ maxWidth: "100%" }} />
        </div>
      )}

      {histogram && (
        <div>
          <h2>Histogram:</h2>
          <canvas id="histogramCanvas" width="600" height="400"></canvas>
        </div>
      )}

      {predictions && (
        <div>
          <h2>Predictions:</h2>
          <ul>
            {predictions.map((prediction, index) => (
              <li key={index}>
                Class: {prediction.class}, Confidence: {(prediction.confidence * 100).toFixed(2)}%, 
                Bounding Box: (X_min: {prediction.x_min.toFixed(2)}, Y_min: {prediction.y_min.toFixed(2)}, 
                X_max: {prediction.x_max.toFixed(2)}, Y_max: {prediction.y_max.toFixed(2)})
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
