import { useState, useRef } from 'react'
import axios from 'axios'
import { Upload, FileImage, Activity, Loader2 } from 'lucide-react'
import DarkVeil from './DarkVeil'
import GlassIcons from './GlassIcons'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [isDragging, setIsDragging] = useState(false)

  const fileInputRef = useRef(null)

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) processFile(selectedFile)
  }

  const processFile = (selectedFile) => {
    setFile(selectedFile)
    setPreview(URL.createObjectURL(selectedFile))
    setResult(null)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile && droppedFile.type.startsWith('image/')) {
      processFile(droppedFile)
    }
  }

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
    } catch (error) {
      console.error("Error:", error);
      alert("Erreur serveur");
    } finally {
      setLoading(false);
    }
  };

  // --- CONFIGURATION DU BOUTON GLASS ---
  const glassButtonItems = [
    {
      // On change l'icône selon si ça charge ou non
      icon: loading ? <Loader2 className="spin" /> : <Activity size={30} />,
      color: 'cyan',
      label: loading ? 'Analyse...' : 'Lancer le Diagnostic',
      align: 'center',
      onClick: handleSubmit,
      disabled: loading || !file // Désactivé si pas de fichier ou chargement en cours
    }
  ];

  return (
    <div className="main-wrapper">
      <div className="background-layer">
        <DarkVeil speed={0.2} noiseIntensity={0.2} scanlineIntensity={0.4} />
      </div>

      <div className="content-layer">
        <div className="glass-card">
          <h1 style={{
            margin: 0,
            color: '#1a1a1a',
            fontFamily: "'Cinzel', serif",
            fontWeight: '400',
            fontSize: '3rem',
            letterSpacing: '2px'
          }}>
            PULMONAR AI
          </h1>
          
          <p style={{
            color: '#555',
            fontSize: '0.8rem',
            letterSpacing: '2px',
            marginBottom: '20px',
            fontFamily: "'Cinzel', serif",
          }}>
            SYSTEME DE DIAGNOSTIC RADIOLOGIQUE
          </p>

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            accept="image/*"
            style={{ display: 'none' }}
          />

          <div
            className={`upload-zone ${isDragging ? 'dragging' : ''}`}
            onClick={() => fileInputRef.current.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {preview ? (
              <img src={preview} alt="Aperçu" className="preview-img" style={{ maxHeight: '200px', borderRadius: '8px' }} />
            ) : (
              <>
                <Upload className="icon-upload" />
                <p style={{ fontFamily: "'Cinzel', serif" }}> Glissez une radio ici</p>
              </>
            )}
          </div>

          {/* --- BOUTON GLASS --- */}
          <div style={{ marginTop: '20px', marginBottom: '10px' }}>
            <GlassIcons items={glassButtonItems} />
          </div>

          {/* --- SECTION RESULTAT --- */}
          {result && (
            <div className={`result-box ${result.diagnosis === 'Normal' ? 'result-good' : 'result-bad'}`}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', justifyContent: 'center' }}>
                <FileImage size={20} />
                <h2 style={{
                  margin: '5px 0',
                  fontSize: '1.2rem',
                  fontFamily: "'Cinzel', serif",
                  fontWeight: 'normal'
                }}>
                  {result.diagnosis.toUpperCase()}
                </h2>
              </div>

              {/* Barre de progression */}
              <div style={{ background: '#ddd', borderRadius: '10px', height: '6px', width: '100%', marginTop: '5px' }}>
                <div style={{
                  width: result.confidence,
                  background: result.diagnosis === 'Normal' ? '#1a8a1a' : '#cc2222',
                  height: '100%',
                  borderRadius: '10px',
                  transition: 'width 1s ease-in-out'
                }}></div>
              </div>

              <p style={{
                fontSize: '0.8rem',
                marginTop: '5px',
                fontFamily: "'Cinzel', serif"
              }}>
                Confiance IA : {result.confidence}
              </p>
            </div>
          )}

        </div> {/* Fin glass-card */}
      </div> {/* Fin content-layer */}
    </div> /* Fin main-wrapper */
  )
}

export default App