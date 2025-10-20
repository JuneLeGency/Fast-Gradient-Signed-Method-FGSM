import React, { useState } from 'react';
import './App.css';

const API_BASE_URL = "/api";

function App() {
  const [activeTab, setActiveTab] = useState('normal');

  // State for Normal Prediction Tab
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [predictionResult, setPredictionResult] = useState('预测结果将显示在这里...');
  const [isLoading, setIsLoading] = useState(false);

  // State for FGSM Attack Tab
  const [epsilon, setEpsilon] = useState(0.05);
  const [fgsmResult, setFgsmResult] = useState(null);
  const [isAttacking, setIsAttacking] = useState(false);

  // State for Targeted Attack Tab
  const [progress, setProgress] = useState(0);
  const [targetedResult, setTargetedResult] = useState(null);
  const [isTargetAttacking, setIsTargetAttacking] = useState(false);
  const [targetedStatus, setTargetedStatus] = useState('等待开始...');

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPredictionResult('图片已选择，请点击“开始识别”。');
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setPredictionResult('错误：请先选择一个图片文件。');
      return;
    }

    setIsLoading(true);
    setPredictionResult('正在识别中，请稍候...');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/predict/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP 错误! 状态: ${response.status}`);
      }

      const data = await response.json();
      if (data.error) {
        setPredictionResult(`识别出错: ${data.error}`);
      } else {
        setPredictionResult(data.predictions);
      }
    } catch (error) {
      console.error("Prediction error:", error);
      setPredictionResult(`请求后端服务时出错: ${error.message}。`);
    }
    setIsLoading(false);
  };

  const handleFgsmAttack = async () => {
    setIsAttacking(true);
    setFgsmResult(null);
    try {
      const response = await fetch(`${API_BASE_URL}/attack/fgsm/?epsilon=${epsilon}`);
      if (!response.ok) {
        throw new Error(`HTTP 错误! 状态: ${response.status}`);
      }
      const data = await response.json();
      if (data.error) {
        alert(`攻击出错: ${data.error}`);
        setFgsmResult(null);
      } else {
        setFgsmResult(data);
      }
    } catch (error) {
      console.error("FGSM attack error:", error);
      alert(`请求后端服务时出错: ${error.message}。`);
    }
    setIsAttacking(false);
  };

  const handleTargetedAttack = () => {
    setIsTargetAttacking(true);
    setTargetedResult(null);
    setProgress(0);
    setTargetedStatus('正在连接到后端服务...');

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/api/attack/targeted_ws`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setTargetedStatus('连接成功，开始执行迭代攻击 (约1-2分钟)...');
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'progress') {
        setProgress(message.value);
        setTargetedStatus(`攻击正在进行中... ${Math.round(message.value * 100)}%`);
      } else if (message.type === 'result') {
        setTargetedResult(message.data);
        setTargetedStatus('攻击完成！');
      } else if (message.type === 'error') {
        setTargetedStatus(`发生错误: ${message.message}`);
        setIsTargetAttacking(false);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setTargetedStatus('WebSocket 连接出错，请检查后端服务是否正在运行。');
      setIsTargetAttacking(false);
    };

    ws.onclose = () => {
      setIsTargetAttacking(false);
      if (progress < 1) { // If closed prematurely
          setTargetedStatus('连接已断开');
      }
    };
  };

  const renderNormalPredictionTab = () => (
    <div className={`tab-pane ${activeTab === 'normal' ? 'active' : ''}`}>
      <h2>正常图像识别</h2>
      <p>上传一张图片，测试AI模型在正常情况下的识别能力。</p>
      <div className="controls">
          <input type="file" accept="image/*" onChange={handleFileChange} className="file-input" />
          <button onClick={handlePredict} disabled={isLoading || !selectedFile}>
            {isLoading ? '识别中...' : '开始识别'}
          </button>
      </div>
      {preview && (
        <div>
          <img src={preview} alt="Preview" className="image-preview" />
        </div>
      )}
      <pre className="results-text">{predictionResult}</pre>
    </div>
  );

  const renderFgsmAttackTab = () => (
    <div className={`tab-pane ${activeTab === 'fgsm' ? 'active' : ''}`}>
        <h2>非定向攻击 (FGSM)</h2>
        <p>此功能将对固定的熊猫图片进行攻击，您可以调整扰动强度 (Epsilon) 来观察效果。</p>
        <div className="controls">
            <label>扰动强度 (Epsilon): {epsilon.toFixed(3)}</label>
            <input 
                type="range" 
                min="0" 
                max="0.2" 
                step="0.005" 
                value={epsilon} 
                onChange={(e) => setEpsilon(parseFloat(e.target.value))} 
            />
            <button onClick={handleFgsmAttack} disabled={isAttacking}>
                {isAttacking ? '攻击中...' : '开始攻击'}
            </button>
        </div>

        {isAttacking && <p>正在生成对抗样本，请稍候...</p>}

        {fgsmResult && (
            <div>
                <div className="attack-container">
                    <div className="attack-column">
                        <h3>原始图像</h3>
                        <img src={`data:image/png;base64,${fgsmResult.original_image}`} alt="Original" className="attack-image" />
                        <pre className="results-text">{fgsmResult.original_text}</pre>
                    </div>
                    <div className="attack-column">
                        <h3>扰动 (放大后)</h3>
                        <img src={`data:image/png;base64,${fgsmResult.perturbation_image}`} alt="Perturbation" className="attack-image" />
                    </div>
                    <div className="attack-column">
                        <h3>对抗样本</h3>
                        <img src={`data:image/png;base64,${fgsmResult.adversarial_image}`} alt="Adversarial" className="attack-image" />
                        <pre className="results-text">{fgsmResult.adversarial_text}</pre>
                    </div>
                </div>
            </div>
        )}
    </div>
  );

  const renderTargetedAttackTab = () => (
      <div className={`tab-pane ${activeTab === 'targeted' ? 'active' : ''}`}>
          <h2>定向攻击</h2>
          <p>此功能将对固定的熊猫图片进行攻击，目标是让模型将其识别为“咖啡杯”。</p>
          <div className="controls">
              <button onClick={handleTargetedAttack} disabled={isTargetAttacking}>
                  {isTargetAttacking ? '攻击进行中...' : '开始定向攻击'}
              </button>
          </div>
          {isTargetAttacking && (
              <div style={{width: '80%', margin: '20px auto'}}>
                  <p>{targetedStatus}</p>
                  <progress value={progress} max="1" style={{width: '100%'}} />
              </div>
          )}
          {targetedResult && (
            <div>
                <div className="attack-container">
                    <div className="attack-column">
                        <h3>原始图像</h3>
                        <img src={`data:image/png;base64,${targetedResult.original_image}`} alt="Original" className="attack-image" />
                        <pre className="results-text">{targetedResult.original_text}</pre>
                    </div>
                    <div className="attack-column">
                        <h3>优化后的扰动</h3>
                        <img src={`data:image/png;base64,${targetedResult.perturbation_image}`} alt="Perturbation" className="attack-image" />
                    </div>
                    <div className="attack-column">
                        <h3>对抗样本</h3>
                        <img src={`data:image/png;base64,${targetedResult.adversarial_image}`} alt="Adversarial" className="attack-image" />
                        <pre className="results-text">{targetedResult.adversarial_text}</pre>
                    </div>
                </div>
            </div>
        )}
      </div>
  );

  return (
    <div className="App">
      <h1>AI 对抗攻击演示</h1>
      <div className="tab-container">
        <div className="tab-buttons">
          <button className={`tab-button ${activeTab === 'normal' ? 'active' : ''}`} onClick={() => setActiveTab('normal')}>正常识别</button>
          <button className={`tab-button ${activeTab === 'fgsm' ? 'active' : ''}`} onClick={() => setActiveTab('fgsm')}>非定向攻击</button>
          <button className={`tab-button ${activeTab === 'targeted' ? 'active' : ''}`} onClick={() => setActiveTab('targeted')}>定向攻击</button>
        </div>
        <div className="tab-content">
          {renderNormalPredictionTab()}
          {renderFgsmAttackTab()}
          {renderTargetedAttackTab()}
        </div>
      </div>
    </div>
  );
}

export default App;
