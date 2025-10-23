import React, { useState, useEffect } from 'react';
import Select from 'react-select';
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
  const [fgsmFile, setFgsmFile] = useState(null);

  // State for Targeted Attack Tab
  const [targetClassList, setTargetClassList] = useState([]);
  const [selectedTarget, setSelectedTarget] = useState({ value: 504, label: '咖啡杯 (coffee mug)' });
  const [progress, setProgress] = useState(0);
  const [targetedResult, setTargetedResult] = useState(null);
  const [isTargetAttacking, setIsTargetAttacking] = useState(false);
  const [targetedStatus, setTargetedStatus] = useState('等待开始...');
  const [targetedFile, setTargetedFile] = useState(null);


  // Fetch class list on component mount
  useEffect(() => {
    const fetchClasses = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/classes`);
        const data = await response.json();
        setTargetClassList(data);
      } catch (error) {
        console.error("Failed to fetch class list:", error);
      }
    };
    fetchClasses();
  }, []);

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
      const response = await fetch(`${API_BASE_URL}/predict/`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`HTTP 错误! 状态: ${response.status}`);
      const data = await response.json();
      setPredictionResult(data.error ? `识别出错: ${data.error}` : data.predictions);
    } catch (error) {
      console.error("Prediction error:", error);
      setPredictionResult(`请求后端服务时出错: ${error.message}。`);
    }
    setIsLoading(false);
  };

  const handleFgsmAttack = async () => {
    setIsAttacking(true);
    setFgsmResult(null);
    
    const formData = new FormData();
    formData.append('epsilon', epsilon);
    if (fgsmFile) {
        formData.append('file', fgsmFile);
    }

    try {
      const response = await fetch(`${API_BASE_URL}/attack/fgsm/`, {
          method: 'POST',
          body: formData,
      });
      if (!response.ok) throw new Error(`HTTP 错误! 状态: ${response.status}`);
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

  const handleTargetedAttack = async () => {
    if (!selectedTarget) {
        alert("请先选择一个攻击目标！");
        return;
    }
    setIsTargetAttacking(true);
    setTargetedResult(null);
    setProgress(0);
    setTargetedStatus('准备中...');

    let imageId = null;
    if (targetedFile) {
        setTargetedStatus('正在上传图片...');
        const formData = new FormData();
        formData.append('file', targetedFile);
        try {
            const response = await fetch(`${API_BASE_URL}/upload`, { method: 'POST', body: formData });
            const data = await response.json();
            if (data.image_id) {
                imageId = data.image_id;
            } else {
                throw new Error(data.error || '上传失败');
            }
        } catch (error) {
            setTargetedStatus(`图片上传失败: ${error.message}`);
            setIsTargetAttacking(false);
            return;
        }
    }

    setTargetedStatus('正在连接到后端服务...');
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let wsUrl = `${wsProtocol}//${window.location.host}/api/attack/targeted_ws?target_class_id=${selectedTarget.value}`;
    if (imageId) {
        wsUrl += `&image_id=${imageId}`;
    }
    
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => setTargetedStatus('连接成功，开始执行迭代攻击 (约1-2分钟)...');
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
      if (progress < 1) setTargetedStatus('连接已断开');
    };
  };

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
          {/* Normal Prediction Tab */}
          <div className={`tab-pane ${activeTab === 'normal' ? 'active' : ''}`}>
            <h2>正常图像识别</h2>
            <p>上传一张图片，测试AI模型在正常情况下的识别能力。</p>
            <div className="controls">
                <input type="file" accept="image/*" onChange={handleFileChange} className="file-input" />
                <button onClick={handlePredict} disabled={isLoading || !selectedFile}>{isLoading ? '识别中...' : '开始识别'}</button>
            </div>
            {preview && <img src={preview} alt="Preview" className="image-preview" />}
            <pre className="results-text">{predictionResult}</pre>
          </div>

          {/* FGSM Attack Tab */}
          <div className={`tab-pane ${activeTab === 'fgsm' ? 'active' : ''}`}>
              <h2>非定向攻击 (FGSM)</h2>
              <p>选择一张图片进行攻击（若不选择则使用默认熊猫图片），并调整扰动强度 (Epsilon) 来观察效果。</p>
              <div className="controls">
                  <input type="file" accept="image/*" onChange={(e) => setFgsmFile(e.target.files[0])} className="file-input" />
              </div>
              <div className="controls">
                  <label>扰动强度 (Epsilon): {epsilon.toFixed(3)}</label>
                  <input type="range" min="0" max="0.2" step="0.005" value={epsilon} onChange={(e) => setEpsilon(parseFloat(e.target.value))} />
                  <button onClick={handleFgsmAttack} disabled={isAttacking}>{isAttacking ? '攻击中...' : '开始攻击'}</button>
              </div>
              {isAttacking && <p>正在生成对抗样本，请稍候...</p>}
              {fgsmResult && (
                  <div className="attack-container">
                      <div className="attack-column"><h3>原始图像</h3><img src={`data:image/png;base64,${fgsmResult.original_image}`} alt="Original" className="attack-image" /><pre className="results-text">{fgsmResult.original_text}</pre></div>
                      <div className="attack-column"><h3>扰动 (放大后)</h3><img src={`data:image/png;base64,${fgsmResult.perturbation_image}`} alt="Perturbation" className="attack-image" /></div>
                      <div className="attack-column"><h3>对抗样本</h3><img src={`data:image/png;base64,${fgsmResult.adversarial_image}`} alt="Adversarial" className="attack-image" /><pre className="results-text">{fgsmResult.adversarial_text}</pre></div>
                  </div>
              )}
          </div>

          {/* Targeted Attack Tab */}
          <div className={`tab-pane ${activeTab === 'targeted' ? 'active' : ''}`}>
              <h2>定向攻击</h2>
              <p>选择一张图片进行攻击（若不选择则使用默认熊猫图片），并从下方选择一个您想让AI认错的目标。</p>
              <div className="controls">
                  <input type="file" accept="image/*" onChange={(e) => setTargetedFile(e.target.files[0])} className="file-input" />
              </div>
              <div className="controls">
                  <div style={{width: '400px', color: 'black'}}>
                    <Select options={targetClassList} defaultValue={selectedTarget} onChange={setSelectedTarget} placeholder="搜索并选择一个攻击目标..." />
                  </div>
                  <button onClick={handleTargetedAttack} disabled={isTargetAttacking}>{isTargetAttacking ? '攻击进行中...' : '开始定向攻击'}</button>
              </div>
              {isTargetAttacking && (
                  <div style={{width: '80%', margin: '20px auto'}}>
                      <p>{targetedStatus}</p>
                      <progress value={progress} max="1" style={{width: '100%'}} />
                  </div>
              )}
              {targetedResult && (
                  <div className="attack-container">
                      <div className="attack-column"><h3>原始图像</h3><img src={`data:image/png;base64,${targetedResult.original_image}`} alt="Original" className="attack-image" /><pre className="results-text">{targetedResult.original_text}</pre></div>
                      <div className="attack-column"><h3>优化后的扰动</h3><img src={`data:image/png;base64,${targetedResult.perturbation_image}`} alt="Perturbation" className="attack-image" /></div>
                      <div className="attack-column"><h3>对抗样本</h3><img src={`data:image/png;base64,${targetedResult.adversarial_image}`} alt="Adversarial" className="attack-image" /><pre className="results-text">{targetedResult.adversarial_text}</pre></div>
                  </div>
              )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;