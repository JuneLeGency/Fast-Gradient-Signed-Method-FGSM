import React, { useState, useEffect } from 'react';
import {
  Layout,
  Tabs,
  Button,
  Upload,
  Slider,
  Select,
  Progress,
  Card,
  Row,
  Col,
  Typography,
  Spin,
  Alert,
  Space,
  notification,
  Divider
} from 'antd';
import { UploadOutlined, ExperimentOutlined, AimOutlined, BugOutlined, DownloadOutlined } from '@ant-design/icons';
import './App.css';

const { Header, Content } = Layout;
const { TabPane } = Tabs;
const { Title, Paragraph, Text } = Typography;
const { Option } = Select;

const API_BASE_URL = "/api";

function App() {
  // State for Normal Prediction Tab
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [predictionResult, setPredictionResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // State for FGSM Attack Tab
  const [epsilon, setEpsilon] = useState(0.05);
  const [fgsmResult, setFgsmResult] = useState(null);
  const [isAttacking, setIsAttacking] = useState(false);
  const [fgsmFile, setFgsmFile] = useState(null);

  // State for Targeted Attack Tab
  const [targetClassList, setTargetClassList] = useState([]);
  const [selectedTarget, setSelectedTarget] = useState(504);
  const [progress, setProgress] = useState(0);
  const [targetedResult, setTargetedResult] = useState(null);
  const [isTargetAttacking, setIsTargetAttacking] = useState(false);
  const [targetedStatus, setTargetedStatus] = useState('');
  const [targetedFile, setTargetedFile] = useState(null);

  useEffect(() => {
    const fetchClasses = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/classes`);
        const data = await response.json();
        setTargetClassList(data);
      } catch (error) {
        notification.error({
          message: '获取类别列表失败',
          description: '无法从后端加载目标类别列表，请检查服务是否正常运行。',
        });
      }
    };
    fetchClasses();
  }, []);

  const handleFileChange = (file) => {
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPredictionResult('');
    }
    return false; // Prevent auto-upload
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      notification.warning({ message: '未选择文件', description: '请先选择一个图片文件再进行识别。' });
      return;
    }
    setIsLoading(true);
    setPredictionResult('');
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const response = await fetch(`${API_BASE_URL}/predict/`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`HTTP 错误! 状态: ${response.status}`);
      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      setPredictionResult(data.predictions);
    } catch (error) {
      notification.error({ message: '识别失败', description: error.message });
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
      const response = await fetch(`${API_BASE_URL}/attack/fgsm/`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`HTTP 错误! 状态: ${response.status}`);
      const data = await response.json();
      if (data.error) throw new Error(data.error);
      setFgsmResult(data);
      notification.success({ message: '非定向攻击成功！' });
    } catch (error) {
      notification.error({ message: '攻击失败', description: error.message });
    }
    setIsAttacking(false);
  };

  const handleTargetedAttack = async () => {
    if (!selectedTarget) {
      notification.warning({ message: '未选择目标', description: '请先选择一个攻击目标！' });
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
        notification.error({ message: '图片上传失败', description: error.message });
        setIsTargetAttacking(false);
        return;
      }
    }
    setTargetedStatus('正在连接到后端服务...');
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let wsUrl = `${wsProtocol}//${window.location.host}/api/attack/targeted_ws?target_class_id=${selectedTarget}`;
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
        notification.success({ message: '定向攻击成功！' });
      } else if (message.type === 'error') {
        setTargetedStatus(`发生错误: ${message.message}`);
        notification.error({ message: '攻击出错', description: message.message });
        setIsTargetAttacking(false);
      }
    };
    ws.onerror = () => {
      notification.error({ message: 'WebSocket 连接出错', description: '请检查后端服务是否正在运行。' });
      setIsTargetAttacking(false);
    };
    ws.onclose = () => {
      setIsTargetAttacking(false);
      if (progress < 1) setTargetedStatus('连接已断开');
    };
  };
  
  const handleDownload = (base64Image, fileName) => {
      const link = document.createElement('a');
      link.href = `data:image/png;base64,${base64Image}`;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
  };

  const renderAttackResults = (result, isTargeted = false) => (
    <Row gutter={24} className="results-grid">
      <Col span={8}>
        <Card title="原始图像" className="result-card">
          <img src={`data:image/png;base64,${result.original_image}`} alt="Original" className="attack-image" />
          <Typography>
            <pre className="results-text">{result.original_text}</pre>
          </Typography>
        </Card>
      </Col>
      <Col span={8}>
        <Card title="扰动" className="result-card">
          <img src={`data:image/png;base64,${result.perturbation_image}`} alt="Perturbation" className="attack-image" />
          <Typography>
             <Paragraph>这是施加在原始图像上的微小扰动（为方便观察已放大）。</Paragraph>
          </Typography>
        </Card>
      </Col>
      <Col span={8}>
        <Card title="对抗样本" className="result-card">
          <img src={`data:image/png;base64,${result.adversarial_image}`} alt="Adversarial" className="attack-image" />
          <Typography>
            <pre className="results-text">{result.adversarial_text}</pre>
          </Typography>
          {isTargeted && (
            <Button 
              type="primary" 
              icon={<DownloadOutlined />} 
              onClick={() => handleDownload(result.adversarial_image, 'targeted-adversarial-sample.png')}
              style={{ marginTop: '16px' }}
            >
              下载对抗样本
            </Button>
          )}
        </Card>
      </Col>
    </Row>
  );

  return (
    <Layout className="App">
      <Header style={{ backgroundColor: 'white', textAlign: 'center', borderBottom: '1px solid #f0f0f0' }}>
        <Title level={2} style={{ margin: '14px 0' }}>
          <ExperimentOutlined /> AI 对抗攻击演示
        </Title>
      </Header>
      <Content style={{ padding: '24px 0' }}>
        <Tabs defaultActiveKey="1" centered type="card">
          <TabPane tab={<span><AimOutlined />正常识别</span>} key="1">
            <Card>
              <Title level={4}>正常图像识别</Title>
              <Paragraph>此功能用于评估AI模型在未经任何攻击的原始图片上的表现。您可以上传一张图片，系统将使用预训练的ResNet-50模型对其进行分类，并显示置信度最高的前5个预测结果。</Paragraph>
              <Divider />
              <Space direction="vertical" style={{ width: '100%' }}>
                <Title level={5}>1. 选择图片</Title>
                <Upload beforeUpload={handleFileChange} showUploadList={false}>
                  <Button icon={<UploadOutlined />}>点击选择文件</Button>
                </Upload>
                {selectedFile && <Text type="secondary">已选择: {selectedFile.name}</Text>}
                {preview && <img src={preview} alt="Preview" style={{ maxWidth: 300, marginTop: 20, border: '1px solid #f0f0f0' }} />}
                <Divider />
                <Title level={5}>2. 开始识别</Title>
                <Button type="primary" onClick={handlePredict} loading={isLoading} disabled={!selectedFile}>
                  {isLoading ? '识别中...' : '开始识别'}
                </Button>
                {isLoading && <Spin tip="正在识别..." />}
                {predictionResult && (
                  <Alert message={<pre>{predictionResult}</pre>} type="info" style={{ marginTop: 20 }} />
                )}
              </Space>
            </Card>
          </TabPane>
          <TabPane tab={<span><BugOutlined />非定向攻击 (FGSM)</span>} key="2">
            <Card>
              <Title level={4}>非定向攻击 (Fast Gradient Sign Method)</Title>
              <Paragraph>非定向攻击旨在让AI模型对图片产生错误分类，但不指定具体的错误类别。我们使用经典的FGSM算法，通过计算损失函数关于输入图像的梯度来生成微小扰动。您可以调整扰动强度(Epsilon)来观察其对攻击效果的影响。</Paragraph>
              <Divider />
              <Space direction="vertical" style={{ width: '100%' }}>
                <Title level={5}>1. 选择攻击图片 (可选)</Title>
                <Paragraph type="secondary">若不选择，将使用默认的熊猫图片进行攻击。</Paragraph>
                 <Upload beforeUpload={(file) => { setFgsmFile(file); return false; }} showUploadList={false}>
                  <Button icon={<UploadOutlined />}>点击选择文件</Button>
                </Upload>
                {fgsmFile && <Text type="secondary">已选择: {fgsmFile.name}</Text>}
                <Divider />
                <Title level={5}>2. 设置扰动强度</Title>
                <Row align="middle" style={{width: '100%'}}>
                  <Col span={4}><Text>扰动强度 (Epsilon):</Text></Col>
                  <Col span={12}><Slider min={0} max={0.2} step={0.005} value={epsilon} onChange={setEpsilon} /></Col>
                  <Col span={4}><Text>{epsilon.toFixed(3)}</Text></Col>
                </Row>
                <Divider />
                <Title level={5}>3. 开始攻击</Title>
                <Button type="primary" onClick={handleFgsmAttack} loading={isAttacking}>
                  {isAttacking ? '攻击中...' : '开始攻击'}
                </Button>
              </Space>
              {isAttacking && <Spin tip="正在生成对抗样本..." style={{ display: 'block', marginTop: 24 }} />}
              {fgsmResult && renderAttackResults(fgsmResult, false)}
            </Card>
          </TabPane>
          <TabPane tab={<span><BugOutlined />定向攻击</span>} key="3">
            <Card>
              <Title level={4}>定向攻击 (Iterative FGSM)</Title>
              <Paragraph>定向攻击的目标是诱导AI模型将图片错误地分类为我们指定的任意类别。此演示采用迭代式FGSM方法，通过多次迭代小幅度的扰动，逐步将模型的预测结果引向目标类别。这是一个更精细、但计算成本更高的攻击方式。</Paragraph>
              <Divider />
               <Space direction="vertical" style={{ width: '100%' }}>
                <Title level={5}>1. 选择攻击图片 (可选)</Title>
                <Paragraph type="secondary">若不选择，将使用默认的熊猫图片进行攻击。</Paragraph>
                 <Upload beforeUpload={(file) => { setTargetedFile(file); return false; }} showUploadList={false}>
                  <Button icon={<UploadOutlined />}>点击选择文件</Button>
                </Upload>
                {targetedFile && <Text type="secondary">已选择: {targetedFile.name}</Text>}
                <Divider />
                <Title level={5}>2. 选择攻击目标</Title>
                <Select
                  showSearch
                  style={{ width: 400 }}
                  placeholder="搜索并选择一个攻击目标..."
                  defaultValue={selectedTarget}
                  onChange={setSelectedTarget}
                  optionFilterProp="children"
                  filterOption={(input, option) =>
                    option.children.toLowerCase().indexOf(input.toLowerCase()) >= 0
                  }
                >
                  {targetClassList.map(c => <Option key={c.value} value={c.value}>{c.label}</Option>)}
                </Select>
                <Divider />
                <Title level={5}>3. 开始攻击</Title>
                <Button type="primary" onClick={handleTargetedAttack} loading={isTargetAttacking}>
                  {isTargetAttacking ? '攻击进行中...' : '开始定向攻击'}
                </Button>
              </Space>
              {isTargetAttacking && (
                <div style={{ marginTop: 24 }}>
                  <Paragraph>{targetedStatus}</Paragraph>
                  <Progress percent={Math.round(progress * 100)} />
                </div>
              )}
              {targetedResult && renderAttackResults(targetedResult, true)}
            </Card>
          </TabPane>
        </Tabs>
      </Content>
    </Layout>
  );
}

export default App;
