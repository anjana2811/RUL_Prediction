import React, { useState, useEffect } from "react";
import {
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  CartesianGrid,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  AreaChart,
  Area,
  Cell
} from "recharts";
import { 
  Button, 
  Typography, 
  Box, 
  Grid, 
  CircularProgress, 
  Alert, 
  Paper, 
  Divider,
  Chip,
  Fade,
  Container
} from "@mui/material";
import { 
  CloudUpload as CloudUploadIcon, 
  Assessment as AssessmentIcon,
  BarChart as BarChartIcon,
  PieChart as PieChartIcon,
  ShowChart as ShowChartIcon,
  Info as InfoIcon,
  Refresh as RefreshIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon
} from '@mui/icons-material';

// Custom theme colors
const theme = {
  primary: "#3f51b5",
  secondary: "#f50057",
  success: "#4caf50",
  warning: "#ff9800",
  background: "#f5f7fa",
  cardBg: "#ffffff",
  chartColors: ["#3f51b5", "#f50057", "#4caf50", "#ff9800", "#2196f3", "#9c27b0", "#00bcd4", "#ffeb3b"]
};

// RUL status thresholds
const getRulStatus = (rul) => {
  if (rul > 150) return { color: theme.success, label: "Healthy", icon: <CheckCircleIcon /> };
  if (rul > 80) return { color: theme.chartColors[2], label: "Good", icon: <CheckCircleIcon /> };
  if (rul > 40) return { color: theme.warning, label: "Monitor", icon: <WarningIcon /> };
  return { color: theme.secondary, label: "Critical", icon: <WarningIcon /> };
};

const Dashboard = () => {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fadeIn, setFadeIn] = useState(false);
  const [showResults, setShowResults] = useState(false);

  // Fade in animation after data loads
  useEffect(() => {
    if (prediction) {
      setTimeout(() => {
        setFadeIn(true);
      }, 100);
    } else {
      setFadeIn(false);
    }
  }, [prediction]);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      const fileName = selectedFile.name.toLowerCase();
      if (!fileName.endsWith('.csv') && !fileName.endsWith('.txt')) {
        setError("Please select a CSV or TXT file");
        setFile(null);
      } else {
        setFile(selectedFile);
        setError(null);
      }
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file first");
      return;
    }
    
    setLoading(true);
    setError(null);
    setFadeIn(false);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        // Transform backend response to match frontend expectations
        const transformedData = {
          predicted_rul: data.predicted_rul,
          engine_id: data.engine_id,
          graphData: data.graphData.map(item => ({
            sensor: item.sensor,
            value: item.value,
            sensorNumber: parseInt(item.sensor.replace('sensor_', ''))
          })),
          rulTrend: data.rulTrend.map(item => ({
            cycle: item.cycle,
            rul: item.rul
          }))
        };
        setPrediction(transformedData);
        setShowResults(true);
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      setError("Failed to process the file. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPrediction(null);
    setError(null);
    setFadeIn(false);
    setShowResults(false);
  };

  // Extract top contributing sensors
  const getTopSensors = () => {
    if (!prediction || !prediction.graphData) return [];
    
    return [...prediction.graphData]
      .sort((a, b) => b.value - a.value)
      .slice(0, 5)
      .map(item => ({
        ...item,
        sensorNumber: parseInt(item.sensor.replace('sensor_', ''))
      }));
  };
  
  // Calculate health percentage based on RUL
  const getHealthPercentage = (rul) => {
    const percentage = Math.min(rul / 200 * 100, 100);
    return Math.round(percentage);
  };

  const rulStatus = prediction ? getRulStatus(prediction.predicted_rul) : null;
  const healthPercentage = prediction ? getHealthPercentage(prediction.predicted_rul) : 0;

  return (
    <div style={{ 
      background: `linear-gradient(to bottom, ${theme.background}, #e8eaf6)`, 
      minHeight: "100vh",
      padding: "20px"
    }}>
      <Container maxWidth="xl">
        <Paper 
          elevation={3} 
          style={{ 
            padding: "20px", 
            background: `linear-gradient(135deg, #303f9f 0%, #3f51b5 100%)`,
            color: "#fff",
            borderRadius: "10px",
            marginBottom: "20px"
          }}
        >
          <Grid container alignItems="center" spacing={2}>
            <Grid item>
              <AssessmentIcon style={{ fontSize: 40 }} />
            </Grid>
            <Grid item>
              <Typography variant="h4" component="h1">
                {showResults ? "Prediction Results" : "Aircraft Engine RUL Prediction"}
              </Typography>
              <Typography variant="subtitle1">
                {showResults ? "Engine Health Analysis" : "Predict Remaining Useful Life with Advanced Analytics"}
              </Typography>
            </Grid>
            {showResults && (
              <Grid item>
                <Button
                  variant="contained"
                  color="secondary"
                  onClick={handleReset}
                >
                  Back to Upload
                </Button>
              </Grid>
            )}
          </Grid>
        </Paper>

        {!showResults ? (
          <Grid container spacing={3} justifyContent="center">
            {/* Upload Section */}
            <Grid item xs={12} md={4}>
              <Paper elevation={3} style={{ borderRadius: "10px", height: "100%" }}>
                <Box p={3} textAlign="center">
                  <Typography variant="h6" gutterBottom>
                    <CloudUploadIcon style={{ verticalAlign: "middle", marginRight: "8px" }} />
                    Data Upload
                  </Typography>
                  <Divider style={{ margin: "16px 0" }} />
                  
                  <Box 
                    border={2} 
                    borderColor={theme.primary} 
                    borderRadius="borderRadius" 
                    p={3} 
                    mb={2}
                    style={{ 
                      borderStyle: 'dashed', 
                      backgroundColor: 'rgba(63, 81, 181, 0.04)',
                      cursor: 'pointer',
                      transition: 'all 0.3s'
                    }}
                    onClick={() => document.getElementById('file-upload').click()}
                  >
                    <input
                      id="file-upload"
                      type="file"
                      onChange={handleFileChange}
                      style={{ display: 'none' }}
                      accept=".csv,.txt"
                    />
                    <CloudUploadIcon style={{ fontSize: 48, color: theme.primary, marginBottom: "8px" }} />
                    <Typography variant="body1">
                      Drag & drop or click to select a file
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      Supported formats: CSV, TXT
                    </Typography>
                  </Box>
                  
                  {file && !error && (
                    <Fade in={!!file}>
                      <Alert 
                        severity="success" 
                        icon={<CheckCircleIcon fontSize="inherit" />}
                        style={{ marginBottom: "16px" }}
                      >
                        File selected: {file.name}
                      </Alert>
                    </Fade>
                  )}
                  
                  {error && (
                    <Alert 
                      severity="error" 
                      style={{ marginBottom: "16px" }}
                    >
                      {error}
                    </Alert>
                  )}
                  
                  <Box display="flex" justifyContent="space-between" mb={2}>
                    <Button
                      variant="contained"
                      color="primary"
                      style={{ width: "48%" }}
                      startIcon={<RefreshIcon />}
                      onClick={handleReset}
                    >
                      Reset
                    </Button>
                    <Button
                      variant="contained"
                      style={{ 
                        width: "48%", 
                        backgroundColor: theme.success,
                        color: "white" 
                      }}
                      disabled={loading || !file}
                      onClick={handleUpload}
                      startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <AssessmentIcon />}
                    >
                      {loading ? "Processing..." : "Analyze"}
                    </Button>
                  </Box>
                </Box>
              </Paper>
            </Grid>

            {/* Placeholder for Main Content Area when no results */}
            <Grid item xs={12} md={8}>
              <Paper 
                elevation={3} 
                style={{ 
                  borderRadius: "10px", 
                  height: "100%", 
                  display: "flex", 
                  alignItems: "center", 
                  justifyContent: "center",
                  padding: "40px",
                  flexDirection: "column",
                  background: `linear-gradient(135deg, ${theme.cardBg} 0%, #f5f5f5 100%)`
                }}
              >
                <InfoIcon style={{ fontSize: 60, color: theme.primary, marginBottom: "16px" }} />
                <Typography variant="h6" align="center" gutterBottom>
                  Upload your engine sensor data
                </Typography>
                <Typography variant="body1" align="center" color="textSecondary" paragraph>
                  Get accurate Remaining Useful Life predictions with advanced deep learning models.
                  Our system analyzes sensor data to predict when maintenance will be required.
                </Typography>
                <Box mt={2}>
                  <Grid container spacing={2} justifyContent="center">
                    <Grid item>
                      <Chip icon={<CheckCircleIcon />} label="Accurate Predictions" />
                    </Grid>
                    <Grid item>
                      <Chip icon={<CheckCircleIcon />} label="Easy to Use" />
                    </Grid>
                    <Grid item>
                      <Chip icon={<CheckCircleIcon />} label="Advanced Analytics" />
                    </Grid>
                  </Grid>
                </Box>
              </Paper>
            </Grid>
          </Grid>
        ) : (
          <Fade in={fadeIn}>
            <div>
              {/* RUL Summary */}
              <Paper 
                elevation={3} 
                style={{ 
                  borderRadius: "10px", 
                  marginBottom: "16px",
                  overflow: "hidden"
                }}
              >
                <Box 
                  p={2} 
                  style={{ 
                    background: `linear-gradient(45deg, ${rulStatus.color} 0%, ${theme.primary} 100%)`,
                    color: "#fff"
                  }}
                >
                  <Typography variant="h6">
                    Engine Health Summary
                  </Typography>
                </Box>
                
                <Box p={3}>
                  <Grid container spacing={3} alignItems="center">
                    <Grid item xs={12} md={4}>
                      <Box textAlign="center">
                        <Typography variant="h6" gutterBottom>RUL Status</Typography>
                        <Box display="flex" alignItems="center" justifyContent="center">
                          {rulStatus.icon}
                          <Typography 
                            variant="h4" 
                            component="span" 
                            style={{ marginLeft: "8px", color: rulStatus.color }}
                          >
                            {rulStatus.label}
                          </Typography>
                        </Box>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={12} md={4}>
                      <Box textAlign="center" position="relative">
                        <Typography variant="h6" gutterBottom>Health</Typography>
                        <Box position="relative" display="inline-flex">
                          <CircularProgress 
                            variant="determinate" 
                            value={healthPercentage} 
                            size={100}
                            thickness={5}
                            style={{ color: rulStatus.color }}
                          />
                          <Box
                            top={0}
                            left={0}
                            bottom={0}
                            right={0}
                            position="absolute"
                            display="flex"
                            alignItems="center"
                            justifyContent="center"
                          >
                            <Typography variant="h5" component="div" color="textPrimary">
                              {`${healthPercentage}%`}
                            </Typography>
                          </Box>
                        </Box>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={12} md={4}>
                      <Box textAlign="center">
                        <Typography variant="h6" gutterBottom>Remaining Cycles</Typography>
                        <Typography variant="h3" color="primary">
                          {Math.round(prediction.predicted_rul)}
                        </Typography>
                        <Chip 
                          label={`Engine #${prediction.engine_id}`} 
                          size="small" 
                          style={{ marginTop: "8px" }}
                        />
                      </Box>
                    </Grid>
                  </Grid>
                </Box>
              </Paper>
              
              {/* Charts Section */}
              <Grid container spacing={2}>
                {/* RUL Trend Chart */}
                <Grid item xs={12}>
                  <Paper elevation={3} style={{ borderRadius: "10px" }}>
                    <Box 
                      p={2} 
                      style={{ 
                        background: theme.primary,
                        color: "#fff",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center"
                      }}
                    >
                      <Box display="flex" alignItems="center">
                        <ShowChartIcon style={{ marginRight: "8px" }} />
                        <Typography variant="h6">RUL Prediction Trend</Typography>
                      </Box>
                    </Box>
                    
                    <Box p={2}>
                      <ResponsiveContainer width="100%" height={280}>
                        <AreaChart data={prediction.rulTrend || []}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="cycle" />
                          <YAxis label={{ value: 'Remaining Useful Life', angle: -90, position: 'insideLeft' }} />
                          <Tooltip 
                            formatter={(value) => [`${value.toFixed(1)} cycles`, 'RUL']}
                            labelFormatter={(label) => `Cycle: ${label}`}
                          />
                          <Area 
                            type="monotone" 
                            dataKey="rul" 
                            stroke={theme.primary} 
                            fill={`${theme.primary}40`} 
                            strokeWidth={2} 
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </Box>
                  </Paper>
                </Grid>
                
                {/* Sensor Charts */}
                <Grid item xs={12} md={6}>
                  <Paper elevation={3} style={{ borderRadius: "10px", height: "100%" }}>
                    <Box 
                      p={2} 
                      style={{ 
                        background: theme.primary,
                        color: "#fff",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center"
                      }}
                    >
                      <Box display="flex" alignItems="center">
                        <BarChartIcon style={{ marginRight: "8px" }} />
                        <Typography variant="h6">Sensor Contribution</Typography>
                      </Box>
                    </Box>
                    
                    <Box p={2}>
                      <ResponsiveContainer width="100%" height={280}>
                        <BarChart 
                          data={getTopSensors()} 
                          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="sensorNumber" label={{ value: 'Sensor Number', position: 'insideBottom', offset: -5 }} />
                          <YAxis label={{ value: 'Value', angle: -90, position: 'insideLeft' }} />
                          <Tooltip 
                            formatter={(value) => [`${value.toFixed(2)}`, 'Average Value']}
                            labelFormatter={(label) => `Sensor ${label}`}
                          />
                          <Bar 
                            dataKey="value" 
                            fill={theme.primary}
                            animationDuration={1500}
                          >
                            {getTopSensors().map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={theme.chartColors[index % theme.chartColors.length]} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </Box>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Paper elevation={3} style={{ borderRadius: "10px", height: "100%" }}>
                    <Box 
                      p={2} 
                      style={{ 
                        background: theme.primary,
                        color: "#fff",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center"
                      }}
                    >
                      <Box display="flex" alignItems="center">
                        <PieChartIcon style={{ marginRight: "8px" }} />
                        <Typography variant="h6">Sensor Distribution</Typography>
                      </Box>
                    </Box>
                    
                    <Box p={2}>
                      <ResponsiveContainer width="100%" height={280}>
                        <RadarChart outerRadius={90} data={getTopSensors()}>
                          <PolarGrid />
                          <PolarAngleAxis dataKey="sensorNumber" />
                          <PolarRadiusAxis angle={30} domain={[0, 'auto']} />
                          <Radar 
                            name="Sensor Values" 
                            dataKey="value" 
                            stroke={theme.primary} 
                            fill={`${theme.primary}80`} 
                            fillOpacity={0.6} 
                          />
                          <Tooltip 
                            formatter={(value) => [`${value.toFixed(2)}`, 'Value']}
                            labelFormatter={(label) => `Sensor ${label}`}
                          />
                        </RadarChart>
                      </ResponsiveContainer>
                    </Box>
                  </Paper>
                </Grid>
              </Grid>
            </div>
          </Fade>
        )}
      </Container>
    </div>
  );
};

export default Dashboard;