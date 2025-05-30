import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import DataUpload from '@/components/DataUpload';
import ForecastDashboard from '@/components/ForecastDashboard';
import ModelConfiguration from '@/components/ModelConfiguration';
import AIChatbot from '@/components/AIChatbot';
import { ProcessedData, ModelConfig, ForecastResult } from '@/types/forecast';

const Index = () => {
  const [processedData, setProcessedData] = useState<ProcessedData | null>(null);
  const [forecasts, setForecasts] = useState<Map<string, ForecastResult>>(new Map());
  
  // Updated model configuration with all three models
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    // Common settings
    frequency: 'M',
    forecastPeriods: 12,
    modelType: 'SARIMA',
    
    // SARIMA specific
    autoSelect: true,
    order: [1, 1, 1],
    seasonalOrder: [1, 1, 1, 12],
    
    // Prophet specific
    prophetConfig: {
      growth: 'linear',
      seasonalityMode: 'additive',
      changePointRange: 0.9,
      changePointPriorScale: 0.5,
      yearlySeasonality: true,
      weeklySeasonality: false,
      dailySeasonality: false,
      addQuarterly: true,
      quarterlyFourierOrder: 5,
    },
    
    // LSTM specific
    lstmConfig: {
      windowSize: 12,
      epochs: 50,
      batchSize: 16,
      lstmUnits1: 64,
      lstmUnits2: 32,
      dropoutRate: 0.2,
      bidirectional: false,
      validationSplit: 0.1,
    },
  });

  const handleForecastUpdate = (itemGroup: string, forecast: ForecastResult) => {
    setForecasts(prev => new Map(prev).set(itemGroup, forecast));
  };

  // Get model-specific badge color
  const getModelBadgeColor = (modelType: string) => {
    switch (modelType) {
      case 'SARIMA':
        return 'bg-blue-600';
      case 'Prophet':
        return 'bg-green-600';
      case 'LSTM':
        return 'bg-purple-600';
      default:
        return 'bg-gray-600';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="container mx-auto p-6">
        <div className="mb-8 text-center">
          <div className="flex items-center justify-center gap-3 mb-2">
            <h1 className="text-4xl font-bold text-slate-800">
              Multi-Model Demand Forecasting
            </h1>
            <Badge className={`${getModelBadgeColor(modelConfig.modelType)} text-white`}>
              {modelConfig.modelType}
            </Badge>
          </div>
          <p className="text-lg text-slate-600">
            Advanced time series analysis with SARIMA, Prophet, and LSTM models + AI insights
          </p>
        </div>

        <Tabs defaultValue="upload" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 lg:w-[500px] mx-auto">
            <TabsTrigger value="upload">Data Upload</TabsTrigger>
            <TabsTrigger value="config" disabled={!processedData}>
              Model Config
            </TabsTrigger>
            <TabsTrigger value="forecast" disabled={!processedData}>
              Forecasting
            </TabsTrigger>
            <TabsTrigger value="ai-chat" disabled={!processedData}>
              AI Assistant
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-6">
            <Card className="bg-white/80 backdrop-blur-sm shadow-xl border-0">
              <CardHeader>
                <CardTitle className="text-2xl text-slate-800">
                  Upload Sales Data
                </CardTitle>
              </CardHeader>
              <CardContent>
                <DataUpload onDataProcessed={setProcessedData} />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="config" className="space-y-6">
            <Card className="bg-white/80 backdrop-blur-sm shadow-xl border-0">
              <CardHeader>
                <CardTitle className="flex items-center gap-3 text-2xl text-slate-800">
                  Model Configuration
                  <Badge className={`${getModelBadgeColor(modelConfig.modelType)} text-white`}>
                    {modelConfig.modelType} Selected
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ModelConfiguration 
                  config={modelConfig}
                  onConfigChange={setModelConfig}
                />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="forecast" className="space-y-6">
            {processedData && (
              <ForecastDashboard 
                data={processedData}
                config={modelConfig}
                onForecastGenerated={handleForecastUpdate}
              />
            )}
          </TabsContent>

          <TabsContent value="ai-chat" className="space-y-6">
            <div className="grid lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <Card className="bg-white/80 backdrop-blur-sm shadow-xl border-0">
                  <CardHeader>
                    <CardTitle className="text-2xl text-slate-800">
                      AI Forecast Assistant
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="h-[600px]">
                    <AIChatbot data={processedData} forecasts={forecasts} />
                  </CardContent>
                </Card>
              </div>
              
              <div className="space-y-4">
                <Card className="bg-white/80 backdrop-blur-sm shadow-xl border-0">
                  <CardHeader>
                    <CardTitle className="text-lg text-slate-800">
                      Quick Stats
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {processedData ? (
                      <div className="space-y-3">
                        <div>
                          <div className="text-2xl font-bold text-blue-600">
                            {processedData.itemGroups.length}
                          </div>
                          <div className="text-sm text-slate-600">Item Groups</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-green-600">
                            {forecasts.size}
                          </div>
                          <div className="text-sm text-slate-600">Active Forecasts</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-purple-600">
                            {processedData.totalRecords.toLocaleString()}
                          </div>
                          <div className="text-sm text-slate-600">Total Records</div>
                        </div>
                        <div>
                          <div className={`text-2xl font-bold ${
                            modelConfig.modelType === 'SARIMA' ? 'text-blue-600' :
                            modelConfig.modelType === 'Prophet' ? 'text-green-600' : 'text-purple-600'
                          }`}>
                            {modelConfig.modelType}
                          </div>
                          <div className="text-sm text-slate-600">Selected Model</div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-slate-500 text-sm">
                        Upload data to see statistics
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Model Info Card */}
                <Card className="bg-white/80 backdrop-blur-sm shadow-xl border-0">
                  <CardHeader>
                    <CardTitle className="text-lg text-slate-800">
                      Model Info
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className={`p-3 rounded-lg ${
                        modelConfig.modelType === 'SARIMA' ? 'bg-blue-50 border border-blue-200' :
                        modelConfig.modelType === 'Prophet' ? 'bg-green-50 border border-green-200' :
                        'bg-purple-50 border border-purple-200'
                      }`}>
                        <div className="font-medium text-sm">
                          {modelConfig.modelType === 'SARIMA' && 'Statistical time series model with seasonality'}
                          {modelConfig.modelType === 'Prophet' && 'Facebook\'s robust forecasting model'}
                          {modelConfig.modelType === 'LSTM' && 'Deep learning neural network model'}
                        </div>
                        <div className="text-xs text-slate-600 mt-1">
                          {modelConfig.modelType === 'SARIMA' && 'Best for: Regular patterns, seasonal data'}
                          {modelConfig.modelType === 'Prophet' && 'Best for: Trend changes, holidays, missing data'}
                          {modelConfig.modelType === 'LSTM' && 'Best for: Complex patterns, large datasets'}
                        </div>
                      </div>
                      
                      <div className="text-xs text-slate-500">
                        Forecast Periods: {modelConfig.forecastPeriods} ({modelConfig.frequency === 'M' ? 'Monthly' : 'Weekly'})
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Index;
