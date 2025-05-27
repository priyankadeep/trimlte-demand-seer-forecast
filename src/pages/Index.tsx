
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import DataUpload from '@/components/DataUpload';
import ForecastDashboard from '@/components/ForecastDashboard';
import ModelConfiguration from '@/components/ModelConfiguration';
import AIChatbot from '@/components/AIChatbot';
import { ProcessedData, ModelConfig, ForecastResult } from '@/types/forecast';

const Index = () => {
  const [processedData, setProcessedData] = useState<ProcessedData | null>(null);
  const [forecasts, setForecasts] = useState<Map<string, ForecastResult>>(new Map());
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    frequency: 'M',
    forecastPeriods: 12,
    autoSelect: true,
    order: [1, 1, 1],
    seasonalOrder: [1, 1, 1, 12]
  });

  const handleForecastUpdate = (itemGroup: string, forecast: ForecastResult) => {
    setForecasts(prev => new Map(prev).set(itemGroup, forecast));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="container mx-auto p-6">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-slate-800 mb-2">
            SARIMA Demand Forecasting
          </h1>
          <p className="text-lg text-slate-600">
            Advanced time series analysis and forecasting dashboard with AI insights
          </p>
        </div>

        <Tabs defaultValue="upload" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 lg:w-[500px] mx-auto">
            <TabsTrigger value="upload">Data Upload</TabsTrigger>
            <TabsTrigger value="config" disabled={!processedData}>
              Configuration
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
                <CardTitle className="text-2xl text-slate-800">
                  Model Configuration
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
                      </div>
                    ) : (
                      <div className="text-slate-500 text-sm">
                        Upload data to see statistics
                      </div>
                    )}
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
