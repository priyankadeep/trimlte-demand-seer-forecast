
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import DataUpload from '@/components/DataUpload';
import ForecastDashboard from '@/components/ForecastDashboard';
import ModelConfiguration from '@/components/ModelConfiguration';
import { ProcessedData, ModelConfig } from '@/types/forecast';

const Index = () => {
  const [processedData, setProcessedData] = useState<ProcessedData | null>(null);
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    frequency: 'M',
    forecastPeriods: 12,
    autoSelect: true,
    order: [1, 1, 1],
    seasonalOrder: [1, 1, 1, 12]
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="container mx-auto p-6">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-slate-800 mb-2">
            SARIMA Demand Forecasting
          </h1>
          <p className="text-lg text-slate-600">
            Advanced time series analysis and forecasting dashboard
          </p>
        </div>

        <Tabs defaultValue="upload" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3 lg:w-96 mx-auto">
            <TabsTrigger value="upload">Data Upload</TabsTrigger>
            <TabsTrigger value="config" disabled={!processedData}>
              Configuration
            </TabsTrigger>
            <TabsTrigger value="forecast" disabled={!processedData}>
              Forecasting
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
              />
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Index;
