
import React, { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent } from '@/components/ui/card';
import { Upload, FileText, CheckCircle } from 'lucide-react';
import { toast } from 'sonner';
import { ProcessedData, RawDataRow } from '@/types/forecast';

interface DataUploadProps {
  onDataProcessed: (data: ProcessedData) => void;
}

const DataUpload: React.FC<DataUploadProps> = ({ onDataProcessed }) => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [processed, setProcessed] = useState(false);

  const parseCSV = (text: string): RawDataRow[] => {
    const lines = text.split('\n');
    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
    
    return lines.slice(1)
      .filter(line => line.trim())
      .map(line => {
        const values = line.split(',').map(v => v.trim().replace(/"/g, ''));
        const row: any = {};
        
        headers.forEach((header, index) => {
          row[header] = values[index] || '';
        });
        
        return {
          'Order Date': row['Order Date'] || '',
          'Item Groups': row['Item Groups'] || '',
          'Qty': parseFloat(row['Qty']) || 0,
          'Ext. Price': parseFloat(row['Ext. Price']) || 0,
          'Tran #': row['Tran #'] || ''
        };
      });
  };

  const processData = useCallback((rawData: RawDataRow[]): ProcessedData => {
    const itemGroupsMap = new Map();
    
    rawData.forEach(row => {
      if (!row['Item Groups'] || !row['Order Date']) return;
      
      const itemGroup = row['Item Groups'];
      const date = new Date(row['Order Date']).toISOString().split('T')[0];
      
      if (!itemGroupsMap.has(itemGroup)) {
        itemGroupsMap.set(itemGroup, []);
      }
      
      itemGroupsMap.get(itemGroup).push({
        date,
        value: row['Qty'],
        revenue: row['Ext. Price'],
        transactions: 1
      });
    });

    const itemGroups = Array.from(itemGroupsMap.entries()).map(([name, points]) => {
      // Aggregate by date
      const dateMap = new Map();
      points.forEach((point: any) => {
        if (dateMap.has(point.date)) {
          const existing = dateMap.get(point.date);
          existing.value += point.value;
          existing.revenue += point.revenue;
          existing.transactions += point.transactions;
        } else {
          dateMap.set(point.date, { ...point });
        }
      });

      const timeSeries = Array.from(dateMap.values()).sort((a, b) => 
        new Date(a.date).getTime() - new Date(b.date).getTime()
      );

      const totalQuantity = timeSeries.reduce((sum, point) => sum + point.value, 0);
      const totalRevenue = timeSeries.reduce((sum, point) => sum + (point.revenue || 0), 0);

      return {
        name,
        timeSeries,
        stats: {
          totalQuantity,
          totalRevenue,
          avgQuantity: totalQuantity / timeSeries.length,
          maxQuantity: Math.max(...timeSeries.map(p => p.value)),
          minQuantity: Math.min(...timeSeries.map(p => p.value))
        }
      };
    });

    const allDates = itemGroups.flatMap(ig => ig.timeSeries.map(p => p.date));
    const sortedDates = allDates.sort();

    return {
      itemGroups,
      dateRange: {
        start: sortedDates[0],
        end: sortedDates[sortedDates.length - 1]
      },
      totalRecords: rawData.length
    };
  }, []);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile && selectedFile.type === 'text/csv') {
      setFile(selectedFile);
      setProcessed(false);
    } else {
      toast.error('Please select a valid CSV file');
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    try {
      const text = await file.text();
      const rawData = parseCSV(text);
      
      if (rawData.length === 0) {
        throw new Error('No valid data found in CSV');
      }

      const processedData = processData(rawData);
      onDataProcessed(processedData);
      setProcessed(true);
      toast.success('Data processed successfully!');
    } catch (error) {
      console.error('Error processing file:', error);
      toast.error('Error processing file. Please check the format.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid gap-4">
        <div className="space-y-2">
          <Label htmlFor="csv-file">Select CSV File</Label>
          <Input
            id="csv-file"
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="cursor-pointer"
          />
        </div>

        {file && (
          <Card className="border-dashed border-2 border-blue-200 bg-blue-50">
            <CardContent className="p-4">
              <div className="flex items-center space-x-3">
                <FileText className="h-8 w-8 text-blue-600" />
                <div>
                  <p className="font-medium text-blue-900">{file.name}</p>
                  <p className="text-sm text-blue-600">
                    Size: {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        <Button 
          onClick={handleUpload} 
          disabled={!file || loading}
          className="bg-blue-600 hover:bg-blue-700"
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
              Processing...
            </>
          ) : processed ? (
            <>
              <CheckCircle className="mr-2 h-4 w-4" />
              Data Processed
            </>
          ) : (
            <>
              <Upload className="mr-2 h-4 w-4" />
              Process Data
            </>
          )}
        </Button>
      </div>

      <div className="text-sm text-slate-600 bg-slate-100 p-4 rounded-lg">
        <h4 className="font-medium mb-2">Expected CSV Format:</h4>
        <ul className="list-disc list-inside space-y-1">
          <li>Order Date (YYYY-MM-DD or MM/DD/YY format)</li>
          <li>Item Groups (product categories)</li>
          <li>Qty (quantity values)</li>
          <li>Ext. Price (extended price/revenue)</li>
          <li>Tran # (transaction numbers)</li>
        </ul>
      </div>
    </div>
  );
};

export default DataUpload;
