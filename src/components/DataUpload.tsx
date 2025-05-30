

import React, { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent } from '@/components/ui/card';
import { Upload, FileText, CheckCircle } from 'lucide-react';
import { toast } from 'sonner';
import { ProcessedData, RawDataRow } from '@/types/forecast';
import Papa from 'papaparse';

interface DataUploadProps {
  onDataProcessed: (data: ProcessedData) => void;
}

const DataUpload: React.FC<DataUploadProps> = ({ onDataProcessed }) => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [processed, setProcessed] = useState(false);

  const parseCSV = (text: string): RawDataRow[] => {
    const { data } = Papa.parse(text, {
      header: true,
      skipEmptyLines: true,
      transformHeader: (header) => header.trim(),
      transform: (value) => value.trim()
    });
    
    return data.map(row => ({
      'Order Date': row['Order Date'] || '',
      'Item Groups': row['Item Groups'] || '',
      'Qty': parseFloat(row['Qty']) || 0,
      'Price': parseFloat(row['Price']) || 0,  // Added Price column
      'Qty - w/Sub-Total': parseFloat(row['Qty - w/Sub-Total']) || row['Qty'] || 0,  // Added correct quantity column
      'Ext. Price': parseFloat(row['Ext. Price']) || 0,
      'Tran #': row['Tran #'] || ''
    }));
  };

  const cleanItemGroupName = (name: string): string => {
    if (!name || typeof name !== 'string') return '';
    return name.trim().toLowerCase().replace(/\s+/g, ' ');
  };

  const processData = useCallback((rawData: RawDataRow[]): ProcessedData => {
    console.log('Raw data length:', rawData.length);
    
    // Clean and validate data
    const validData = rawData.filter(row => {
      const hasValidDate = row['Order Date'] && !isNaN(Date.parse(row['Order Date']));
      const hasValidItemGroup = row['Item Groups'] && row['Item Groups'].trim() !== '';
      const hasValidQty = !isNaN(row['Qty']) && row['Qty'] >= 0;
      
      return hasValidDate && hasValidItemGroup && hasValidQty;
    });
    
    console.log('Valid data length after filtering:', validData.length);
    
    // Get unique item groups (case-insensitive, trimmed)
    const uniqueItemGroups = new Set();
    validData.forEach(row => {
      const cleanName = cleanItemGroupName(row['Item Groups']);
      if (cleanName) {
        uniqueItemGroups.add(cleanName);
      }
    });
    
    console.log('Unique item groups found:', uniqueItemGroups.size);
    console.log('Item groups:', Array.from(uniqueItemGroups));
    
    // Create a mapping from clean names back to original names (use the first occurrence)
    const nameMapping = new Map();
    validData.forEach(row => {
      const cleanName = cleanItemGroupName(row['Item Groups']);
      if (cleanName && !nameMapping.has(cleanName)) {
        nameMapping.set(cleanName, row['Item Groups'].trim());
      }
    });
    
    const itemGroupsMap = new Map();
    
    validData.forEach(row => {
      const cleanName = cleanItemGroupName(row['Item Groups']);
      if (!cleanName) return;
      
      const date = new Date(row['Order Date']).toISOString().split('T')[0];
      
      if (!itemGroupsMap.has(cleanName)) {
        itemGroupsMap.set(cleanName, []);
      }
      
      // FIXED REVENUE CALCULATION
      // Use Price * (Qty - w/Sub-Total) instead of Qty * Ext. Price
      const quantity = row['Qty - w/Sub-Total'] || row['Qty'];
      const price = row['Price'] || 0;
      const revenue = price * quantity;
      
      itemGroupsMap.get(cleanName).push({
        date,
        value: row['Qty'],
        revenue: row['Ext. Price'],
        transactions: 1
      });
    });

    console.log('Item groups map size:', itemGroupsMap.size);

    const itemGroups = Array.from(itemGroupsMap.entries()).map(([cleanName, points]) => {
      const originalName = nameMapping.get(cleanName) || cleanName;
      
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
      const totalRevenue = timeSeries.reduce((sum, point) => sum + (point.revenue ?? 0), 0);
      const totalTransactions = timeSeries.reduce((sum, point) => sum + point.transactions, 0);

      console.log(`Item group "${originalName}": ${timeSeries.length} data points, total qty: ${totalQuantity}, total revenue: ${totalRevenue.toFixed(2)}`);

      return {
        name: originalName,
        timeSeries,
        stats: {
          totalQuantity,
          totalRevenue,
          avgQuantity: totalTransactions > 0 ? totalQuantity / totalTransactions : 0,
          maxQuantity: Math.max(...timeSeries.map(p => p.value)),
          minQuantity: Math.min(...timeSeries.map(p => p.value))
        }
      };
    });

    const allDates = itemGroups.flatMap(ig => ig.timeSeries.map(p => p.date));
    const sortedDates = allDates.sort();

    const result = {
      itemGroups,
      dateRange: {
        start: sortedDates[0] || '',
        end: sortedDates[sortedDates.length - 1] || ''
      },
      totalRecords: validData.length
    };

    console.log('Final processed data:', result);
    return result;
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
      
      if (processedData.itemGroups.length === 0) {
        throw new Error('No valid item groups found after processing');
      }
      
      onDataProcessed(processedData);
      setProcessed(true);
      toast.success(`Data processed successfully! Found ${processedData.itemGroups.length} unique item groups.`);
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

{/*       <div className="text-sm text-slate-600 bg-slate-100 p-4 rounded-lg">
        <h4 className="font-medium mb-2">Expected CSV Format:</h4>
        <ul className="list-disc list-inside space-y-1">
          <li>Order Date (YYYY-MM-DD or MM/DD/YY format)</li>
          <li>Item Groups (product categories)</li>
          <li>Qty (quantity values)</li>
          <li>Price (unit price)</li>
          <li>Qty - w/Sub-Total (net quantity for revenue calculation)</li>
          <li>Ext. Price (extended price/revenue - optional)</li>
          <li>Tran # (transaction numbers)</li>
        </ul>
        <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded">
          <p className="text-yellow-800 text-xs">
            <strong>Revenue Calculation:</strong> Price × (Qty - w/Sub-Total)
          </p>
        </div>
      </div> */}
    </div>
  );
};

export default DataUpload;
