
import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Bot, MessageCircle, Send, User } from 'lucide-react';
import { ProcessedData, ForecastResult } from '@/types/forecast';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
}

interface AIChatbotProps {
  data: ProcessedData | null;
  forecasts: Map<string, ForecastResult>;
}

const AIChatbot: React.FC<AIChatbotProps> = ({ data, forecasts }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [showApiKeyInput, setShowApiKeyInput] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const generateDataSummary = () => {
    if (!data) return "No data has been uploaded yet.";

    const totalItemGroups = data.itemGroups.length;
    const totalRecords = data.totalRecords;
    const dateRange = `${data.dateRange.start} to ${data.dateRange.end}`;
    
    const topItemGroups = data.itemGroups
      .sort((a, b) => b.stats.totalQuantity - a.stats.totalQuantity)
      .slice(0, 5)
      .map(ig => `${ig.name} (${ig.stats.totalQuantity} total qty)`)
      .join(', ');

    const forecastSummary = Array.from(forecasts.entries())
      .map(([itemGroup, forecast]) => {
        const avgForecast = forecast.forecast.reduce((sum, f) => sum + f.mean, 0) / forecast.forecast.length;
        return `${itemGroup}: avg ${avgForecast.toFixed(1)} per period`;
      })
      .join(', ');

    return `Dataset: ${totalItemGroups} item groups, ${totalRecords} records spanning ${dateRange}. Top performers: ${topItemGroups}. ${forecasts.size > 0 ? `Active forecasts: ${forecastSummary}` : 'No forecasts generated yet.'}`;
  };

  const generateAIResponse = async (userMessage: string): Promise<string> => {
    const storedApiKey = localStorage.getItem('groq_api_key') || apiKey;
    
    if (!storedApiKey) {
      return "Please provide your Groq API key to enable AI responses. I can help you analyze your SARIMA forecast data, explain trends, compare item groups, and answer questions about your demand forecasting results.";
    }

    const dataSummary = generateDataSummary();
    
    const systemPrompt = `You are an AI assistant specialized in SARIMA demand forecasting and time series analysis. You have access to the following data:

${dataSummary}

Help the user understand their data, forecasts, and provide insights about demand patterns. Be concise and practical in your responses. Focus on actionable insights for demand planning and inventory management.`;

    try {
      const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${storedApiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'llama-3.1-70b-versatile',
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userMessage }
          ],
          max_tokens: 500,
          temperature: 0.7,
        }),
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const result = await response.json();
      return result.choices[0]?.message?.content || "I couldn't generate a response. Please try again.";
    } catch (error) {
      console.error('Error calling Groq API:', error);
      return "I'm having trouble connecting to the AI service. Please check your API key and try again.";
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const aiResponse = await generateAIResponse(inputMessage);
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: aiResponse,
        sender: 'ai',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error generating AI response:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const saveApiKey = () => {
    if (apiKey.trim()) {
      localStorage.setItem('groq_api_key', apiKey.trim());
      setShowApiKeyInput(false);
      setApiKey('');
    }
  };

  const suggestedQuestions = [
    "What are the top performing item groups?",
    "Which products show seasonal patterns?",
    "How accurate are my current forecasts?",
    "What trends do you see in the data?",
    "Which items should I stock more of?",
  ];

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <Bot className="h-5 w-5 text-blue-600" />
          AI Forecast Assistant
          <Badge variant="secondary" className="ml-auto">
            {data ? `${data.itemGroups.length} groups` : 'No data'}
          </Badge>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="flex-1 flex flex-col gap-4">
        {!localStorage.getItem('groq_api_key') && !showApiKeyInput && (
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <p className="text-sm text-blue-700 mb-2">
              Connect your Groq API key to enable AI insights
            </p>
            <Button 
              onClick={() => setShowApiKeyInput(true)}
              size="sm"
              variant="outline"
            >
              Add Groq API Key
            </Button>
          </div>
        )}

        {showApiKeyInput && (
          <div className="space-y-2 p-4 bg-slate-50 rounded-lg">
            <Input
              type="password"
              placeholder="Enter your Groq API key"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
            <div className="flex gap-2">
              <Button onClick={saveApiKey} size="sm">Save</Button>
              <Button 
                onClick={() => setShowApiKeyInput(false)} 
                size="sm" 
                variant="outline"
              >
                Cancel
              </Button>
            </div>
          </div>
        )}

        <ScrollArea className="flex-1 h-64">
          <div className="space-y-4 pr-4">
            {messages.length === 0 && (
              <div className="text-center py-8 text-slate-500">
                <MessageCircle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">Ask me anything about your forecast data!</p>
                
                {data && (
                  <div className="mt-4 space-y-2">
                    <p className="text-xs font-medium">Try asking:</p>
                    {suggestedQuestions.slice(0, 3).map((question, index) => (
                      <button
                        key={index}
                        onClick={() => setInputMessage(question)}
                        className="block text-xs text-blue-600 hover:text-blue-800 mx-auto"
                      >
                        "{question}"
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex gap-2 max-w-[80%] ${message.sender === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                    message.sender === 'user' ? 'bg-blue-600' : 'bg-slate-600'
                  }`}>
                    {message.sender === 'user' ? (
                      <User className="h-4 w-4 text-white" />
                    ) : (
                      <Bot className="h-4 w-4 text-white" />
                    )}
                  </div>
                  <div className={`p-3 rounded-lg ${
                    message.sender === 'user' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-slate-100 text-slate-900'
                  }`}>
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                    <p className="text-xs opacity-70 mt-1">
                      {message.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex gap-3 justify-start">
                <div className="w-8 h-8 rounded-full bg-slate-600 flex items-center justify-center flex-shrink-0">
                  <Bot className="h-4 w-4 text-white" />
                </div>
                <div className="bg-slate-100 p-3 rounded-lg">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        <div className="flex gap-2">
          <Textarea
            placeholder="Ask about your forecast data..."
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            className="min-h-[40px] max-h-[120px] resize-none"
            disabled={isLoading}
          />
          <Button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            size="sm"
            className="self-end"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default AIChatbot;
