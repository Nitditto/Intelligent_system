import { useState } from 'react';
import { PredictionForm } from './components/PredictionForm';
import { ResultsList } from './components/ResultsList';
import { ModelDetailsPanel } from './components/ModelDetailsPanel';
import axios from 'axios';
import { Activity } from 'lucide-react';

export type ModelSummary = {
  model: string;
  accuracy: number;
};

export type PredictionResponse = {
  id: string;
  results: ModelSummary[];
};

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [loading, setLoading] = useState(false);
  const [resultId, setResultId] = useState<string | null>(null);
  const [summaries, setSummaries] = useState<ModelSummary[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const handlePredict = async (data: Record<string, number>) => {
    setLoading(true);
    setResultId(null);
    setSummaries([]);
    setSelectedModel(null);
    try {
      const response = await axios.post<PredictionResponse>(`${API_URL}/api/predict`, data);
      setResultId(response.data.id);
      setSummaries(response.data.results);
    } catch (error) {
      console.error('Prediction failed:', error);
      alert('Prediction failed. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-slate-100 p-8 flex flex-col relative overflow-hidden">
      {/* Background decoration */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-primary-900/30 blur-[120px] rounded-full pointer-events-none" />
      
      <header className="mb-12 relative z-10 flex items-center justify-center gap-4">
        <div className="bg-primary-500/20 p-3 rounded-xl">
          <Activity className="w-8 h-8 text-primary-400" />
        </div>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-primary-400 to-purple-400 bg-clip-text text-transparent">
          Intelligent Diabetes Predictor
        </h1>
      </header>

      <main className="max-w-6xl mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-8 relative z-10">
        <div className="lg:col-span-5">
          <PredictionForm onSubmit={handlePredict} loading={loading} />
        </div>
        <div className="lg:col-span-7">
          {resultId && summaries.length > 0 ? (
            <ResultsList 
              summaries={summaries} 
              onSelectModel={setSelectedModel} 
              selectedModel={selectedModel}
            />
          ) : (
            <div className="glass-panel h-full min-h-[400px] flex items-center justify-center text-slate-400 flex-col gap-4">
              <Activity className="w-12 h-12 opacity-50" />
              <p>Fill out the form and click Predict to see model results.</p>
            </div>
          )}
        </div>
      </main>

      <ModelDetailsPanel 
        isOpen={!!selectedModel} 
        onClose={() => setSelectedModel(null)}
        model={selectedModel}
        resultId={resultId}
        apiUrl={API_URL}
      />
    </div>
  );
}

export default App;
